import argparse, yaml, logging, sys, subprocess, os, shutil, re, random, math
from pathlib import Path
from utils import *
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.io import import_file, export_file
import numpy as np

# basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('LammpsUtils')
logging.getLogger("matplotlib").setLevel(logging.FATAL)

# define environment variable so LAMMPS can find potentials without needing a valid relative path
PKG_DIR = Path(__file__).parent.parent
os.environ['LAMMPS_POTENTIALS'] = (PKG_DIR / 'potentials').as_posix()
logger.debug(f'Defined LAMMPS_POTENTIALS environment variable')

NTASKS = os.environ['SLURM_NTASKS']

def main():
    # load input file
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input file')
    args = parser.parse_args()

    input_fp = Path(args.input).resolve()
    assert input_fp.exists(), f'[{input_fp}] File does not exist'

    # load user input
    with open(input_fp, 'r') as f:
        input_params: dict = yaml.safe_load(f)
    logger.debug(f'Loaded input file {input_fp}')

    # initialize a study
    study_type = input_params['type']
    study: Study = study_registry[study_type](input_params)
    logger.debug(f'Initialized study type {study_type}')

    # build directory tree and copy in input file
    study.build_directory()
    logger.debug(f'Built directory tree at {study.dir}')
    shutil.copy(input_fp, study.dir)

    # run vasp
    logger.debug(f'Starting LAMMPS simulations...')
    study.run_lammps()

class Study:
    def __init__(self, input_yml: dict[str, dict]):
        pass

    def init_state(self):
        pass
                
    def build_directory(self):
        pass

    def run_lammps(self):
        pass

study_registry: dict[str, Study] = {}
def register_study(cls):
    """Registry enrollment so that Study subclasses can be instantiated by string name."""
    study_registry[cls.__name__] = cls
    return cls

@register_study
class PointDefectDiffusion(Study):
    def __init__(self, input_yml: dict[str, dict]):
        self.input_yml = input_yml
        self.dir = next_path(Path(input_yml['dir']) / f'{input_yml['defect']}_diffusion')

        self.name = self.input_yml['name']
        logger.debug(f'Starting study: {self.name}')

        # add/modify some input parameters
        self.params = deepcopy(input_yml)

        self.params['size_x'] = self.params['size'][0]
        self.params['size_y'] = self.params['size'][1]
        self.params['size_z'] = self.params['size'][2]

        if input_yml['defect'] == 'int':
            self.params['pd_fn'] = 'interstitial.in'
        elif input_yml['defect'] == 'vac':
            self.params['pd_fn'] = 'vacancy.in'

        self.params['pd_x'] = self.params['position'][0]
        self.params['pd_y'] = self.params['position'][1]
        self.params['pd_z'] = self.params['position'][2]

        self.params['equil'] = unprefix(self.params['equil'])
        self.params['diffusion'] = unprefix(self.params['diffusion'])
        self.params['snapshot'] = unprefix(self.params['snapshot'])
        self.params['num_snapshots'] = int(self.params['diffusion'] / self.params['snapshot'])

        # initialize state starting with input files for each temperature
        self.sim_ids = self.input_yml['temperatures']
        if type(self.sim_ids) != list:
            self.sim_ids = [self.sim_ids]

        self.state = {key: {'input_files': {}} for key in self.sim_ids}

        for temp in self.sim_ids:
            template_dir = PKG_DIR / 'templates' / self.__class__.__name__

            for fp in template_dir.iterdir():
                # define minimize.in as an empty file if not going to quench snapshots
                if fp == 'minimize.in' and self.params['quench'] == False:
                    in_file = LammpsInput()
                
                # skip defining the input file for the opposite defect type 
                elif fp == 'interstitial.in' or fp == 'vacancy.in':
                    if self.params['defect'] not in fp.name:
                        continue

                else:
                    in_file = LammpsInput(file_path=fp)
                    in_file.add_params(self.params)

                # save file object
                self.state[temp]['input_files'][fp.name] = in_file
            
            logger.debug(f'Defined input files for temperature {temp}')

    def build_directory(self):
        self.dir.mkdir()

        for temp in self.sim_ids:
            subdir = self.dir / f'{temp}K'
            subdir.mkdir()

            for m in range(self.params['members']):
                member_subdir = subdir / str(m)
                member_subdir.mkdir()

            self.state[temp].update({'dir' : subdir})

    def run_lammps(self):
        for temp in self.sim_ids:
            sim_dir = self.state[temp]['dir']

            # define seeds for velocity initial conditions
            seeds = set()
            while len(seeds) < self.input_yml['members']:
                seeds.add(random.randint(0, 100000))
            seeds = list(seeds)

            sq_dis = None
            num_running, num_left = 0, copy(self.input_yml['members'])

            # launch jobs until all are finished
            jobs: dict[int, LammpsJob] = dict.fromkeys(range(self.input_yml['members']))
            while num_left > 0:
                # check for finished processes
                for member_i, job in jobs.items():
                    if job is None:
                        continue
                    elif job.poll():
                        if job.finished and not job.counted:
                            num_left -= 1
                            num_running -= 1
                            logger.debug(f'LAMMPS finished for member {member_i}')
                
                # launch a job if possible
                if (num_running+1)*self.input_yml['processors'] < NTASKS and num_running < num_left:
                    # next member = next index that is None
                    for key, val in jobs.items():
                        if val is None:
                            member_i = key
                            break
                    
                    # write input files
                    for fn, lmpfile in self.state[temp]['input_files'].items():
                        # update seed in velocity initialization
                        for i, line in enumerate(lmpfile.lines): 
                            if 'velocity' in line:
                                vel_line = strip_split(line)
                                vel_line[-1] = seeds[member_i]
                                lmpfile.lines[i] = tilps(vel_line)
                            lmpfile.write_to_file(sim_dir / str(member_i) / fn)

                    # run LAMMPS
                    jobs.update({member_i: LammpsJob(sim_dir/str(member_i), self.params['processors'])})
                    num_running += 1

class LammpsJob:
    def __init__(self, member_dir: Path, num_processors: int):
        self.member_dir = member_dir
        self.outfile = open(member_dir / 'lmp.out', 'w')
        self.finished = False
        self.counted = False

        self.lammps_cmd = [
            'srun', 
            f'--ntasks={num_processors}', 
            '--export=ALL', 
            'lmp', 
            '-in', 
            'main.in']
        
        self.process = subprocess.Popen(self.lammps_cmd, cwd=self.member_dir, stdout=self.outfile, stderr=subprocess.STDOUT)
        logger.debug(f'Launching LAMMPS for T={self.member_dir.parent.name} and member={self.member_dir.name}...')

    def poll(self):
        if self.process.poll():
            self.outfile.close()
            self.finished = True

class LammpsInput:
    def __init__(self, file_path: Path = None, content_str: str = None):
        self.lines = []
        self.last_read_path = None
        self.last_write_path = None

        if file_path:
            self.load_from_file(file_path)
        elif content_str:
            self.load_from_string(content_str)
    
    def load_from_file(self, read_path: Path):
        with open(read_path, 'r') as f:
            self.lines = f.readlines()
        self.last_read_path = deepcopy(read_path)
        logger.debug(f'{self.__class__.__name__}: read lines from {self.last_read_path}')

    def load_from_string(self, contents_str: str):
        for l in contents_str.split('\n'):
            self.lines.append(l)

    def write_to_file(self, write_path: Path):
        with open(write_path, 'w') as d:
            for l in self.lines:
                d.write(l)
        self.last_write_path = deepcopy(write_path)
        logger.debug(f'{self.__class__.__name__}: wrote lines to {write_path}')

    def add_params(self, params: dict):
        params = deepcopy(params)

        # loop through each string in each line and replace ?param? with params[param]
        for i, line in enumerate(self.lines):
            while '?' in line:
                # pairs of indices corresponding to the starting ? and stopping ? for ?param?
                p_idcs = [j for j, c in enumerate(line) if c == '?']
                start, stop = p_idcs[0]+1, p_idcs[1]

                # replace with the actual param val
                p = line[start:stop]
                assert p in params.keys(), f'LammpsInput: parameter {p} not a key in params dictionary'
                line = re.sub(f'\?{p}\?', str(params[p]), line)

            # update line
            self.lines[i] = line

class LammpsLog:
    def __init__(self, file_path: Path):
        self.path = file_path
        self.lines = []

        with open(self.path, 'r') as log:
            self.lines = log.readlines()
        
        # determine which lines correspond to thermo data
        start, stop = [], []
        for i, line in enumerate(self.lines):
            line = strip_split(line)
            if len(line) == 0:
                continue
            elif line[0] == 'Per':
                start.append(i+1)
            elif line[0] == 'Loop':
                stop.append(i-1)

        # determine name of each column in thermo data (shouldn't change within the same log file)
        data_labels = None
        for i in start:
            new_data_labels = strip_split(self.lines[i])
            if data_labels is None:
                data_labels = new_data_labels
            else:
                assert data_labels == new_data_labels, \
                    f'Thermo data labels changed between runs for log file at {self.path}'
        
        # load the data as one contiguous list
        self.data: dict[str, list] = dict.fromkeys(data_labels)
        for key in self.data.keys():
            self.data[key] = []

        for i in range(len(start)):
            for line in self.lines[start[i]+1:stop[i]]:
                line = strip_split(line)
                for j, val in enumerate(line):
                    self.data[data_labels[j]].append(float(val))

    def plot_values(self):
        try:
            x = self.data['Step']
        except:
            raise KeyError(f'`Step` must be one of the data labels for log file at {self.path}')
        
        y_labels = list(self.data.keys())
        y_labels.pop(y_labels.index('Step'))

        for y_lab in y_labels:
            plt.plot(x, self.data[y_lab])
            plt.xlabel('Timestep')
            plt.ylabel(y_lab)
            plt.savefig(self.path.parent / f'{y_lab}.png')
            plt.close()

main()