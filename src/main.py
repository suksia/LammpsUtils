import argparse, yaml, logging, sys, subprocess, os, shutil
from pathlib import Path
from utils import *
from copy import deepcopy
import matplotlib.pyplot as plt
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.io import import_file, export_file
import numpy as np
import re

# basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('LammpsUtils')
logging.getLogger("matplotlib").setLevel(logging.FATAL)

# define environment variable so LAMMPS can find potentials without needing a valid relative path
PKG_DIR = Path(__file__).parent.parent
os.environ['LAMMPS_POTENTIALS'] = (PKG_DIR / 'potentials').as_posix()
logger.debug(f'Defined LAMMPS_POTENTIALS environment variable')

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
class VacancyDiffusion(Study):
    def __init__(self, input_yml: dict[str, dict]):
        self.input_yml = input_yml
        self.dir = next_path(Path(input_yml['dir']) / 'vac_diffusion')
        self.file_order = None

        self.name = self.input_yml['name']
        logger.debug(f'Starting study: {self.name}')

        self.state, self.sim_ids, self.skip_sim_ids = {}, [], []
        logger.debug(f'Initializing state...')
        self.init_state()

    def init_state(self):
        self.sim_ids = self.input_yml['temperatures']
        self.file_order = ['main.in', 'equil.in', 'diffusion.in', 'minimize.in']

        self.state = dict.fromkeys(self.sim_ids)

        # initialize each sim with a new dict (dict.fromkeys initializes them with the same value reference)
        for key in self.state.keys():
            self.state[key] = {'input_files': {}}

        # add/modify some parameters that are implicit
        file_params = deepcopy(self.input_yml)

        file_params['size_x'] = file_params['size'][0]
        file_params['size_y'] = file_params['size'][1]
        file_params['size_z'] = file_params['size'][2]

        file_params['equil'] = unprefix(file_params['equil'])
        file_params['vac_equil'] = unprefix(file_params['vac_equil'])
        file_params['diffusion'] = unprefix(file_params['diffusion'])
        file_params['snapshot'] = unprefix(file_params['snapshot'])
        file_params['num_snapshots'] = int(file_params['diffusion'] / file_params['snapshot'])

        # loop through temperatures and load+update each input files
        for temp in self.sim_ids:
            file_params['temp'] = temp

            input_files = {}
            for fn in self.file_order:
                # load input file lines
                in_file = LammpsInput(file_path = PKG_DIR / 'templates' / self.__class__.__name__ / fn)
                in_file.add_params(file_params)

                # save file objects
                self.state[temp]['input_files'][fn] = in_file
            
            logger.debug(f'Defined input files for temperature {temp}')

    def build_directory(self):
        self.dir.mkdir()

        for temp in self.sim_ids:
            subdir = self.dir / f'{temp}K'
            subdir.mkdir()
            self.state[temp].update({'dir' : subdir})

    def run_lammps(self):
        for temp in self.sim_ids:
            sim_dir = self.state[temp]['dir']

            # write input files
            for fn, lmpfile in self.state[temp]['input_files'].items():
                lmpfile.write_to_file(sim_dir / fn)
            
            # run LAMMPS
            lmp_out = open(sim_dir / 'lmp.out', 'w')
            lammps_cmd = ['srun', '--export=ALL', 'lmp', '-in', self.file_order[0]]
            logger.debug(f'Launching LAMMPS')
            subprocess.run(lammps_cmd, cwd=sim_dir, stdout=lmp_out, stderr=subprocess.STDOUT)

            # obtain vacancy positions using Wigner-Seitz cell analysis on quenched snapshots
            pipeline = import_file(sim_dir / 'minimize.dump')

            # outputs sites as particles DataCollection with "Occupancy" property
            ws = WignerSeitzAnalysisModifier()
            pipeline.modifiers.append(ws)

            # custom modifier to obtain a vacancy position for each snapshot
            def modify(frame, data):
                # per-site occupancy (0 = vacancy, 1 = normal site, 2 = interstitial)
                occupancies = data.particles['Occupancy']

                # add a boolean "Selection" property which will be 1 for sites with a vacancy
                selection = data.particles_.create_property('Selection')
                selection[...] = occupancies == 0

                # add a data attribute for the current frame which is the vacancy position
                vacancy_idx = np.nonzero(selection)
                data.attributes['VacancyCount'] = np.sum(selection)
                data.attributes['VacancyPosition'] = data.particles.positions[vacancy_idx]

            # write vacancy positions
            pipeline.modifiers.append(modify)
            export_file(
                pipeline, 
                sim_dir / 'vacancies.txt', 
                'txt/attr',
                columns = ['Timestep', 'VacancyCount', 'VacancyPosition'],
                multiple_frames = True)

            # compute square displacement as a function of time
            frames = [frame for frame in pipeline.frames]
            t, msd = [0], [0.0]
            ref_vac_pos = frames[1].attributes['VacancyPosition']

            for frame in frames[2:]:
                t_step =  frame.attributes['Timestep']
                vac_pos = frame.attributes['VacancyPosition']

                if len(vac_pos) > 1:
                    raise Warning(f'[timestep={t_step}] More than 1 vacancy detected! Quenching failed.')
                vac_pos = vac_pos[0]
                
                t.append(t_step*float(self.input_yml['timestep'])/1000)
                msd.append(np.linalg.norm(vac_pos - ref_vac_pos)**2)

            # plot and save the square displacement for a single microstate
            plt.plot(t, msd)
            plt.xlabel('Time [ns]')
            plt.ylabel('Squared Displacement')
            plt.title(sim_dir)
            plt.savefig(sim_dir / 'msd.png')
            plt.close()

class LammpsFile:
    def __init__(self):
        pass

class LammpsInput(LammpsFile):
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
            if line[0] == 'Per':
                start.append(i+1)
            elif line[0] == 'Loop':
                stop.append(i-1)

        # determine name of each column in thermo data (shouldn't change within the same log file)
        data_labels = None
        for i in range(len(start)):
            new_data_labels = strip_split(self.lines[i])
            if data_labels is None:
                data_labels = new_data_labels
            else:
                assert data_labels == new_data_labels, \
                    f'Thermo data labels changed between runs for log file at {self.path}'
        
        # load the data as one contiguous list
        self.data: dict[str, list] = dict.fromkeys(data_labels, [])
        for i in range(len(start)):
            for line in self.lines[start[i+1]:stop[i]]:
                line = strip_split(line)
                for j, val in enumerate(line):
                    self.data[data_labels[j]].append(float(val))

    def plot_values(self):
        try:
            x = self.data['Step']
        except:
            raise KeyError(f'`Step` must be one of the data labels for log file at {self.path}')
        
        y_labels = list(self.data.keys)
        y_labels.pop('Step')

        for y_lab in y_labels:
            plt.plot(x, self.data[y_lab])
            plt.xlabel('Timestep')
            plt.ylabel(y_lab)
            plt.savefig(self.path.parent / f'{y_lab}.png')
            plt.close()

main()