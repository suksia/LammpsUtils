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

NTASKS = int(os.environ['SLURM_NTASKS'])

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

    # run lammps
    logger.debug(f'Starting LAMMPS simulations...')
    study.run_lammps()

    # analyze data
    logger.debug(f'Analyzing simulation data...')
    study.analyze()

    # analyze data
    logger.debug(f'Plotting and saving simulation data...')
    study.save_data()

    logger.debug('Done.')

class Study:
    def __init__(self, input_yml: dict[str, dict]):
        pass

    def init_state(self):
        pass
                
    def build_directory(self):
        pass

    def run_lammps(self):
        pass

    def analyze(self):
        pass

    def save_data(self):
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
        self.dir = next_path(Path(input_yml['dir']) / f"{input_yml['defect']}_diffusion")

        self.name = self.input_yml['name']
        logger.debug(f'Starting study: {self.name}')

        # dictionaries
        self.state = {}
        self.data = {}       
        self.params = deepcopy(input_yml)

        # finish initializing
        self.init_state()

    def init_state(self):
        # add/modify some input parameters
        self.params['size_x'] = self.params['size'][0]
        self.params['size_y'] = self.params['size'][1]
        self.params['size_z'] = self.params['size'][2]

        if self.input_yml['defect'] == 'int':
            self.params['pd_fn'] = 'interstitial.in'
        elif self.input_yml['defect'] == 'vac':
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
            self.params['temp'] = temp
            template_dir = PKG_DIR / 'templates' / self.__class__.__name__

            for fp in template_dir.iterdir():
                # skip defining the input file for the opposite defect type 
                if fp.name == 'interstitial.in' or fp.name == 'vacancy.in':
                    if self.params['defect'] not in fp.name:
                        continue
                
                # define minimize.in as an empty file if not going to quench snapshots
                elif fp.name == 'minimize.in' and self.params['quench'] == False:
                    in_file = LammpsInput()
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
        """Continuously launch LAMMPS in parallel until the entire ensemble has been computed for each temperature."""
        # 0 = ready, 1 = running, 2 = counted
        members_dict = {mem_i: 0 for mem_i in range(self.input_yml['members'])}
        jobs_status = {temp: deepcopy(members_dict) for temp in self.sim_ids}

        # replace 0 with jobs as they're scheduled 
        jobs = deepcopy(jobs_status)
        
        def check_status(status: int, return_next=False, return_all=False):
            if return_next:
                next_found = False
                for temp, mem_dict in jobs_status.items():
                    for mem_i, stat in mem_dict.items():
                        if stat == status:
                            next_found = True
                            break
                    if next_found:
                        break
                if next_found:
                    return temp, mem_i
                else:
                    return None
            
            elif return_all:
                kw_pairs = []
                for temp, mem_dict in jobs_status.items():
                    for mem_i, stat in mem_dict.items():
                        if stat == status:
                            kw_pairs.append((temp, mem_i))
                return kw_pairs

            else:
                num_status = 0
                for mem_dict in jobs_status.values():
                    for stat in mem_dict.values():
                        if stat == status:
                            num_status += 1
                return num_status
        
        tot_num_jobs = len(self.sim_ids)*self.input_yml['members']

        # define seeds for velocity initial conditions
        seeds = set()
        while len(seeds) < tot_num_jobs:
            seeds.add(random.randint(0, 100000))
        seeds = list(seeds)
               
        # launch jobs until all have been counted
        while check_status(2) < tot_num_jobs:
            num_running, num_left = check_status(1), check_status(0)
            
            # poll running jobs to update their state if finished
            for temp, mem_i in check_status(1, return_all=True):
                job: LammpsJob = jobs[temp][mem_i]
                job.poll()
                if job.finished and not job.counted:
                    jobs_status[temp][mem_i] = 2
                    logger.debug(f'LAMMPS finished for T={temp} and member={mem_i}')

            # launch a job if possible
            if num_running < math.floor(NTASKS / self.input_yml['processors']) and num_running < num_left:
                temp, mem_i = check_status(0, return_next=True)

                job_dir = self.state[temp]['dir']/str(mem_i)

                # write input files
                for fn, lmpfile in self.state[temp]['input_files'].items():
                    # update seed in velocity initialization
                    for i, line in enumerate(lmpfile.lines): 
                        if 'velocity' in line:
                            seed_idx = self.sim_ids.index(temp)*10 + mem_i
                            vel_line = strip_split(line)
                            vel_line[-1] = seeds[seed_idx]
                            lmpfile.lines[i] = tilps(vel_line)
                    lmpfile.write_to_file(job_dir/fn)

                # run LAMMPS and save process
                jobs[temp][mem_i] = LammpsJob(job_dir, self.params['processors'])
                jobs_status[temp][mem_i] = 1

    def analyze(self):
        """Obtain the squared displacements and MSD for each temperature and method (self-diffusion, defect diffusion)."""
        # initialize data dictionary mem_id -> SD list, MSD -> np.array/list, t -> list
        members_dict = {mem_i: None for mem_i in range(self.input_yml['members'])}
        members_dict.update({'msd': None})

        temp_dict = {temp: deepcopy(members_dict) for temp in self.sim_ids}

        self.data.update({'self': deepcopy(temp_dict), 'defect': deepcopy(temp_dict)})

        # self-diffusion data first
        for temp in self.sim_ids:
            for mem_i in range(self.input_yml['members']):
                sq_file = LammpsLog(self.state[temp]['dir'] / str(mem_i) / 'diffusion.log')
                sq_dis = sq_file.data['c_msdvar[4]']

                # squared displacement
                self.data['self'][temp][mem_i] = sq_dis
                
                # MSD as a stack of squared displacement curves, to be averaged column-wise 
                if self.data['self'][temp]['msd'] is None:
                    self.data['self'][temp]['msd'] = np.array(sq_dis)
                else:
                    self.data['self'][temp]['msd'] = np.vstack((self.data['self'][temp]['msd'], np.array(sq_dis)))
            
            # compute MSD for temperature
            logger.debug(f'Computing mean squared displacement for T={temp}')
            msd = []
            if self.input_yml['members'] > 1:
                for col in range(len(self.data['self'][temp]['msd'][0, :])):
                    msd.append(float(np.mean(self.data['self'][temp]['msd'][:, col])))
            else:
                msd = self.data['self'][temp]['msd'].tolist()
                logger.debug('WARNING: Ensemble consists of only 1 member. Do not trust the MSD!')
            self.data['self'][temp]['msd'] = msd

        # save time in ns using previous sq_file loaded
        self.data.update({'t': [step*self.input_yml['timestep'] / 1000 for step in sq_file.data['Step']]})
        
    def save_data(self):
        """Plot curves and write out data."""
        # plot equilibriation curves
        logger.debug(f'Plotting equilibriation curves...')
        for temp in self.sim_ids:
            for mem_i in range(self.input_yml['members']):
                equil_log = LammpsLog(self.state[temp]['dir'] / str(mem_i) / 'equil.log')
                equil_log.plot_values()

        # plot squared displacement curves together
        logger.debug(f'Plotting displacement curves...')
        for temp in self.sim_ids:
            for mem_i in range(self.input_yml['members']):
                plt.plot(self.data['t'], self.data['self'][temp][mem_i])
            plt.xlabel('Time [ns]')
            plt.ylabel('Squared Displacement [$Å^2$]')
            plt.savefig(self.state[temp]['dir'] / 'self_sd.png')
            plt.close()

        # save msd data
        logger.debug(f'Writing MSD data...')
        for temp in self.sim_ids:
            with open(self.state[temp]['dir'] / 'self_msd.txt', 'w') as msd_file:
                msd_file.write('time[ns]\t\t msd[Å2]\n')
                for i in range(len(self.data['t'])):
                    msd_file.write(f"{self.data['t'][i]}\t\t {self.data['self'][temp]['msd'][i]}\n")

        # plot MSD for each temperature
        logger.debug(f'Plotting MSD curves...')
        for temp in self.sim_ids:
            plt.plot(self.data['t'], self.data['self'][temp]['msd'])
            plt.xlabel('Time [ns]')
            plt.ylabel('Mean Squared Displacement [$Å^2$]')
            plt.savefig(self.state[temp]['dir'] / 'self_msd.png')
            plt.close()
        
        # compare msd for each temp
        logger.debug(f'Plotting MSD temperature comparison...')
        for temp in self.sim_ids:
            plt.plot(self.data['t'], self.data['self'][temp]['msd'], label=f'{temp}K')
        plt.legend()
        plt.xlabel('Time [ns]')
        plt.ylabel('Mean Squared Displacement [$Å^2$]')
        plt.savefig(self.dir / 'self_msd_by_T.png')
        plt.close()


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
        poll =self.process.poll()
        if poll == 0:
            self.finished = True
            self.outfile.close()
            
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