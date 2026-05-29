import argparse, yaml, logging, sys, subprocess, os
from pathlib import Path
from utils import next_path, strip_split, tilps
from copy import deepcopy

# basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('LammpsUtils')

# define environment variable so LAMMPS can find potentials without needing a valid relative path
os.environ['LAMMPS_POTENTIALS'] = (Path(__file__).parent.parent / 'potentials').as_posix()
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
    study_type = input_params['study']['type']
    study: Study = study_registry[study_type](input_params)
    logger.debug(f'Initialized study type {study_type}')

class Study:
    def __init__(self, input_yml: dict[str, dict]):
        self.input_yml = input_yml
        self.params = input_yml['study']
        self.sim_params = input_yml['simulations']
        self.parent_dir = Path(input_yml['study']['dir'])
        self.dir = None

        self.name = self.params['name']
        logger.debug(f'Starting study: {self.name}')

        self.state, self.sim_ids, self.skip_sim_ids = {}, [], []
        logger.debug(f'Initializing state...')
        self.init_state()

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
class MeanSquareDisplacement(Study):
    def init_state(self):
        self.sim_ids = self.params['temperatures']
        self.state = dict.fromkeys(self.sim_ids, {})

        # build directory
        self.dir = next_path(self.parent_dir / 'individual')
        self.dir.mkdir()

        for sim_id in self.sim_ids:
            subdir = self.dir / f'{sim_id}K'
            subdir.mkdir()
            self.state[sim_id]['dir'] = subdir
        logger.debug(f'Built directory tree at {self.dir}')

        # define input files
        equil_order = ['system', 'potential', 'equil']
        msd_order = ['potential', 'msd']
        for sim_id in self.sim_ids:
            # equilibriation 
            equil_file = LammpsFile(
                content_strs=[self.sim_params[key] for key in equil_order],
                write_restart=self.state[sim_id]['dir'] / 'equil.restart'
            )
            equil_file.update_temp(float(sim_id))
            self.state[sim_id]['equil_file'] = equil_file
            logger.debug(f'Defined equilibriation input file for temperature {sim_id}')

            # msd
            msd_file = LammpsFile(
                content_strs=[self.sim_params[key] for key in msd_order],
                read_restart=self.state[sim_id]['dir'] / 'equil.restart'
            )
            msd_file.update_temp(float(sim_id))
            self.state[sim_id]['msd_file'] = msd_file
            logger.debug(f'Defined MSD input file for temperature {sim_id}')

    def run_lammps(self):
        for sim_id in self.sim_ids:
            # equilibriate
            sim_dir = self.state[sim_id]['dir']
            self.state[sim_id]['equil_file'].write_to_file(sim_dir / 'equil.in')

            equil_out = open(sim_dir / 'equil.out', 'w')
            lammps_cmd = ['srun', '--export=ALL', 'lmp', '-in', 'equil.in']
            logger.debug(f'Launching LAMMPS')
            subprocess.run(lammps_cmd, cwd=sim_dir, stdout=equil_out, stderr=subprocess.STDOUT)

            # msd
            self.state[sim_id]['msd_file'].write_to_file(sim_dir / 'msd.in')

            msd_out = open(sim_dir / 'msd.out', 'w')
            lammps_cmd = ['srun', '--export=ALL', 'lmp', '-in', 'msd.in']
            logger.debug(f'Launching LAMMPS')
            subprocess.run(lammps_cmd, cwd=sim_dir, stdout=msd_out, stderr=subprocess.STDOUT)

class LammpsFile:
    def __init__(self, file_path: Path = None, content_strs: list[str] = None, write_restart: Path = None, read_restart: Path = None):
        self.lines = []
        self.path = None

        self.write_restart = write_restart
        self.read_restart = read_restart

        if content_strs:
            self.load_from_strings(content_strs)

    def load_from_strings(self, contents_strs: list[str]):
        if self.read_restart:
            self.lines.append(f'read_restart {self.read_restart}')
            self.lines[-1] += '\n'

        for s in contents_strs:
            for l in s.split('\n'):
                self.lines.append(l)
            self.lines[-1] += '\n'
        
        if self.write_restart:
            self.lines.append(f'write_restart {self.write_restart}')

    def write_to_file(self, dest_path: Path, overwrite_path=False):
        with open(dest_path, 'w') as d:
            for l in self.lines:
                d.write(l+'\n')
            logger.debug(f'{LammpsFile.__name__}: wrote lines to {dest_path}')
        if overwrite_path:
            self.path = dest_path
            logger.debug(f'{LammpsFile.__name__}: updated current path to {dest_path}')

    def update_temp(self, temp: float):
        temp = f'{temp:.2f}'
        for i, line in enumerate(self.lines):
            line = strip_split(line)
            if 'velocity' in line:
                line[3] = temp
            elif 'nvt' in line:
                line[5] = temp
                line[6] = temp
            else:
                continue
            self.lines[i] = tilps(line)

class LammpsLog:
    pass
    
main()