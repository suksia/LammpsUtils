import logging, random, math, os, time, subprocess
from copy import copy, deepcopy
from pathlib import Path
from lammps_file import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.io import import_file, export_file
from typing import Any

logger = logging.getLogger('LammpsUtils')
logging.getLogger("matplotlib").setLevel(logging.FATAL)

PKG_DIR = Path(__file__).parent.parent
try:
    NTASKS = int(os.environ['SLURM_NTASKS'])
except:
    NTASKS = 1

class LmpJob:
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

class Study:
    """Base class for studies in LAMMPS."""
    def __init__(self, input_yml: dict[str, dict]):
        # input_yml serves as a "master copy", params is updated as necessary
        self.input_yml = input_yml
        self.params = deepcopy(input_yml)

        self.name = self.input_yml['name']
        self.dir = None
        self.restart = None
        self.templates_dir = PKG_DIR / 'templates' / self.__class__.__name__

        # containers defined in subclasses
        self.state = {}
        self.sim_ids = []
        self.data = {}

        # finish initializing
        self.init_state()

    def init_state(self):
        """Populates a state dictionary with simulation parameters and input files for running LAMMPS."""
        pass

    def build_directory(self):
        """Loads a restart file if present, otherwise a new full study directory is created."""
        # check directory for restart file first
        input_dir = Path(self.input_yml['dir'])
        assert input_dir.exists(), f'Directory {input_dir} does not exist'

        for file in input_dir.iterdir():
            if file.name == 'LammpsUtils.restart':
                logger.debug(f'Restart file found. Reading contents...')
                self.dir = input_dir
                
                self.restart: dict[int, dict[int, list[int]]] = {}
                with open(input_dir/'LammpsUtils.restart', 'r') as rf:
                    for line in rf.readlines():
                        line = strip_split(line)
                        if len(line) == 2:
                            conf_i, sim_i, mem_i = None, int(line[0]), int(line[1])
                        elif len(line) == 3:
                            conf_i, sim_i, mem_i = int(line[0]), int(line[1]), int(line[2])
                        else:
                            continue
                        
                        if len(line) == 2:
                            if sim_i not in self.restart.keys():
                                self.restart.update({sim_i: [mem_i]})
                            else:
                                self.restart[sim_i].append(mem_i)

                        elif len(line) == 3:
                            if conf_i not in self.restart.keys():
                                self.restart.update({conf_i: {sim_i: [mem_i]}})
                            else:
                                self.restart[conf_i][sim_i].append(mem_i)
                break

        if self.dir is None:
            self.restart = False
            self.dir = next_path(Path(self.input_yml['dir']) / self.name)

    def run_lammps(self, state: dict):
        """Continuously launch LAMMPS in parallel until all simulations and members have finished running."""
        # replace 0 with jobs as they're scheduled 
        jobs = {sim_i: {mem_i: 0 for mem_i in state[sim_i].keys()} for sim_i in self.sim_ids}

        def check_status(status: int, return_next=False, return_all=False):
            """Helper function for determining which jobs to run next based on status (0 = ready, 1 = running, 2 = finished)."""
            if return_next:
                next_found = False
                for sim_i in self.sim_ids:
                    for mem_i, mem_dict in state[sim_i].items():
                        if mem_dict['status'] == status:
                            next_found = True
                            break
                    if next_found:
                        break
                if next_found:
                    return sim_i, mem_i
                else:
                    return None
            
            elif return_all:
                kw_pairs = []
                for sim_i in self.sim_ids:
                    for mem_i, mem_dict in state[sim_i].items():
                        if mem_dict['status'] == status:
                            kw_pairs.append((sim_i, mem_i))
                return kw_pairs

            else:
                num_status = 0
                for sim_i in self.sim_ids:
                    for mem_dict in state[sim_i].values():
                        if mem_dict['status'] == status:
                            num_status += 1
                return num_status
        
        tot_num_jobs = 0
        for sim_i in self.sim_ids:
            tot_num_jobs += len(state[sim_i])

        # make sure while loop does not run forever due to insufficient number of processors
        ntasks_per_job = math.floor(NTASKS / self.input_yml['processors'])
        if ntasks_per_job < 1:
            raise ValueError(f"{NTASKS} processors available. Not enough for a single job ({self.input_yml['processors']})")
        
        # replace restart file with a copy that can be updated
        restart_file = open(self.dir / 'LammpsUtils.restart', 'w')

        # launch jobs until all have been counted
        while check_status(2) < tot_num_jobs:
            num_running, num_left = check_status(1), check_status(0)
            
            # poll running jobs to update their state if finished
            for sim_i, mem_i in check_status(1, return_all=True):
                job: LmpJob = jobs[sim_i][mem_i]
                job.poll()
                if job.finished and not job.counted:
                    state[sim_i][mem_i]['status'] = 2
                    restart_file.write(f'{sim_i}\t{mem_i}\n')
                    logger.debug(f'LAMMPS finished for sim={sim_i} and member={mem_i}')

            # launch a job if possible
            if num_running < ntasks_per_job and num_running < num_left:
                sim_i, mem_i = check_status(0, return_next=True)
                
                if self.restart:
                    if sim_i in self.restart.keys():
                        if mem_i in self.restart[sim_i]:
                            state[sim_i][mem_i]['status'] = 2
                            restart_file.write(f'{sim_i}\t{mem_i}\n')
                            logger.debug(f'LAMMPS has already been run for sim={sim_i} and member={mem_i}. Skipping it')
                            continue

                job_dir = state[sim_i][mem_i]['dir']

                # write input files
                for fn, lmpfile in state[sim_i][mem_i]['input_files'].items():
                    lmpfile.write_to_file(job_dir/fn)

                # run LAMMPS and save process
                jobs[sim_i][mem_i] = LmpJob(job_dir, self.params['processors'])
                state[sim_i][mem_i]['status'] = 1
        
        restart_file.close()

    def analyze(self):
        """Analyze data from out, log, and dump files specific to the study."""
        pass

    def save_data(self):
        """Write output files and visualize analysis data."""
        pass

study_registry: dict[str, Study] = {}
def register_study(cls):
    """Registry enrollment so that Study subclasses can be instantiated by string name."""
    study_registry[cls.__name__] = cls
    return cls

@register_study
class GenerateConfigurations(Study):
    def init_state(self):
        self.sim_ids = ['runs']
        self.state.update({'runs': {mem_i: {'input_files': {}, 'status': 0, 'dir': None} for mem_i in range(self.params['members'])}})
        
        # update params common to all members/configurations
        self.params.update({
            'elements': tilps(list(self.input_yml['composition'].keys())),
            'temp': self.input_yml['temperature'],
            'equil': unprefix(self.input_yml['equil']),
            'mc_freq': unprefix(self.input_yml['mc'][0]),
            'mc_attempts': unprefix(self.input_yml['mc'][1]),
            'mc_thermo_freq': max(1, int(unprefix(0.25*self.input_yml['mc'][1]))),
            'mc': unprefix(self.input_yml['mc'][2]),
            'snapshot': unprefix(self.input_yml['snapshot'])
        })

        # define seeds for atom/swap RNG 
        mc_seeds = create_seeds(self.params['members'])

        # generate configurations and add input files
        for mem_i in range(self.params['members']):
            self.params.update({'mc_seed': mc_seeds[mem_i]})
            
            main_in = LmpInput(file_path=self.templates_dir/'main.in')
            main_in.add_params(self.params)

            struct_in = LmpStructure(self.params)
            self.state['runs'][mem_i]['input_files'].update({'config.in': struct_in, 'main.in': main_in})

    def build_directory(self):
        super().build_directory()
        self.dir.mkdir(exist_ok=True)
        
        runs_dir: Path = self.dir / 'runs'
        runs_dir.mkdir(exist_ok=True)

        for mem_i in range(self.input_yml['members']):
            subdir = runs_dir / str(mem_i)
            subdir.mkdir(exist_ok=True)
            self.state['runs'][mem_i].update({'dir' : subdir})

        self.dataset_dir = self.dir / 'dataset'
        self.dataset_dir.mkdir(exist_ok=True)

    def run_lammps(self):
        super().run_lammps(self.state)

        # convert final configuration dumps into LAMMPS input files and save to dataset folder
        for mem_i in range(self.params['members']):
            dump = LmpDump(file_path=self.state['runs'][mem_i]['dir']/'final.dump')
            dump.write_structure_file(self.dataset_dir/f'config_{mem_i}.lmp', [el for el in self.input_yml['composition'].keys()])

@register_study
class PointDefectInsertion(Study):
    def init_state(self):
        # setup containers
        self.sim_ids = ['runs']
        self.state.update({'runs': {mem_i: {'input_files': {}, 'status': 0, 'dir': None} for mem_i in range(self.params['members'])}})

        # add elements parameter for defining potential
        self.params.update({'elements': tilps(list(self.input_yml['composition'].keys()))})

        # define input files for each member
        for mem_i in range(self.params['members']):
            main_in = LmpInput(file_path=self.templates_dir/'main.in')
            main_in.add_params(self.params)
            self.state['runs'][mem_i]['input_files'].update({'main.in': main_in})

            pristine_struct = LmpStructure(self.params)
            self.state['runs'][mem_i]['input_files'].update({'pristine.in': pristine_struct})
    
            if self.input_yml['defect'] == 'vac':
                def_type = 'vac'
                def_species = None
                def_orientation = None
            else:
                def_type = self.input_yml['int_type']
                def_species = self.input_yml['int_species']
                def_orientation = str(self.input_yml['int_orient'])

            defective_struct = deepcopy(pristine_struct)
            defective_struct.insert_point_defect(def_type, def_species, def_orientation)
            self.state['runs'][mem_i]['input_files'].update({'defective.in': defective_struct})

    def run_lammps(self):
        super().run_lammps(self.state)

    def build_directory(self):
        super().build_directory()
        self.dir.mkdir(exist_ok=True)
        
        # self.state[sim_id] dict should only have keys that are member indices
        runs_dir: Path = self.dir / 'runs'
        runs_dir.mkdir(exist_ok=True)

        for mem_i in range(self.input_yml['members']):
            subdir = runs_dir / str(mem_i)
            subdir.mkdir(exist_ok=True)
            self.state['runs'][mem_i].update({'dir' : subdir})

    def analyze(self):
        # setup container
        self.data.update({'pristine_e': [], 'defective_e': [], 'insertion_e': []})
        
        # read in potential energy from last thermo output
        for mem_i in range(self.input_yml['members']):
            subdir = self.state['runs'][mem_i]['dir']

            pris_log = LmpLog(file_path=subdir/'pristine.log')
            pris_e = pris_log.data['PotEng'][-1]

            def_log = LmpLog(file_path=subdir/'defective.log')
            def_e = def_log.data['PotEng'][-1]

            self.data['pristine_e'].append(pris_e)
            self.data['defective_e'].append(def_e)
            self.data['insertion_e'].append(pris_e-def_e)

        # bin energy data
        self.data.update({'insertion_histogram': np.histogram(self.data['insertion_e'], bins=40)})
    
    def save_data(self):
        # write the energies out
        with open(self.dir/'energies.out', 'w') as e:
            for mem_i in range(self.input_yml['members']):
                line = f"{mem_i:<5} {self.data['pristine_e'][mem_i]:<15.5f} {self.data['defective_e'][mem_i]:<15.5f} {self.data['insertion_e'][mem_i]:<15.5f}"
                e.write(line+'\n')
        
        # plot energy histogram
        y, x = self.data['insertion_histogram']
        plt.bar(x[:-1], y, linewidth=1, edgecolor='navy', width=np.diff(x))
        plt.xlabel('Insertion Energy [eV]')
        plt.ylabel('Frequency')
        plt.savefig(self.dir/'insertion_histo.png', bbox_inches="tight")
        plt.close()

@register_study
class PointDefectDiffusion(Study):
    """Samples configurations and computes the migration energy for each one by diffusing point defects."""
    def init_state(self):
        # setup containers
        self.sim_ids = self.input_yml['temperatures']
        if type(self.sim_ids) != list:
            self.sim_ids = [self.sim_ids]

        # update common parameters
        self.params['equil'] = unprefix(self.input_yml['equil'])
        self.params['diffusion'] = unprefix(self.input_yml['diffusion'])
        self.params['snapshot'] = unprefix(self.input_yml['snapshot'])
        self.params['num_snapshots'] = int(self.input_yml['diffusion'] / self.input_yml['snapshot'])

        # define common input_files
        main_in = LmpInput(file_path=self.templates_dir/'main.in')
        main_in.add_params(self.params)

        diffusion_in = LmpInput(file_path=self.templates_dir/'diffusion.in')
        diffusion_in.add_params(self.params)

        if self.input_yml['quench'] == False:
            quench_in = LmpInput()
        else:
            quench_in = LmpInput(file_path=self.templates_dir/'quench.in')
            quench_in.add_params(self.params)

        # self.state has an additional layer to iterate over configurations
        for conf_i in range(self.input_yml['configurations']):
            conf_dict = {sim_i: {mem_i: {'input_files': {}, 'status': 0, 'dir': None} for mem_i in range(self.params['members'])} for sim_i in self.sim_ids}

            # sample a configuration and insert a point defect
            struct_in = LmpStructure(self.params)

            if self.input_yml['defect'] == 'vac':
                def_type = 'vac'
                def_species = None
                def_orientation = None
            else:
                def_type = self.input_yml['int_type']
                def_species = self.input_yml['int_species']
                def_orientation = str(self.input_yml['int_orient'])
            
            struct_in.insert_point_defect(def_type, def_species, def_orientation)

            # define seeds for initializing velocities for all members
            seeds = {}
            while len(seeds) < len(self.sim_ids)*self.params['members']:
                seeds.update({random.randint(0, 100000): None})
            seeds = list(seeds.keys())

            # define input files for all temperatures and members (just equil.in is unique to each member) 
            i = 0
            for sim_i in self.sim_ids:
                self.params['temp'] = sim_i

                for mem_i in range(self.params['members']):
                    self.params['seed'] = seeds[i]

                    equil_in = LmpInput(file_path=self.templates_dir/'equil.in')
                    equil_in.add_params(self.params)

                    for lf in [main_in, struct_in, equil_in, diffusion_in, quench_in]:
                        conf_dict[sim_i][mem_i]['input_files'].update({lf.fn: lf})
                    i += 1

            self.state.update({conf_i: conf_dict})
    
    def build_directory(self):
        super().build_directory()
        self.dir.mkdir(exist_ok=True)
        
        for conf_i in range(self.input_yml['configurations']):
            conf_subdir: Path = self.dir / conf_i
            conf_subdir.mkdir(exist_ok=True)

            for sim_i in self.sim_ids:
                sim_subdir: Path = conf_subdir / sim_i
                sim_subdir.mkdir(exist_ok=True)

                for mem_i in range(self.input_yml['members']):
                    mem_subdir: Path = sim_subdir / mem_i
                    mem_subdir.mkdir(exist_ok=True)
                    self.state[conf_i][sim_i][mem_i].update({'dir' : mem_subdir})

    def run_lammps(self):
        for conf_i in range(self.input_yml['configurations']):
            self.run_lammps(self.state[conf_i])

    def analyze(self):
        # compute Emig for each configuration

        # 1. Extract squared displacement curves from each member
        # 2. Compute MSD for each temperature
        # 3. Fit MSD to get D, fit D to get Emig for a temperature
        # 4. Save Emig and repeat 1-3 for each configuration

        self.data.update()