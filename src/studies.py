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
    def __init__(self, lmp_fp: Path, num_processors: int):
        self.member_dir = lmp_fp.parent
        self.outfile = open(self.member_dir / f'{lmp_fp.stem}.out', 'w')
        self.finished = False
        self.counted = False

        self.lammps_cmd = [
            'srun', 
            f'--ntasks={num_processors}',
            '--export=ALL',
            'lmp', 
            '-in',
            lmp_fp.name]
        
        self.process = subprocess.Popen(self.lammps_cmd, cwd=self.member_dir, stdout=self.outfile, stderr=subprocess.STDOUT)
        logger.debug(f'Launching LAMMPS for sim={self.member_dir.parent.name} and member={self.member_dir.name}...')

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
                            conf_i, sim_i, mem_i = None, line[0], int(line[1])
                        elif len(line) == 3:
                            conf_i, sim_i, mem_i = int(line[0]), line[1], int(line[2])
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

    def run_lammps(self, sim_ids: list = None, lmp_fn = 'main.in'):
        """Continuously launch LAMMPS in parallel until all simulations and members have finished running."""
        if not sim_ids:
            sim_ids = self.sim_ids

        # replace 0 with jobs as they're scheduled 
        jobs = {sim_i: {mem_i: 0 for mem_i in self.state[sim_i].keys()} for sim_i in sim_ids}

        def check_status(status: int, return_next=False, return_all=False):
            """Helper function for determining which jobs to run next based on status (0 = ready, 1 = running, 2 = finished)."""
            if return_next:
                next_found = False
                for sim_i in sim_ids:
                    for mem_i, mem_dict in self.state[sim_i].items():
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
                for sim_i in sim_ids:
                    for mem_i, mem_dict in self.state[sim_i].items():
                        if mem_dict['status'] == status:
                            kw_pairs.append((sim_i, mem_i))
                return kw_pairs

            else:
                num_status = 0
                for sim_i in sim_ids:
                    for mem_dict in self.state[sim_i].values():
                        if mem_dict['status'] == status:
                            num_status += 1
                return num_status
        
        tot_num_jobs = 0
        for sim_i in sim_ids:
            tot_num_jobs += len(self.state[sim_i])

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
                    self.state[sim_i][mem_i]['status'] = 2
                    restart_file.write(f'{sim_i}\t{mem_i}\n')
                    logger.debug(f'LAMMPS finished for sim={sim_i} and member={mem_i}')

            # launch a job if possible
            if num_running < ntasks_per_job and num_running < num_left:
                sim_i, mem_i = check_status(0, return_next=True)
                
                if self.restart:
                    if sim_i in self.restart.keys():
                        if mem_i in self.restart[sim_i]:
                            self.state[sim_i][mem_i]['status'] = 2
                            restart_file.write(f'{sim_i}\t{mem_i}\n')
                            logger.debug(f'LAMMPS has already been run for sim={sim_i} and member={mem_i}. Skipping it')
                            continue

                job_dir: Path = self.state[sim_i][mem_i]['dir']

                # write input files
                for fn, lmpfile in self.state[sim_i][mem_i]['input_files'].items():
                    lmpfile.write_to_file(job_dir/fn)

                # run LAMMPS and save process
                jobs[sim_i][mem_i] = LmpJob(job_dir/lmp_fn, self.params['processors'])
                self.state[sim_i][mem_i]['status'] = 1
        
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
            'species': list(self.input_yml['composition'].keys()),
            'elements': tilps(list(self.input_yml['composition'].keys())),
            'temp': self.input_yml['temperature'],
            'equil': unprefix(self.input_yml['equil']),
            'mc_freq': unprefix(self.input_yml['mc'][0]),
            'mc_attempts': unprefix(self.input_yml['mc'][1]),
            'mc_thermo_freq': max(1, int(unprefix(0.25*self.input_yml['mc'][1]))),
            'mc': unprefix(self.input_yml['mc'][2]),
            'snapshot': unprefix(self.input_yml['snapshot'])
        })

        if 'wc_shell' not in self.input_yml.keys():
            self.params['wc_shell'] = 1
        
        if self.params['lattice'] == 'bcc':
            if self.params['wc_shell'] == 1:
                self.params['wc_num_neighbors'] = 8

            elif self.params['wc_shell'] == 2:
                self.params['wc_num_neighbors'] = 14

            elif self.params['wc_shell'] == 3:
                self.params['wc_num_neighbors'] = 26

        # define seeds for atom/swap RNG 
        mc_seeds = create_seeds(self.params['members'])

        # generate configurations and add input files
        for mem_i in range(self.params['members']):
            self.params.update({'mc_seed': mc_seeds[mem_i]})
            
            main_in = LmpInput(file_path=self.templates_dir/'main.in')
            main_in.add_params(self.params)

            struct_in = LmpStructure(lattice_params=self.params)
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
        super().run_lammps()

        # convert final configuration dumps into LAMMPS input files and save to dataset folder
        for mem_i in range(self.params['members']):
            old_struct: LmpStructure = self.state['runs'][mem_i]['input_files']['config.in'] 
            dump = LmpDump(file_path=self.state['runs'][mem_i]['dir']/'final.dump')

            new_struct_write_path = self.dataset_dir/f'config{mem_i}.in'
            new_struct_params = {
                'lattice': old_struct.lattice,
                'species': list(old_struct.species_to_type.keys()),
                'size': old_struct.size,
                'composition_str': old_struct.composition_str
            }

            dump.write_structure_file(new_struct_write_path, new_struct_params)

    def analyze(self):
        # compute Warren-Cowley parameters of all configurations
        for mem_i in range(self.params['members']):
            # compute parameters for all snapshots for plotting the evolution
            snapshots_dump = LmpDump(file_path=self.state['runs'][mem_i]['dir']/'mc.dump')
            
            wc = {
                'timesteps': [0]*len(snapshots_dump.frames),
                'snapshots': np.zeros((len(snapshots_dump.frames), len(self.params['species']), len(self.params['species']))),
                'final': np.zeros((len(self.params['species']), len(self.params['species'])))
            }

            i = 0
            for timestep, snapshot in snapshots_dump.frames.items():
                wc['timesteps'][i] = timestep  
                wc['snapshots'][i, :, :] = warren_cowley(
                    self.params['wc_num_neighbors'], 
                    snapshot['position'], 
                    snapshot['type'], 
                    np.array([snapshot['box']['xlo'], snapshot['box']['ylo'], snapshot['box']['zlo']]),
                    snapshot['boxsize'],
                )
                i += 1

            # compute WC on final frame for reference
            final_frame = list(LmpDump(file_path=self.state['runs'][mem_i]['dir']/'final.dump').frames.values())[0]
            wc['final'] = warren_cowley(
                self.params['wc_num_neighbors'], 
                final_frame['position'], 
                final_frame['type'], 
                np.array([final_frame['box']['xlo'], final_frame['box']['ylo'], final_frame['box']['zlo']]),
                final_frame['boxsize'],
            )
            
            self.state['runs'][mem_i]['wc'] = wc
    
    def save_data(self):
        wc_final_file = open(self.dir / 'wc.out', 'w')
        all_wc_final = np.zeros((self.params['members'], len(self.params['species']), len(self.params['species'])))

        # save WC parameters data
        for mem_i in range(self.params['members']):
            subdir = self.state['runs'][mem_i]['dir']
            wc_evolution_file = open(subdir/'wc_snapshots.out', 'w')

            wc_dict = self.state['runs'][mem_i]['wc']

            # plot evolution for each initial configuration
            for i in range(len(self.params['species'])):
                for j in range(len(self.params['species']))[i:]:
                    pair_str = f"{self.params['species'][i]}-{self.params['species'][j]}"
                    plt.plot(wc_dict['timesteps'], wc_dict['snapshots'][:, i, j], label=pair_str)
            
            plt.hlines(0, wc_dict['timesteps'][0], wc_dict['timesteps'][-1])
            plt.ylim([-1, 1])
            plt.xlabel('Timestep')
            plt.ylabel('Warren-Cowley Parameter')
            plt.legend()
            plt.savefig(self.state['runs'][mem_i]['dir']/'wc_evolution.png', bbox_inches="tight")
            plt.close()
            
            # write all computed WC parameters for each snapshot so they can be recovered later
            for i, t in enumerate(wc_dict['timesteps']):
                wc_evolution_file.write(str(t)+'\n')
                wc_evolution_file.write(np.array2string(wc_dict['snapshots'][i, :, :])+'\n\n')

            wc_evolution_file.close()

            # write final wc parameters
            wc_final_file.write(str(mem_i)+'\n')
            wc_final_file.write(np.array2string(wc_dict['final'])+'\n\n')

            all_wc_final[mem_i, :, :] = wc_dict['final']
        
        # compute average WC parameters
        all_wc_average = np.average(all_wc_final, axis=0)

        wc_final_file.write('Average\n')
        wc_final_file.write(np.array2string(all_wc_average))
        wc_final_file.close()

@register_study
class PointDefectInsertion(Study):
    def init_state(self):
        # setup containers
        self.sim_ids = ['pristine', 'defective']
        mem_dict = {mem_i: {'input_files': {}, 'status': 0, 'dir': None} for mem_i in range(self.params['members'])}
        self.state.update({sim_i: deepcopy(mem_dict) for sim_i in self.sim_ids})

        # add elements parameter for defining potential
        self.params.update({'elements': tilps(list(self.input_yml['composition'].keys()))})

        # minimization stopping criteria
        self.params.update({
            'etol': f"{self.input_yml['minimize'][0]:.2e}",
            'ftol': f"{self.input_yml['minimize'][1]:.2e}",
            'maxiter': unprefix(self.input_yml['minimize'][2]),
            'maxeval': unprefix(self.input_yml['minimize'][3])
        })

        # randomly choose configurations
        if 'dataset' in self.input_yml.keys():
            dataset_dir = Path(self.input_yml['dataset'])
            if not dataset_dir.exists():
                raise ValueError(f'Dataset directory {dataset_dir} does not exist')
            
            dataset_configs = [fp for fp in dataset_dir.iterdir() if fp.is_file()]

            # make sure there are enough configurations
            if self.params['members'] > len(dataset_configs):
                raise ValueError(f"Number of members ({self.params['members']}) exceeds number of configurations available in dataset ({len(dataset_configs)})")
            
            dataset_configs = [dataset_configs[i] for i in random_range(0, len(dataset_configs))]

        # define input files for each member
        for mem_i in range(self.params['members']):
            # main input files for perfect and defective systems
            pris_in = LmpInput(file_path=self.templates_dir/'pristine.in')
            pris_in.add_params(self.params)
            self.state['pristine'][mem_i]['input_files'].update({'pristine.in': pris_in})
            
            def_in = LmpInput(file_path=self.templates_dir/'defective.in')
            def_in.add_params(self.params)
            self.state['defective'][mem_i]['input_files'].update({'defective.in': def_in})
            
            # either load a configuration or make one from scratch
            if 'dataset' in self.input_yml.keys():
                pris_struct = LmpStructure(file_path=dataset_configs[mem_i])
            else:
                pris_struct = LmpStructure(lattice_params=self.params)

            self.state['pristine'][mem_i]['input_files'].update({'pristine.struct': pris_struct})

    def build_directory(self):
        super().build_directory()
        self.dir.mkdir(exist_ok=True)

        # self.state[sim_id] dict should only have keys that are member indices
        runs_dir: Path = self.dir / 'runs'
        runs_dir.mkdir(exist_ok=True)

        for mem_i in range(self.input_yml['members']):
            subdir = runs_dir / str(mem_i)
            subdir.mkdir(exist_ok=True)
            self.state['pristine'][mem_i].update({'dir' : subdir})
            self.state['defective'][mem_i].update({'dir' : subdir})

    def run_lammps(self):
        # relax pristine system first
        logger.debug(f'Starting with first set of LAMMPS simulations for pristine system')
        super().run_lammps(sim_ids=['pristine'], lmp_fn='pristine.in')

        # insert point defect into pristine system
        for mem_i in range(self.params['members']):
            subdir = self.state['pristine'][mem_i]['dir']
            pris_dump = LmpDump(subdir / 'pristine.dump')

            if self.input_yml['defect'] == 'vac':
                def_type = 'vac'
                def_species = None
                def_orientation = None
            else:
                def_type = self.input_yml['int_type']
                def_species = self.input_yml['int_species']
                def_orientation = str(self.input_yml['int_orient'])

            pris_struct: LmpStructure = self.state['pristine'][mem_i]['input_files']['pristine.struct']
            pris_lat_params = {
                'lattice': pris_struct.lattice,
                'size': pris_struct.size,
                'composition_str': pris_struct.composition_str
            }

            def_struct: LmpStructure = pris_dump.to_struct(pris_lat_params)
            def_struct.insert_point_defect(def_type, def_species, def_orientation)

            self.state['defective'][mem_i]['input_files'].update({'defective.struct': def_struct})
        
        # relax defective system
        logger.debug(f'Running second set of LAMMPS simulations for defective system')
        super().run_lammps(sim_ids=['defective'], lmp_fn='defective.in')

    def analyze(self):
        # setup container
        self.data.update({'pristine_e': [], 'defective_e': [], 'insertion_e': []})
        
        # read in potential energy from last thermo output
        for mem_i in range(self.input_yml['members']):
            subdir = self.state['pristine'][mem_i]['dir']

            pris_log = LmpLog(file_path=subdir/'energies.log')
            pris_e = pris_log.data['PotEng'][-2]

            def_log = LmpLog(file_path=subdir/'energies.log')
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
        raise NotImplementedError()
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