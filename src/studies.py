import logging, random, math, os, time, subprocess
from copy import copy, deepcopy
from pathlib import Path
from lammps_file import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.io import import_file, export_file

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
        logger.debug(f'Launching LAMMPS in dir={self.member_dir.parent.name} for member={self.member_dir.name}...')

    def poll(self):
        poll = self.process.poll()
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
                
                # delete restart file so it will be empty for first run_lammps() call
                Path(input_dir/'LammpsUtils.restart').unlink()              
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
        max_parallel_njobs = math.floor(NTASKS / self.input_yml['processors'])
        if max_parallel_njobs < 1:
            raise ValueError(f"{NTASKS} processors available. Not enough for a single job ({self.input_yml['processors']})")
        
        # launch jobs until all have been counted
        while check_status(2) < tot_num_jobs:
            num_running, num_left = check_status(1), check_status(0)
            
            # poll running jobs to update their state if finished
            for sim_i, mem_i in check_status(1, return_all=True):
                job: LmpJob = jobs[sim_i][mem_i]
                job.poll()
                if job.finished and not job.counted:
                    self.state[sim_i][mem_i]['status'] = 2

                    with open(self.dir / 'LammpsUtils.restart', 'a') as f:
                        f.write(f'{sim_i}\t{mem_i}\n')

                    logger.debug(f'LAMMPS finished for sim={sim_i} and member={mem_i}')

            # launch a job if possible
            if num_running < max_parallel_njobs and num_left:
                sim_i, mem_i = check_status(0, return_next=True)
                
                if self.restart:
                    if sim_i in self.restart.keys():
                        if mem_i in self.restart[sim_i]:
                            self.state[sim_i][mem_i]['status'] = 2

                            with open(self.dir / 'LammpsUtils.restart', 'a') as f:
                                f.write(f'{sim_i}\t{mem_i}\n')

                            logger.debug(f'LAMMPS has already been run for sim={sim_i} and member={mem_i}. Skipping it')
                            continue

                job_dir: Path = self.state[sim_i][mem_i]['dir']

                # write input files
                for fn, lmpfile in self.state[sim_i][mem_i]['input_files'].items():
                    lmpfile.write_to_file(job_dir/fn)

                # run LAMMPS and save process
                jobs[sim_i][mem_i] = LmpJob(job_dir/lmp_fn, self.params['processors'])
                self.state[sim_i][mem_i]['status'] = 1

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
class ShortRangeOrder(Study):
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
            'mc': unprefix(self.input_yml['mc'][2]),
            'snapshot': unprefix(self.input_yml['snapshot']),
            'etol': f"{self.input_yml['minimize'][0]:.2e}",
            'ftol': f"{self.input_yml['minimize'][1]:.2e}",
            'maxiter': unprefix(self.input_yml['minimize'][2]),
            'maxeval': unprefix(self.input_yml['minimize'][3])
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
        # compute average enthalpy and acceptance ratio from mc.log files
        enthalpy = np.zeros((self.params['members'], int(self.params['mc']/self.params['mc_freq'])+1))
        acc_ratio = np.zeros((self.params['members'], int(self.params['mc']/self.params['mc_freq'])+1))

        for mem_i in range(self.params['members']):
            mc_log = LmpLog(self.state['runs'][mem_i]['dir']/'mc.log')

            enthalpy[mem_i, :] = mc_log.data_df['Enthalpy'].to_numpy()
            acc_ratio[mem_i, :] = mc_log.data_df['v_acc_ratio'].to_numpy()

        self.data['timesteps'] = mc_log.data_df.index.to_numpy()

        self.data['enthalpy'] = np.mean(enthalpy, axis=0)
        self.data['enthalpy_std'] = np.std(enthalpy, axis=0)

        self.data['acc_ratio'] = np.mean(acc_ratio, axis=0)
        self.data['acc_ratio_std'] = np.std(acc_ratio, axis=0)

        # compute Warren-Cowley parameters for all timesteps and final configurations to be averaged later

        # dim0 = members, dim1 = timesteps, dim2 = central atom, dim3 = neighbor
        wc = np.zeros((self.params['members'], int(self.params['mc']/self.params['snapshot'])+1, len(self.params['species']), len(self.params['species'])))
        wc_final = np.zeros((self.params['members'], len(self.params['species']), len(self.params['species'])))

        for mem_i in range(self.params['members']):
            mc_dump = LmpDump(file_path=self.state['runs'][mem_i]['dir']/'mc.dump')

            # WC evolution over time for each member
            for i, snapshot in enumerate(mc_dump.frames.values()):
                wc[mem_i, i, :, :] = warren_cowley(
                    self.params['wc_num_neighbors'], 
                    snapshot['position'], 
                    snapshot['type'], 
                    np.array([snapshot['box']['xlo'], snapshot['box']['ylo'], snapshot['box']['zlo']]),
                    snapshot['boxsize'])
            
            # WC for final frame for each member
            final_frame = list(LmpDump(file_path=self.state['runs'][mem_i]['dir']/'final.dump').frames.values())[0]
            wc_final[mem_i, :, :] = warren_cowley(
                self.params['wc_num_neighbors'], 
                final_frame['position'], 
                final_frame['type'], 
                np.array([final_frame['box']['xlo'], final_frame['box']['ylo'], final_frame['box']['zlo']]),
                final_frame['boxsize'])

        self.data['wc_timesteps'] = np.array(list(mc_dump.frames.keys()))
        self.data['wc'] = np.mean(wc, axis=0)
        self.data['wc_std'] = np.std(wc, axis=0)
        self.data['wc_final'] = wc_final

    def save_data(self):
        # plot enthalpy
        plt.plot(self.data['timesteps'], self.data['enthalpy'], '--o', ms=2)
        plt.fill_between(self.data['timesteps'], self.data['enthalpy']-self.data['enthalpy_std'], self.data['enthalpy']+self.data['enthalpy_std'], alpha=0.5)
        plt.xlabel('Timestep')
        plt.ylabel('Enthalpy [eV]')
        plt.savefig(self.dir/'enthalpy.png', bbox_inches="tight")
        plt.close()

        # write out enthalpy data
        with open(self.dir/'enthalpy.out', 'w') as e:
            for i in range(len(self.data['timesteps'])):
                e.write(f"{self.data['timesteps'][i]:<10} {self.data['enthalpy'][i]:<15.3f} {self.data['enthalpy_std'][i]:<5.3f}\n")

        # plot acceptance ratio
        plt.plot(self.data['timesteps'], self.data['acc_ratio'], '--o', ms=2)
        plt.fill_between(self.data['timesteps'], self.data['acc_ratio']-self.data['acc_ratio_std'], self.data['acc_ratio']+self.data['acc_ratio_std'], alpha=0.5)
        plt.ylim([0, 1])
        plt.xlabel('Timestep')
        plt.ylabel('Metropolis Acceptance Ratio')
        plt.savefig(self.dir/'acc_ratio.png', bbox_inches="tight")
        plt.close()

        # plot WC evolution data averaged across members
        for i in range(len(self.params['species']))[:-1]:
            for j in range(len(self.params['species']))[i+1:]:
                pair_str = f"{self.params['species'][i]}-{self.params['species'][j]}"
                x, y = self.data['wc_timesteps'], self.data['wc'][:, i, j]
                yerr = (y - self.data['wc_std'][:, i, j], y + self.data['wc_std'][:, i, j])
                plt.plot(x, y, '--o', ms=2, label=pair_str)
                plt.fill_between(x, yerr[0], yerr[1], alpha=0.5)
        
        plt.hlines(0, self.data['wc_timesteps'][0], self.data['wc_timesteps'][-1], color='black', ls='--')
        plt.xlabel('Timestep')
        plt.ylabel('Warren-Cowley Parameter')
        plt.legend()
        plt.savefig(self.dir/'wc.png', bbox_inches="tight" )
        plt.close()

        # write WC evolution data
        with open(self.dir/'wc.out', 'w') as f:
            for i, t in enumerate(self.data['wc_timesteps']):
                f.write(str(t)+'\n')
                f.write(np.array2string(self.data['wc'][i, :, :])+'\n')
                f.write(np.array2string(self.data['wc_std'][i, :, :])+'\n\n')

        # bin final configuration WC parameters and plot the histogram
        for i in range(len(self.params['species']))[:-1]:
            for j in range(len(self.params['species']))[i+1:]:
                pair_str = f"{self.params['species'][i]}-{self.params['species'][j]}"
                histo, bin_edges = np.histogram(self.data['wc_final'][:, i, j], bins=40, density=True)
                plt.bar(bin_edges[:-1], histo, label=pair_str, linewidth=1, edgecolor='navy', width=np.diff(bin_edges))

        plt.vlines(0, 0, np.max(histo), color='black', ls='--')
        plt.xlabel('Warren-Cowley Parameter')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(self.dir/'wc_final.png', bbox_inches="tight" )
        plt.close()

        # write final configuration WC data
        with open(self.dir/'wc_final.out', 'w') as f:
            for mem_i in range(self.params['members']):
                f.write(str(mem_i)+'\n')
                f.write(np.array2string(self.data['wc_final'][mem_i, :, :])+'\n\n')
        
            # compute average
            f.write('Average\n')
            f.write(np.array2string(np.average(self.data['wc_final'], axis=0)))

@register_study
class PointDefect(Study):
    def init_state(self):
        # setup containers
        self.sim_ids = ['pristine', 'defective']
        mem_dict = {mem_i: {'input_files': {}, 'status': 0, 'dir': None} for mem_i in range(self.params['members'])}
        self.state.update({sim_i: deepcopy(mem_dict) for sim_i in self.sim_ids})
        
        # update params common to all members/configurations
        self.params.update({
            'species': list(self.input_yml['composition'].keys()),
            'elements': tilps(list(self.input_yml['composition'].keys())),
            'temp': self.input_yml['temperature'],
            'equil': unprefix(self.input_yml['equil']),
            'mc_freq': unprefix(self.input_yml['mc'][0]),
            'mc_attempts': unprefix(self.input_yml['mc'][1]),
            'mc': unprefix(self.input_yml['mc'][2]),
            'snapshot': unprefix(self.input_yml['snapshot']),
            'etol': f"{self.input_yml['minimize'][0]:.2e}",
            'ftol': f"{self.input_yml['minimize'][1]:.2e}",
            'maxiter': unprefix(self.input_yml['minimize'][2]),
            'maxeval': unprefix(self.input_yml['minimize'][3])})

        self.params.update({'num_snapshots': int(self.params['mc']/self.params['snapshot'])-1})

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

        # create seeds for thermostat, velocities, and atom/swap
        seeds = create_seeds(3*self.params['members'])

        # define input files for each member
        seed_idx = 0
        for mem_i in range(self.params['members']):
            # main input files for perfect and defective systems
            pris_in = LmpInput(file_path=self.templates_dir/'pristine.in')
            pris_in.add_params(self.params)
            self.state['pristine'][mem_i]['input_files'].update({'pristine.in': pris_in})

            self.params.update({
                'thermo_seed': seeds[seed_idx],
                'vel_seed': seeds[seed_idx+1],
                'mc_seed': seeds[seed_idx+2],
            })
            seed_idx += 3

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
        
        # run MC+MD loop for defective system
        logger.debug(f'Running second set of LAMMPS simulations for defective system')
        super().run_lammps(sim_ids=['defective'], lmp_fn='defective.in')
    
    def analyze(self):
        # compute insertion energy evolution
        e_pris = np.zeros(self.params['members'])
        e_def = np.zeros((self.params['members'], self.params['num_snapshots']))
        e_ins = np.zeros((self.params['members'], self.params['num_snapshots']))

        for mem_i in range(self.params['members']):
            energies_log = LmpLog(self.state['defective'][mem_i]['dir']/'energies.log')

            e_pris[mem_i] = energies_log.data_df['PotEng'][0]
            e_def[mem_i, :] = energies_log.data_df['PotEng'][1:].to_numpy()

            e_ins[mem_i, :] = e_def[mem_i, :] - e_pris[mem_i]

        self.data['timesteps'] = energies_log.data_df.index.to_numpy()[1:]

        self.data['e_pris'] = e_pris
        self.data['e_def'] = e_def
        self.data['e_ins'] = e_ins

        # obtain positions of defective cells
        for mem_i in range(self.input_yml['members']):
            pipeline = import_file(self.state['defective'][mem_i]['dir']/'quench.dump')

            # outputs sites as particles DataCollection with "Occupancy" property
            ws = WignerSeitzAnalysisModifier()
            pipeline.modifiers.append(ws)

            # custom modifier to obtain a vacancy position for each snapshot
            def modify(frame, data):
                # per-site occupancy (0 = vacancy, 1 = normal site, 2 = interstitial)
                occupancies = data.particles['Occupancy']

                # add a boolean "Selection" property which will be 1 only for pristine cells
                selection = data.particles_.create_property('Selection')
                selection[...] = occupancies != 1

                # add a data attribute for the current frame which is the vacancy position
                data.attributes['DefectCount'] = np.sum(selection)
                data.attributes['DefectPosition']  = data.particles.positions[np.nonzero(selection)]
                data.attributes['Occupancy'] = data.particles['Occupancy'][np.nonzero(selection)]

            # write vacancy positions
            pipeline.modifiers.append(modify)
            export_file(
                pipeline, 
                self.state['defective'][mem_i]['dir']/'ovito.out', 
                'txt/attr',
                columns = ['Timestep', 'Occupancy', 'DefectCount', 'DefectPosition'],
                multiple_frames = True)

    def save_data(self):
        # write out pristine and defective energies
        with open(self.dir/'energies.out', 'w') as e:
            e.write(f"Timesteps: {np.array2string(self.data['timesteps'])}\n\n")

            for mem_i in range(self.input_yml['members']):
                e.write(f"{mem_i:<5}  {self.data['e_pris'][mem_i]:<15.4f}  {np.array2string(self.data['e_def'][mem_i])}\n")

        # compute and plot insertion energy histogram for first and last snapshot
        first_histo, bin_edges = np.histogram(self.data['e_ins'][:, 0], bins=40, density=True)
        last_histo, bin_edges = np.histogram(self.data['e_ins'][:, -1], bins=40, density=True)

        fig, ax = plt.subplots(2, figsize=(10, 10))

        ax[0].bar(bin_edges[:-1], first_histo/np.max(first_histo), linewidth=1, edgecolor='navy', width=np.diff(bin_edges))
        ax[1].bar(bin_edges[:-1], last_histo/np.max(first_histo), linewidth=1, edgecolor='navy', width=np.diff(bin_edges))

        ax[0].vlines(0, 0, 1, color='black', ls='--')
        ax[1].vlines(0, 0, 1, color='black', ls='--')

        fig.savefig(self.dir/'final_insertion.png', bbox_inches="tight")

        # compute average insertion energy evolution

        # write out pristine and defective energies