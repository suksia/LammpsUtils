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
        self.dir = None

        # check directory for restart file first
        input_dir = Path(input_yml['dir'])
        assert input_dir.exists(), f'Directory {input_dir} does not exist'

        for file in input_dir.iterdir():
            if file.name == 'LammpsUtils.restart':
                logger.debug(f'Restart file found. Reading contents...')
                self.dir = input_dir

                self.restart: dict[int, list[int]] = {}
                with open(input_dir/'LammpsUtils.restart', 'r') as rf:
                    for line in rf.readlines():
                        temp, mem_i = strip_split(line)

                        if temp not in self.restart.keys():
                            self.restart.update({temp: [mem_i]})
                        else:
                            self.restart[temp].append(mem_i)
                break
        
        if self.dir is None:
            self.dir = next_path(Path(input_yml['dir']) / f"{input_yml['defect']}_diffusion")
            self.restart = False

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

        if self.input_yml['defect'] == 'int':
            sf = 1/self.params['lattice_const']
        else:
            sf = 1
        self.params['pd_x'] = sf*self.params['position'][0]
        self.params['pd_y'] = sf*self.params['position'][1]
        self.params['pd_z'] = sf*self.params['position'][2]

        self.params['equil'] = unprefix(self.params['equil'])
        self.params['diffusion'] = unprefix(self.params['diffusion'])
        self.params['snapshot'] = unprefix(self.params['snapshot'])
        self.params['num_snapshots'] = int(self.params['diffusion'] / self.params['snapshot'])

        if 'tlo' in self.input_yml.keys():
            self.params['tlo'] = self.input_yml['tlo']
        else:
            self.params['tlo'] = None
        
        if 'thi' in self.input_yml.keys():
            self.params['thi'] = self.input_yml['thi']
        else:
            self.params['thi'] = None

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
                    else:
                        in_file = LammpsInput(file_path=fp)
                
                # define quench.in as an empty file if not going to quench snapshots
                elif fp.name == 'quench.in' and self.params['quench'] == False:
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
            subdir.mkdir(exist_ok=True)

            for m in range(self.params['members']):
                member_subdir = subdir / str(m)
                member_subdir.mkdir(exist_ok=True)

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

        # create/overwrite 
        restart_file = open(self.dir / 'LammpsUtils.restart', 'w')

        # launch jobs until all have been counted
        while check_status(2) < tot_num_jobs:
            num_running, num_left = check_status(1), check_status(0)
            
            # poll running jobs to update their state if finished
            for temp, mem_i in check_status(1, return_all=True):
                job: LammpsJob = jobs[temp][mem_i]
                job.poll()
                if job.finished and not job.counted:
                    jobs_status[temp][mem_i] = 2
                    restart_file.write(f'{temp}\t{mem_i}\n')
                    logger.debug(f'LAMMPS finished for T={temp} and member={mem_i}')

            # launch a job if possible
            if num_running < math.floor(NTASKS / self.input_yml['processors']) and num_running < num_left:
                temp, mem_i = check_status(0, return_next=True)
                
                # check if temp/mem_i combo has already been run
                if self.restart:
                    if mem_i in self.restart[temp]:
                        jobs_status[temp][mem_i] = 2
                        logger.debug(f'LAMMPS has already been run for T={temp} and member={mem_i}. Skipping it')
                        continue

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
        
        restart_file.close()

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

        # save time in ns using previous sq_file loaded
        self.data.update({'t': [step*self.input_yml['timestep'] / 1000 for step in sq_file.data['Step']]})

        # defect diffusion next using Wigner-Seitz cell anaylsis in OVITO
        if self.input_yml['quench']:
            dump_fn = 'quench.dump'
        else:
            dump_fn = 'diffusion.dump'
        
        if self.input_yml['defect'] == 'int':
            occupancy = 2
        elif self.input_yml['defect'] == 'vac':
            occupancy = 0
        
        for temp in self.sim_ids:
            # create file for writing number of jumps for each temperature
            with open(self.state[temp]['dir'] / 'jumps.txt', 'w') as jumps:
                jumps.write(f"{'Member':<10} {'Jumps':<10}\n")

            for mem_i in range(self.input_yml['members']):
                job_dir: Path = self.state[temp]['dir'] / str(mem_i)
                pipeline = import_file(job_dir / dump_fn)

                # outputs sites as particles DataCollection with "Occupancy" property
                ws = WignerSeitzAnalysisModifier()
                pipeline.modifiers.append(ws)

                # custom modifier to obtain a vacancy position for each snapshot
                def modify(frame, data):
                    # per-site occupancy (0 = vacancy, 1 = normal site, 2 = interstitial)
                    occupancies = data.particles['Occupancy']

                    # add a boolean "Selection" property which will be 1 for sites with a vacancy
                    selection = data.particles_.create_property('Selection')
                    selection[...] = occupancies == occupancy

                    # add a data attribute for the current frame which is the vacancy position
                    selected_idx = np.nonzero(selection)
                    data.attributes['DefectCount'] = np.sum(selection)
                    data.attributes['DefectPosition'] = data.particles.positions[selected_idx]
                    data.attributes['Occupancy'] = occupancy

                # write vacancy positions
                pipeline.modifiers.append(modify)
                export_file(
                    pipeline, 
                    job_dir / 'ovito.txt', 
                    'txt/attr',
                    columns = ['Timestep', 'Occupancy', 'DefectCount', 'DefectPosition'],
                    multiple_frames = True)

                # compute square displacement
                sq_dis = [0]
                frames = [frame for frame in pipeline.frames]
                ref_def_pos = frames[1].attributes['DefectPosition'][0]
                
                defect_tsteps, num_jumps, num_crosses = [0], 0, [0, 0, 0]
                box_width = [self.input_yml['lattice_const']*size for size in self.input_yml['size']]
                boundary_jump_tol = 0.80*min(box_width)

                with open(job_dir / 'unwrapped.txt', 'w') as unw:
                    unw.write(f"{'Time':<10} {'Current Pos':<25} {'Number of crosses':<20} {'Unwrapped Pos':<25} {'Squared Disp':<15}\n")

                prev_def_pos = copy(ref_def_pos)
                for frame in frames[1:]:
                    t_step =  frame.attributes['Timestep']
                    def_pos = frame.attributes['DefectPosition']

                    if len(def_pos) > 1:
                        logger.debug(f'WARNING: More than 1 defect detected for timestep={t_step}!')
                    def_pos = def_pos[0]

                    # determine if a jump occured and unwrap coordinates if a boundary was crossed
                    dr = def_pos - prev_def_pos
                    if np.linalg.norm(dr) > 0.1:
                        num_jumps += 1

                    # unwrap each coord
                    unwrapped_def_pos = np.array([0.0, 0.0, 0.0])
                    for i in range(3):
                        # crossed a box boundary
                        if abs(dr[i]) > boundary_jump_tol:
                            # direction of cross (+1 = high, -1 = low)
                            cross_dir = -np.sign(dr[i])

                            # number of times already crossed this boundary
                            num_crosses[i] += int(cross_dir)
                            
                        # update coord
                        unwrapped_def_pos[i] = def_pos[i] + num_crosses[i]*box_width[i]

                    defect_tsteps.append(t_step)

                    sq_dis_val = float(np.linalg.norm(unwrapped_def_pos - ref_def_pos)**2)
                    sq_dis.append(sq_dis_val)

                    with open(job_dir / 'unwrapped.txt', 'a') as unw:
                        unw.write(f"{t_step*self.input_yml['timestep']/1000:<10} {def_pos:<25} {num_crosses:<20} {unwrapped_def_pos:<25} {sq_dis_val:6.3f}\n")

                    # create dumps file with vacancy trajectory
                    trj_header_lines = [
                        'ITEM: TIMESTEP\n',
                        f'{t_step}\n'
                        'ITEM: NUMBER OF ATOMS\n',
                        '1\n'
                        'ITEM: BOX BOUNDS pp pp pp\n',
                        f'0.0 {box_width[0]}\n',
                        f'0.0 {box_width[1]}\n',
                        f'0.0 {box_width[2]}\n',
                        'ITEM: ATOMS id type x y z\n',
                    ]

                    with open(job_dir / 'def_trj.dump', 'a') as trj:
                        trj.writelines(trj_header_lines)
                        trj.write(f'1 1 {def_pos[0]} {def_pos[1]} {def_pos[2]}\n')

                    with open(job_dir / 'def_trj_unw.dump', 'a') as trj:
                        trj.writelines(trj_header_lines)
                        trj.write(f'1 1 {unwrapped_def_pos[0]} {unwrapped_def_pos[1]} {unwrapped_def_pos[2]}\n')

                    prev_def_pos = def_pos
                
                # write out number of jumps
                with open(job_dir.parent / 'jumps.txt', 'a') as jf:
                    jf.write(f'{mem_i:<10} {num_jumps:<10}\n')

                # save squared displacement for current member
                self.data['defect'][temp][mem_i] = sq_dis

                if self.data['defect'][temp]['msd'] is None:
                    self.data['defect'][temp]['msd'] = np.array(sq_dis)
                else:
                    self.data['defect'][temp]['msd'] = np.vstack((self.data['defect'][temp]['msd'], sq_dis))
            
        for method in ['self', 'defect']:
            # compute MSD for temperature
            msd = []
            if self.input_yml['members'] > 1:
                for col in range(len(self.data[method][temp]['msd'][0, :])):
                    msd.append(float(np.mean(self.data[method][temp]['msd'][:, col])))
            else:
                msd = self.data[method][temp]['msd'].tolist()
                logger.debug('WARNING: Ensemble consists of only 1 member. Do not trust the MSD!')
            self.data[method][temp]['msd'] = msd

        # define lower/upper bounds for time axis for fitting diffusivities next
        if self.params['tlo']:
            tlo_i = np.argmin(np.abs(np.array(self.data['t'])-self.params['tlo']))
        else:
            tlo_i = 0

        if self.params['thi']:
            thi_i = np.argmin(np.abs(np.array(self.data['t'])-self.params['thi']))
        else:
            thi_i = -1

        # fit the data to obtain diffusivities and migration energies
        for method in ['self', 'defect']:
            x = self.data['t'][tlo_i:thi_i]

            # fit MSD data first to obtain diffusivity
            for temp in self.sim_ids:
                y = self.data[method][temp]['msd'][tlo_i:thi_i]
                D, D_int, r2 = linear_fit(x, y)
                self.data[method][temp].update({'D': float(D/6), 'D_intercept': float(D_int), 'D_err': float(r2)})    

            # fit diffusivities next to obtain migration energy
            if len(self.sim_ids) == 1:
                logger.debug('WARNING: Not enough temperatures to fit an Arrhenius plot. Skipping migration energy calculation')
                self.data[method].update({'Emig': None})
            else:
                x = [1/temp for temp in self.sim_ids]
                y = [math.log(self.data[method][temp]['D']) for temp in self.sim_ids]
                Emig, Emig_int, r2 = linear_fit(x, y)
                self.data[method].update({'arrhenius_data': (x, y), 'Emig': float(Emig*8.61733e-5), 'Emig_intercept': float(Emig_int), 'Emig_err': float(r2)})
    
    def save_data(self):
        """Plot curves and write out data."""
        # plot equilibriation and squared displacement curves
        for temp in self.sim_ids:
            for mem_i in range(self.input_yml['members']):
                equil_log = LammpsLog(self.state[temp]['dir'] / str(mem_i) / 'equil.log')
                equil_log.plot_values(save_prefix='equil')
                
                for method in ['self', 'defect']:
                    plt.plot(self.data['t'], self.data[method][temp][mem_i])
                    plt.xlabel('Time [ns]')
                    plt.ylabel('Squared Displacement [$Å^2$]')
                    plt.savefig(self.state[temp]['dir'] / f'{method}_sd.png', bbox_inches="tight")
                    plt.close()

        for method in ['self', 'defect']:
            # plot squared displacement curves together
            for temp in self.sim_ids:
                for mem_i in range(self.input_yml['members']):
                    plt.plot(self.data['t'], self.data[method][temp][mem_i])
                plt.xlabel('Time [ns]')
                plt.ylabel('Squared Displacement [$Å^2$]')
                plt.savefig(self.state[temp]['dir'] / f'{method}_sd.png', bbox_inches="tight")
                plt.close()

            # save msd data
            for temp in self.sim_ids:
                with open(self.state[temp]['dir'] / f'{method}_msd.txt', 'w') as msd_file:
                    msd_file.write(f"{'time[ns]':<10} {'msd[Å2]':<10}\n")
                    for i in range(len(self.data['t'])):
                        msd_file.write(f"{self.data['t'][i]:<10} {self.data[method][temp]['msd'][i]:6.3f}\n")

            # plot MSD for each temperature with fitting line
            for temp in self.sim_ids:
                a, b, r2 = self.data[method][temp]['D'], self.data[method][temp]['D_intercept'], self.data[method][temp]['D_err']
                plt.plot(self.data['t'], self.data[method][temp]['msd'])
                plt.plot(self.data['t'], [6*a*t+b for t in self.data['t']], '--', label=f"$R^2$={100*r2:2.2f}%")
                plt.title(f'D = {a:1.2e} [$Å^2$/ns]')
                plt.xlabel('Time [ns]')
                plt.ylabel('Mean Squared Displacement [$Å^2$]')
                plt.legend()
                plt.savefig(self.state[temp]['dir'] / f'{method}_msd.png', bbox_inches="tight")
                plt.close()
            
            # compare msd for each temp
            for temp in self.sim_ids:
                plt.plot(self.data['t'], self.data[method][temp]['msd'], label=f'{temp}K')
            plt.legend()
            plt.xlabel('Time [ns]')
            plt.ylabel('Mean Squared Displacement [$Å^2$]')
            plt.savefig(self.dir / f'{method}_msd_by_T.png', bbox_inches="tight")
            plt.close()

            # write diffusivity data
            with open(self.dir / f'{method}_fit.txt', 'w') as df:
                df.write(f"{'T [K]':<10} {'D [Å2/ns]':<10} {'D [cm2/s]':<10} {'Error [%]':<10}\n")
                for temp in self.sim_ids:
                    df.write(f"{temp:<10} {self.data[method][temp]['D']:3.2e} {self.data[method][temp]['D']*10e-7:3.2e} {100*self.data[method][temp]['D_err']:2.2f}\n")
                df.write('\n')
                df.write(f"{'Emig [eV]':<10} {'Error [%]':<10}\n")
                df.write(f"{self.data['Emig']:7.3f} {100*self.data[method][temp]['Emig_err']:2.2f}")

            # plot migration energy fitting
            if self.data['Emig']:
                x, y = self.data['arrhenius_data']
                a, b, r2 = self.data['Emig'], self.data['Emig_intercept'], self.data['Emig_err']
                plt.plot(x, y)
                plt.plot(x, x*a/8.61733e-5+b, '--', label=f"$R^2$={100*r2:2.2f}%")
                plt.title(f'Emig = {a:1.2f} [eV]')
                plt.xlabel('1/T [$K^{-1}$]')
                plt.ylabel('ln(D)')
                plt.legend()
                plt.savefig(self.dir / f'{method}_arrhenius.png')


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

    def plot_values(self, save_prefix:str = None):
        try:
            x = self.data['Step']
        except:
            raise KeyError(f'`Step` must be one of the data labels for log file at {self.path}')
        
        y_labels = list(self.data.keys())
        y_labels.pop(y_labels.index('Step'))

        if save_prefix:
            save_prefix += '_'
        else:
            save_prefix = ''

        for y_lab in y_labels:
            plt.plot(x, self.data[y_lab])
            plt.xlabel('Timestep')
            plt.ylabel(y_lab)
            plt.savefig(self.path.parent / f'{save_prefix}{y_lab}.png')
            plt.close()

main()