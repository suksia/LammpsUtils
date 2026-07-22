import argparse, yaml, logging, sys, os, shutil
from pathlib import Path
from LammpsUtils.studies import Study, study_registry

# basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('LammpsUtils')

# define environment variable so LAMMPS can find potentials without needing a valid relative path
PKG_DIR = Path(__file__).parent.parent.parent
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
    with open(input_fp, 'r') as f:
        input_params_lines = f.readlines()
    logger.debug(f'Loaded input file {input_fp}')

    # initialize a study
    study_type = input_params['type']
    study: Study = study_registry[study_type](input_params)
    logger.debug(f'Initialized study type {study_type}')

    # build directory tree and copy in input file
    study.build_directory()
    logger.debug(f'Built directory tree at {study.dir}')
    with open(study.dir/input_fp.name, 'w') as f:
        f.writelines(input_params_lines)
        
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

main()