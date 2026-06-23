import logging, re, random
from pathlib import Path
from copy import deepcopy
import numpy as np
from utils import *
from masses import masses
import matplotlib.pyplot as plt

logger = logging.getLogger('LammpsUtils')
logging.getLogger("matplotlib").setLevel(logging.FATAL)

class LmpFile:
    def __init__(self, file_path: Path = None, content_str: str = None):      
        self.lines: list[str] = []
        self.last_read_path = file_path
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

    def write_to_file(self, write_path: Path, append_newline=False, lines=None):
        if lines is None:
            lines = self.lines
        if append_newline:
            lines = [l+'\n' for l in lines]

        with open(write_path, 'w') as d:
            d.writelines(lines)
            
        self.last_write_path = deepcopy(write_path)
        logger.debug(f'{self.__class__.__name__}: wrote lines to {write_path}')

class LmpInput(LmpFile):
    def add_params(self, params: dict):
        # loop through lines and replace ?param? with params[param]
        for kw, val in params.items():
            for i, line in enumerate(self.lines):
                self.lines[i] = re.sub(f'\?{kw}\?', str(val), line)
                
class LmpStructure(LmpFile):
    """Input structure data file for LAMMPS which is a randomized bcc/fcc lattice of elements."""
    def __init__(self, file_path: Path = None, lattice_params: dict = None):
        # attributes required to fully define the structure
        self.ids: np.ndarray = None
        self.types: np.ndarray = None
        self.positions: np.ndarray = None

        self.size = []
        self.box = {}
        self.boxsize = np.zeros(3)

        self.lattice: str = ''
        self.lattice_const = 0.0
        self.num_types = 0
        self.num_atoms = 0
        self.composition: dict = {}
        self.composition_str = ''
        
        # dictionary mapping element species to LAMMPS atom type
        self.species_to_type = {}
        
        # initializes some attributes and reads in contents if a file path was provided
        super().__init__(file_path=file_path)

        # build structure from scratch as a random disorded alloy (reduces to bulk metal for single component)
        if file_path is None and lattice_params is not None:
            self.create_lattice(lattice_params)
            
    def create_lattice(self, params):
        """Constructs a random disordered cubic alloy."""
        # read in parameters
        self.lattice = params['lattice']
        self.lattice_const = params['lattice_const']
        self.size = params['size']

        self.box = {
            'xlo': 0.0,
            'xhi': self.lattice_const*self.size[0],
            'ylo': 0.0,
            'yhi': self.lattice_const*self.size[1],
            'zlo': 0.0,
            'zhi': self.lattice_const*self.size[2],
        }

        for i, d in enumerate(['x', 'y', 'z']):
            self.boxsize[i] = self.box[f'{d}hi']-self.box[f'{d}lo']

        # composition in terms of at%
        self.composition = params['composition']

        tot_conc = sum(self.composition.values())
        if tot_conc != 100.0:
            raise ValueError(f'Combined all concentrations must sum exactly to 100. Calculated {tot_conc}')
        
        for el, conc in self.composition.items():
            self.composition_str += f'{el}{int(conc)}-'
        self.composition_str = self.composition_str[:-1]

        # composition in terms of number of atoms for each species (determined precisely later)
        at_composition = {}

        # enumerate LAMMPS atom types
        self.species_to_type = {}
        for i, el in enumerate(self.composition.keys()):
            i += 1
            self.species_to_type.update({el: i})
        
        # enumerate all translation vectors
        transv = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    transv.append(np.array([i,j,k]))

        # define positions in conventional unit cell
        sc_pos = [np.array([0, 0, 0], dtype=np.float32)]

        if self.lattice == 'bcc':
            unit_pos = sc_pos + [np.array([0.5, 0.5, 0.5], dtype=np.float32)]
        elif self.lattice == 'fcc':
            unit_pos = sc_pos + [np.array([0.5, 0.5, 0], dtype=np.float32), np.array([0.5, 0, 0.5], dtype=np.float32), np.array([0, 0.5, 0.5], dtype=np.float32)]

        # expand to supercell size by combining translation vectors with unit cell positions
        pos = []
        for p in unit_pos:
            for t in transv:
                pos.append(p+t)

        # remove duplicate positions
        upos = [pos[0]]
        for i, p in enumerate(pos[1:]):
            i += 1
            append = True
            for j, op in enumerate(pos[:i]):
                if p[0] == op[0] and p[1] == op[1] and p[2] == op[2]:
                    append = False
            if append:
                upos.append(self.lattice_const*p)

        self.positions = np.array(upos, dtype=np.float32)
        self.num_atoms = len(self.positions)

        # initialize ids and types arrays
        self.types = np.zeros(self.num_atoms, dtype=np.int8)
        self.ids = np.arange(1, self.num_atoms+1, dtype=np.int32)

        # determine number atoms to be assigned to each element
        for el, conc in self.composition.items():
            at_composition.update({el: round(self.num_atoms*conc/100)})

        # select random elements and add or remove single atoms until the total is correct
        random.seed()
        while sum(at_composition.values()) != self.num_atoms:
            rand_el_idx = random.randint(0, len(self.composition)-1)
            el = list(self.composition.keys())[rand_el_idx]

            val = sum(at_composition.values()) - self.num_atoms
            at_composition[el] -= sign(val)

        # generate a set of indices corresponding to random positions
        rand_pos_idx = {}
        while len(rand_pos_idx) < self.num_atoms:
            rand_pos_idx.update({random.randint(0, self.num_atoms-1): None})
        rand_pos_idx = list(rand_pos_idx)

        # assign elements to random positions by setting the types
        i = 0
        for el, n_at in at_composition.items():
            for j in range(n_at):
                self.types[rand_pos_idx[i]] = self.species_to_type[el]
                i += 1

    def load_from_file(self, read_path):
        super().load_from_file(read_path)
        
        # header comment metadata: bcc 3.07 3x3x3 W43-Mo57
        header_comment = strip_split(self.lines[0])

        self.lattice = header_comment[0]
        self.lattice_const = float(header_comment[1])
        self.size = strip_split(header_comment[2], 'x', as_type=int)
        
        self.composition_str = header_comment[3]
        for c in strip_split(self.composition_str, sep='-'):
            if c[:2] not in masses.keys():
                el = c[0]
                conc = float(c[1:])
            else:
                el = c[:2]
                conc = float(c[2:])
            self.composition[el] = conc

        # header lines
        for l, line in enumerate(self.lines):
            line_params = strip_split(line)
            if 'atoms' in line:
                self.num_atoms = int(line_params[0])
            elif 'atom types' in line:
                self.num_types = int(line_params[0])
            elif 'lo' in line:
                d = line_params[-1][0]
                self.box.update({
                    f'{d}lo': float(line_params[0]),
                    f'{d}hi': float(line_params[1]),
                })
            
            # start of body lines
            if any([True if bl in line else False for bl in ['Atoms', 'Masses']]):
                break

        for i, d in enumerate(['x', 'y', 'z']):
            self.boxsize[i] = self.box[f'{d}hi']-self.box[f'{d}lo']        

        eoh_l = l

        # body lines
        self.ids = np.zeros(self.num_atoms, dtype=np.int32)
        self.types = np.zeros(self.num_atoms, dtype=np.int8)
        self.positions = np.zeros((self.num_atoms, 3), dtype=np.float32)

        skip = 0
        for lo, line in enumerate(self.lines[eoh_l:]):
            line = line.strip()
            lo += eoh_l

            if skip:
                skip -= 1
                continue

            if line == 'Masses':
                for iline in self.lines[lo+2:lo+2+self.num_types]:
                    iline_params = strip_split(iline)
                    self.species_to_type.update({iline_params[-1]: int(iline_params[0])})
                skip = 2 + self.num_types

            elif line == 'Atoms':
                for li, iline in enumerate(self.lines[lo+2:lo+2+self.num_atoms]):
                    iline_params = strip_split(iline)
                    self.ids[li] = int(iline_params[0])
                    self.types[li] = int(iline_params[1])
                    self.positions[li] = np.array([float(val) for val in iline_params[2:]])
                skip = 2 + self.num_atoms

    def write_to_file(self, write_path):
        self.lines = []
        self.lines.append(f"{self.lattice}  {self.lattice_const:2.3f}  {self.size[0]}x{self.size[1]}x{self.size[2]}  {self.composition_str}\n")
        self.lines.append(f"{self.num_atoms} atoms")
        self.lines.append(f'{len(self.species_to_type)} atom types\n')
        self.lines.append(f"{self.box['xlo']:9.8f}  {self.box['xhi']:11.8f}  xlo xhi")
        self.lines.append(f"{self.box['ylo']:9.8f}  {self.box['yhi']:11.8f}  ylo yhi")
        self.lines.append(f"{self.box['zlo']:9.8f}  {self.box['zhi']:11.8f}  zlo zhi\n")
        self.lines.append('Masses\n')
        for el, t in self.species_to_type.items():
            self.lines.append(f"{t}  {masses[el]:3.4f}  # {el}")
        self.lines.append("\nAtoms\n")
        for a in range(self.num_atoms):
            self.lines.append(f"{self.ids[a]:<8}  {self.types[a]:<2}  {self.positions[a][0]:<12.8f}  {self.positions[a][1]:<12.8f}  {self.positions[a][2]:<12.8f}")

        super().write_to_file(write_path, append_newline=True)
            
    def insert_point_defect(self, defect_type: str, defect_species: str, defect_orientation: str):
        """Inserts a point defect at or near the center of the supercell."""
        center = self.lattice_const*np.array(self.size)/2
        
        dist = []
        for pos in self.positions:
            dist.append(np.linalg.norm(center-pos))

        ref_pos_i = dist.index(min(dist))

        # vacancy -> remove reference atom
        if defect_type == 'vac':
            vac_at_type = self.types[ref_pos_i]

            self.ids =  np.delete(self.ids, (ref_pos_i), axis=0)
            self.types =  np.delete(self.types, (ref_pos_i), axis=0)
            self.positions = np.delete(self.positions, (ref_pos_i), axis=0)

            self.num_atoms -= 1
            if len(set(self.types)) != self.num_types:
                raise RuntimeError(f"Inserting vacancy at {self.positions[ref_pos_i]} removed the last of atom type {vac_at_type}")
        
        # crowdion -> add atom between two others
        elif defect_type == 'crowd':
            if defect_orientation == '111':
                int_pos = self.positions[ref_pos_i] + self.lattice_const/4
            
            self.ids = np.insert(self.ids, (ref_pos_i), (self.num_atoms), axis=0)
            self.types =  np.insert(self.types, (ref_pos_i), self.species_to_type[defect_species], axis=0)
            self.positions = np.insert(self.positions, (ref_pos_i), (int_pos), axis=0)
        
            self.num_atoms += 1

        # dumbbell -> move reference atom over and add atom on other side
        elif defect_type == 'db':
            if defect_orientation == '100':
                spacing = np.array([self.lattice_const/6, 0, 0])
            elif defect_orientation == '111':
                spacing = np.array([self.lattice_const/6, self.lattice_const/6, self.lattice_const/6])

            ref_at_pos, int_pos = self.positions[ref_pos_i] - spacing, self.positions[ref_pos_i] + spacing

            self.ids = np.insert(self.ids, (ref_pos_i), (self.num_atoms), axis=0)
            self.types =  np.insert(self.types, (ref_pos_i), self.species_to_type[defect_species], axis=0)
            self.positions = np.insert(self.positions, (ref_pos_i), (int_pos), axis=0)
            self.positions[ref_pos_i] = ref_at_pos

            self.num_atoms += 1

class LmpLog:
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
            for line in self.lines[start[i]+1:stop[i]+1]:
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

class LmpDump(LmpFile):
    def __init__(self, file_path=None, content_str=None):
        super().__init__(file_path=file_path, content_str=content_str)
        
        # dictionary mapping timestep to a dictionary of numpy arrays where each atom corresponds to the same index 
        self.frames: dict[int, dict[str, np.ndarray]] = {}

        timestep = None
        for l, line in enumerate(self.lines):
            # save previous frame and initialize a new one
            if line.strip() == 'ITEM: TIMESTEP':
                if timestep is not None:
                    self.frames[timestep] = frame
                frame = {}
                timestep = strip_split(self.lines[l+1], as_type=int)[0]

            # define number of atoms
            elif line.strip() == 'ITEM: NUMBER OF ATOMS':
                frame['num_atoms'] = strip_split(self.lines[l+1], as_type=int)[0]
            
            # get box size (it will change from fix box/relax)
            elif line.strip() == 'ITEM: BOX BOUNDS pp pp pp':
                xlo, xhi = strip_split(self.lines[l+1], as_type=float)
                ylo, yhi = strip_split(self.lines[l+2], as_type=float)
                zlo, zhi = strip_split(self.lines[l+3], as_type=float)

                frame['box'] = {'xlo': xlo, 'xhi': xhi, 'ylo': ylo, 'yhi': yhi, 'zlo': xlo, 'zhi': xhi}
                frame['boxsize'] = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

            # read in per-atom data
            elif 'ITEM: ATOMS' in line.strip():
                # initialize data containers
                column_names = strip_split(re.sub('ITEM: ATOMS', '', line.strip()))
                frame.update({key: np.zeros(frame['num_atoms']) for key in column_names})

                # read in values for each atom and populate containers
                for a, atom in enumerate(self.lines[l+1:l+1+frame['num_atoms']]):
                    for v, val in enumerate(strip_split(atom, as_type=float)):
                        frame[column_names[v]][a] = val
                
                # combine related data like x, y, z -> (x, y, z)
                if set(['x', 'y', 'z']).issubset(column_names):
                    frame['position'] = np.column_stack((frame['x'], frame['y'], frame['z']))
                    frame.pop('x')
                    frame.pop('y')
                    frame.pop('z')
                
                # update data type of arrays
                if 'id' in frame.keys():
                    frame['id'] = frame['id'].astype(np.int32)
                if 'type' in frame.keys():
                    frame['type'] = frame['type'].astype(np.int8)
                for k in ['x', 'y', 'z', 'position']:
                    if k in frame.keys():
                        frame[k] = frame[k].astype(np.float32)

        # save last frame
        self.frames[timestep] = frame

    def write_structure_file(self, write_path: Path, lattice_params: dict, timestep = None):
        """Generate a LAMMPS data file from the dump data at a given timestep."""
        struct = self.to_struct(lattice_params, timestep=timestep)
        struct.write_to_file(write_path)

    def to_struct(self, lattice_params: dict, timestep = None):
        """Instantiate a LmpStructure object using dump data at a given timestep."""
        # determine frame to pull data from
        if timestep is None:
            timestep = list(self.frames.keys())[-1]
        else:
            timestep = int(timestep)
            if timestep not in self.frames.keys():
                raise KeyError(f'Dump file at {self.last_read_path} does not have the timestep {timestep}')
        
        frame = self.frames[timestep]
        if not set(['type', 'id', 'position']).issubset(frame.keys()):
            raise KeyError(f'Dump file at {self.last_read_path} must at least have the atom type, id, x, y, z coords to define a valid structure input file')
        
        # initialize structure and manually update attributes since create_lattice nor load_from_file was called
        struct = LmpStructure()

        struct.ids = frame['id']
        struct.types = frame['type']
        struct.positions = frame['position']

        struct.size = lattice_params['size']
        struct.box = frame['box']
        struct.boxsize = frame['boxsize']

        struct.lattice = lattice_params['lattice']
        struct.lattice_const = (product(frame['boxsize']) / product(lattice_params['size']))**(1/3)
        struct.num_atoms = frame['num_atoms']
        
        struct.composition_str = lattice_params['composition_str']
        for c in strip_split(struct.composition_str, sep='-'):
            if c[:2] not in masses.keys():
                el = c[0]
                conc = float(c[1:])
            else:
                el = c[:2]
                conc = float(c[2:])
            struct.composition[el] = conc
        
        struct.num_types = len(struct.composition)

        for i, el in enumerate(struct.composition.keys()):
            i += 1
            struct.species_to_type.update({el: i})

        return struct