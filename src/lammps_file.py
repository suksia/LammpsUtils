import logging, re, random
from pathlib import Path
from copy import deepcopy
import numpy as np
from utils import sign, strip_split
from masses import masses
import matplotlib.pyplot as plt

logger = logging.getLogger('LammpsUtils')
logging.getLogger("matplotlib").setLevel(logging.FATAL)

class LmpFile:
    def __init__(self, file_path: Path = None, content_str: str = None):
        self.fp = file_path
        if self.fp:
            self.fn = self.fp.name
        else:
            self.fn = None
        
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
                d.write(l+'\n')
        self.last_write_path = deepcopy(write_path)
        logger.debug(f'{self.__class__.__name__}: wrote lines to {write_path}')

class LmpInput(LmpFile):
    def add_params(self, params: dict):
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

class LmpStructure(LmpFile):
    """Input structure data file for LAMMPS which is a randomized bcc/fcc lattice of elements."""
    def __init__(self, struct_params: dict):
        super().__init__()
        self.fn = 'struct.in'

        self.lattice = struct_params['lattice']
        self.lattice_const = struct_params['lattice_const']
        self.size = struct_params['size']

        # composition in terms of at%
        self.composition: dict = struct_params['composition']
        self.composition_str = ''
        for el, conc in self.composition.items():
            self.composition_str += f'{el}{int(conc)}-'
        self.composition_str = self.composition_str[:-1]

        # composition in terms of number of atoms for each species
        self.at_composition = {}

        # dictionary items = species, type, position 
        self.atoms: dict[int, dict[str, str|int|np.ndarray]] = {}
        self.num_atoms = None
        
        self.lmp_types = {}
        for i, el in enumerate(self.composition.keys()):
            i += 1
            self.lmp_types.update({el: i})

        self.create_lattice()
    
    def create_lattice(self):
        """Defines a cubic crystal lattice as a dictionary with site positions, species, and LAMMPS types."""
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

        self.num_atoms = len(upos)

        # determine number atoms to be assigned to each element
        tot_conc = sum(self.composition.values())
        if tot_conc != 100.0:
            raise ValueError(f'Combined all concentrations must sum exactly to 100. Calculated {tot_conc}')

        for el, conc in self.composition.items():
            self.at_composition.update({el: round(self.num_atoms*conc/100)})

        # select random elements and add or remove single atoms until the total is correct
        while sum(self.at_composition.values()) != self.num_atoms:
            rand_el_idx = random.randint(0, len(self.composition)-1)
            el = list(self.composition.keys())[rand_el_idx]

            val = sum(self.at_composition.values()) - self.num_atoms
            self.at_composition[el] -= sign(val)

        # generate a set of indices corresponding to random positions
        rand_pos_idx = {}
        while len(rand_pos_idx) < self.num_atoms:
            rand_pos_idx.update({random.randint(0, self.num_atoms-1): None})
        rand_pos_idx = list(rand_pos_idx)

        # assign elements to random positions
        i = 0
        for el, n_at in self.at_composition.items():
            for j in range(n_at):
                rand_i = rand_pos_idx[i]
                self.atoms.update({i: {'position': upos[rand_i], 'species': el, 'type': self.lmp_types[el]}})
                i += 1

    def write_to_file(self, write_path):
        # write lines for a LAMMPS input file
        self.lines = []
        self.lines.append(f'{self.size[0]}x{self.size[1]}x{self.size[2]} {self.lattice} {self.composition_str}\n')
        self.lines.append(f'{self.num_atoms} atoms')
        self.lines.append(f'{len(self.composition)} atom types\n')
        self.lines.append(f'{0.0:9.8f}  {self.lattice_const*self.size[0]:11.8f}  xlo xhi')
        self.lines.append(f'{0.0:9.8f}  {self.lattice_const*self.size[1]:11.8f}  ylo yhi')
        self.lines.append(f'{0.0:9.8f}  {self.lattice_const*self.size[2]:11.8f}  zlo zhi\n')
        self.lines.append('Masses\n')
        for el in self.composition.keys():
            self.lines.append(f'{self.lmp_types[el]}  {masses[el]:3.4f}  # {el}')
        self.lines.append('\nAtoms\n')
        for i, at_dict in self.atoms.items():
            i += 1
            self.lines.append(f"{i}  {self.lmp_types[at_dict['species']]}  {at_dict['position'][0]:12.8f}  {at_dict['position'][1]:12.8f}  {at_dict['position'][2]:12.8f}")

        super().write_to_file(write_path)
            
    def insert_point_defect(self, defect_type: str, defect_species: str, defect_orientation: str):
        """Inserts a point defect at or near the center of the supercell."""
        center = self.lattice_const/2*np.array(self.size)

        dist = []
        for at_dict in self.atoms.values():
            dist.append(np.linalg.norm(center-at_dict['position']))

        ref_pos_i = dist.index(min(dist))
        ref_at_dict = self.atoms[ref_pos_i]

        # vacancy -> remove reference atom
        if defect_type == 'vac':
            self.num_atoms -= 1
            self.atoms.pop(ref_pos_i)
        
        # crowdion -> add atom between two others
        elif defect_type == 'crowd':
            self.num_atoms += 1

            if defect_orientation == '111':
                int_pos = ref_at_dict['position'] + self.lattice_const/4

            self.atoms.update({self.num_atoms-1: {'position': int_pos, 'species': defect_species, 'type': self.lmp_types[defect_species]}})
        
        # dumbbell -> move reference atom over and add atom on other side
        elif defect_type == 'db':
            self.num_atoms += 1

            if defect_orientation == '100':
                spacing = np.array([self.lattice_const/4, 0, 0])
            elif defect_orientation == '111':
                spacing = np.array([self.lattice_const/4, self.lattice_const/4, self.lattice_const/4])
            
            ref_at_pos, int_pos = ref_at_dict['position'] - spacing, ref_at_dict['position'] + spacing

            self.atoms[ref_pos_i]['position'] = ref_at_pos
            self.atoms.update({self.num_atoms-1: {'position': int_pos, 'species': defect_species, 'type': self.lmp_types[defect_species]}})
        
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