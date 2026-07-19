This package supports research on concentrated alloys conducted by the RDMAP research group within the Penn State Nuclear Engineering department.

Three study classes are currently available: `MCMD` for evaluating short range order (SRO) in bcc refractory alloys, `PDI` for evaluating the distribution of point defect insertion energies due to variations in the local chemical environment, and `SCC` for creating single collision cascades to evaluate material damage performance and ballistic mixing. 

For each study, a brief workflow is provided along with the input keyword argument pairs required to run a study.

## `MCMD`

This study is designed to perform a series of independent hybrid Monte Carlo with molecular dynamics (MCMD) simulations on unique starting configurations. The final configuration for each simulation is then quenched and saved as part of a reusable dataset. 

### Workflow

1. Sample a random, separated, or B2 ordered configuration
2. Enthalpy minimize
3. Equilibrate to the target temperature in the NVT or NPT ensemble
4. Perform a MCMD run to sample from configuration space by minimizing the potential energy
5. Energy minimize the final configuration

### Input File

```yaml
name: <directory name of new study>
type: MCMD
dir: <parent or restart directory path>

lattice: <bcc>
lattice_const: <conventional cell length>
size: <box length in terms of number of replicated unit cells (e.g., 50 -> 50x50x50 box)>
composition:
    <element 1 in potential file>: <atomic percentage as a whole number>
    <element 2 in potential file>: <atomic percentage as a whole number>
    ...
order: <random, separated, B2>

pair_style: <LAMMPS pair style type>
potential: <filename of interatomic potential in LammpsUtils/potentials/>
skin: <skin distance for neighbor list>

members: <number of independent simulations>
timestep: <MD timestep in ps>
temperature: <Metropolis sampling and MD temperature>
ensemble: <langevin, npt>
Tdamp: <coupling or friction constant for thermostat>
Pdamp: <coupling constant for barostate (for npt)>
processors: <number of MPI ranks for each independent simulation to be run in parallel>

minimize: <minimization criteria for final quenching as a list [etol, ftol, maxiter, maxeval]>
equil: <number of equilibration timesteps before MCMD>
mc: <fix atom/swap criteria as a list [freq, nswaps, nsteps]>
snapshot: <number of timesteps between snapshots>
wc_shell: <number of shells to compute Warren-Cowley parameters for (default: 3, max: 5)>
```

### Example

This example studies the SRO in equiatomic W-Mo using an NPT ensemble.

```yaml
name: 1000K_random
type: MCMD
dir: /storage/group/xvw5285/default/LAMMPS/WMo/

lattice: bcc
lattice_const: 3.15
size: [10, 10, 10]
composition:
  W: 50
  Mo: 50
order: random

pair_style: eam/fs
potential: WMo.eam.fs
skin: 2.0

members: 1000
timestep: 0.005
temperature: 1000
ensemble: npt
Tdamp: 10.0
Pdamp: 5.0
processors: 8

minimize: [1.0e-7, 0.0, 10000, 1000000]
equil: 25k
mc: [50, 5, 20k]
snapshot: 1k

wc_shell: 5
```

## `PDI`

This study is designed to insert a point defect into a series of independent configurations and evaluate the distribution of _insertion_ energy. Note, the insertion energy does not contain chemical, electrostatic, or finite-size corrections. Point defect formation energy can be computed from the insertion energy via $$E_\text{form} = E_\text{ins} \pm \mu + qE_F + E_\text{corr},$$
where $+\mu$ corresponds to a vacancy and $-\mu$ for a self-interstitial. Note that the 0K lattice constant is used for both pristine and defective cells.

### Workflow

1. Sample a random, separated, or B2 ordered configuration, or load one from a dataset
2. Enthalpy minimize pristine cell
3. Insert a point defect on the lattice site closest to the center of the simulation box
4. Energy minimize the defective configuration
5. Compute the insertion energy $E_\text{ins} = E_\text{def} - E_\text{pris}$

### Input File

```yaml
name: <directory name of new study>
type: PDI
dir: <parent or restart directory path>

dataset: <path to directory containing LAMMPS data files>

OR

lattice: <bcc>
lattice_const: <conventional cell length>
size: <box length in terms of number of replicated unit cells (e.g., 50 -> 50x50x50 box)>
composition:
    <element 1 in potential file>: <atomic percentage as a whole number>
    <element 2 in potential file>: <atomic percentage as a whole number>
    ...
order: <random, separated, B2>

pair_style: <LAMMPS pair style type>
potential: <filename of interatomic potential in LammpsUtils/potentials/>
skin: <skin distance for neighbor list>

members: <number of independent simulations>
defect: <vac, int>
int_type: <crowd, db; type of interstitial structure>
int_species: <element name of interstitial>
int_orientation: <crystal direction indices as a string (e.g., 111 is the <111> direction)>
processors: <number of MPI ranks for each independent simulation to be run in parallel>

minimize: <minimization criteria for final quenching as a list [etol, ftol, maxiter, maxeval]>
```

### Example

This example creates M-V $\langle 111\rangle$ dumbbells (M = W, V) into the final configurations of a previous `MCMD` study. 

```yaml
name: sro_vac
type: PDI
dir: /storage/group/xvw5285/default/LAMMPS/WV/

dataset: /storage/group/xvw5285/default/LAMMPS/WV/1000K_random_000/dataset/

pair_style: eam/fs
potential: WV.eam.fs
skin: 2.0

defect: int
int_type: db
int_species: V
int_orientation: 111
members: 1000
processors: 4

minimize: [1.0e-7, 0.0, 10000, 1000000]
```

## `SCC`

This study is designed to create a single collision cascade from neutron irradiation in a series of independent simulations and evaluate the displacements per atom (DPA), number of Frenkel pairs, ballistic mixing, and structural energy change due to defect formation. 

A variety of parameters are used to fully define the PKA. The only required parameter is the distance from the center of the simulation box, expressed as a fraction of half the box length. For example, a value of 0.73 means the PKA will be chosen such that is 73% of the distance between the simulation center the boundary of the box, including the thermostatic shell. Thus a value of 0.0 is the box center and 1.0 is the box boundary.

Other parameters like the PKA type and direction are chosen randomly unless explicitly specified. The PKA type is sampled from the dicrete probability distribution defined by the composition, and the direction is determined by randomly sampling lattice sites on/near the sphere with a radius defined by the PKA distance (see previous paragraph). If the type and direction _are_ specified, then the atom of the desired type closest to the "true" position is chosen. Note, for dilute species, finding the right atom close by can be unlikely, thus the direction (and distance) can deviate significantly from the anticipated one.

System size is also very important for collision cascades since finite-size and thermostatic interactions should be minimal. However, a very large system is expensive, so a balance must be struck. One such rule of thumb is that the system should have 25k atoms per keV of the PKA energy. For reference, a typical PKA energy for heavy metals like tungsten under fission neutron irradiation is roughly 10-100 keV and thus systems tend to have a maximum of about 10M atoms. For these reasons, the default simulation box size is determined at runtime based on the assumed neutron energy and PKA type, though a size can be specified directly like usual.

### Workflow

1. Sample a random, separated, or B2 ordered configuration, or load one from a dataset
2. Determine the PKA type, position, and direction, then compute the velocity from the maximum energy transfer by a neutron (head-on collision)
3. If the simulation box size not specified, it is determined using the provided scaling factor (e.g., 25k atoms per keV)
4. Energy minimize and equilibrate with the NPT ensemble
5. Setup a thermostatic boundary and create a collision cascade

### Input File

```yaml
name: <directory name of new study>
type: SCC
dir: <parent or restart directory path>

dataset: <path to directory containing LAMMPS data files>

OR

lattice: <bcc>
lattice_const: <conventional cell length>
size: <optional; box length in terms of number of replicated unit cells (e.g., 50 -> 50x50x50 box)>
box_sf: <optional if size above is specified; number of atoms per keV of PKA energy (default: 25k)>
composition:
    <element 1 in potential file>: <atomic percentage as a whole number>
    <element 2 in potential file>: <atomic percentage as a whole number>
    ...
order: <random, separated, B2>

members: <number of independent simulations>
timestep: <timestep for equilibration>
temperature: <equilibration and thermostatic boundary temperature>
Tdamp: <equilibration temperature damping coefficient and thermostatic boundary Langevin friction coefficient>
Pdamp: <equilibration pressure damping coefficient>
processors: <number of MPI ranks for each independent simulation to be run in parallel>

bath: <thickness of thermostatic boundary in lattice constants>
neutron: <energy of neutron radiation projectile>
pka_dist: <distance from box center to PKA as a fraction/percentage (see description)> 
pka_type: <optional; species of PKA>

minimize: <minimization criteria for initial minimization and quenching as a list [etol, ftol, maxiter, maxeval]>
equil: <number of equilibration timesteps before cascade>
cascade: <cascade adaptive timestep criteria as a list [Nsteps, mindt, maxdt, maxdr]
snapshot: <number of timesteps between snapshots>
pe_thresh: <potential energy threshold for each species as a list so that only disrupted atoms are dumped>
```

## Notes

1. When restarting a simulation, the new input parameters are used even if they conflict with the previous ones. Care must be taken to avoid potential conflicts by only changing parameters delibrately. For example, if one ran a study with 25 members then restarted with 12, only those first 12 members will be included in the analysis. Also keep in mind that simulation results are only updated if the member ID is not included in the restart file.