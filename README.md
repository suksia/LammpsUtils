This package supports research on concentrated alloys conducted by the RDMAP research group within the Penn State Nuclear Engineering department.

Three study classes are currently available: `MCMD` for evaluating short range order (SRO) in bcc refractory alloys, `PDI` for evaluating the distribution of point defect insertion energies due to variations in the local chemical environment, and `SCC` for creating single collision cascades to evaluate material damage performance and ballistic mixing. 

For each study, a brief workflow is provided along with the input keyword argument pairs required to run a study.

## `MCMD`

Acronym: **M**onte **C**arlo w/ **M**olecular **D**ynamics

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
Pdamp: <coupling constant for barostat (for npt)>
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

## `REPT`

Acronym: **R**eplic **E**xchange / **P**arallel **T**empering

### Workflow

### Input File

### Example

## `ODT`

Acronym: **O**rder-**D**isorder **T**ransition

This study computes the Gibbs free energy difference between ordered $(T\to 0)$ and disordered $(T\to\infty)$ ensembles as a function of temperature. 

For a real multicomponent alloy, understanding short-range order (SRO) is crucial because it determines the typical motifs for the local chemical environment. SRO is dictated by the Gibbs free energy $$G=H-TS,$$ where the $G$ is at a global minimum for the equilibrium ensemble. For any two unique ensembles (with different partition functions), $\Delta G = G_1-G_2$ describes the relative stability or thermodynamic preference for either ensemble. The one closer to the global minimum is the more favored state. 

A competition exists between the enthalpy, reduced by ordering or separating, and the entropy, increased by disordering. In other words, ordering is favored when the number of microstates lost (due to ordering) is relatively small compared to the gain in negative enthalpy. If it is assumed that the real system, the equilibrium ensemble in this case, lies somewhere between the ordered and disordered states, comparing their free energy gives some insight into the proximity to each limit. Furthermore, the temperature where $\Delta G = G_\text{order} - G_\text{random} = 0$ is called the *order-disorder transition temperature*.

The free energy for the ideal limits can be computed directly because $$s(T\to 0)\approx 0, \qquad s(T\to\infty)=-k_\text{B}\sum_i x_i \ln x_i,$$ where $x_i$ is the concentration of each alloy species, expressed in at.%, and the configurational entropy is in its intensive form. The real system entropy is bounded by these limits as SRO only decreases the number of occupied microstates. It can, however, be estimated with good accuracy using techniques like 
thermodynamic integration.

### Workflow

1. Generate ordered (or separated) and random configurations
2. Equilibrate to $T_i$
3. Repeat 1-2 for many configurations and compute $\langle H \rangle$ for each ensemble
4. Repeat 1-3 for multiple temperatures
5. Compute $\Delta G(T) = G_\text{random} - G_\text{order}$ 

### Input File

```yaml
name: <directory name of new study>
type: ODT
dir: <parent or restart directory path>

lattice: <bcc>
lattice_const: <conventional unit cell length>
size: <box length in terms of number of replicated unit cells (e.g., 50 -> 50x50x50 box)>
composition:
    <element 1 in potential file>: <atomic percentage as a whole number>
    <element 2 in potential file>: <atomic percentage as a whole number>
    ...
order: <separated, B2>

pair_style: <LAMMPS pair style type>
potential: <filename of interatomic potential in LammpsUtils/potentials/>
skin: <skin distance for neighbor list>

members: <number of independent simulations for each temperature>
processors: <number of MPI ranks for each independent simulation to be run in parallel>

timestep: <MD timestep in ps>
temperature: <temperature range parameters as a list [Tmin, Tmax, Ntemps]>
Tdamp: <damping constant for thermostat>
Pdamp: <damping constant for barostat>
minimize: <minimization criteria as a list [etol, ftol, maxiter, maxeval]>

equil: <number of equilibration timesteps>
thermo: <frequency (in timesteps) for computing enthalpy during equilibration>
time_avg: <number of timesteps for computing time-averaged enthalpy>
```



## `PDI`

Acronym: **P**oint **D**efect **I**nsertion

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

## `PDM`

Acronym: **P**oint **D**efect **M**igration

This study inserts a point defect into a configuration and runs an MD loop to allow for diffusion via defect migration. Diffusion is evaluated using the mean squared displacement (MSD), where the squared displacement $|\mathbf{r}(t)-\mathbf{r}_0|^2$ is computed for every particle in a given group at some time $t$, and then averaged over all particles. $$\text{MSD} = \langle |\mathbf{r}(t)-\mathbf{r}_0|^2\rangle$$ The quality of MSD statistics stems from the number of particles used to compute the average, so for individual point defects many different configurations must be used. Similarly, the amount of mass diffusion is small as it can only proceed via the migration of individual point defects, so even the diffusion of alloy species must be averaged over many different configurations.

If an MSD curve is typical, in that a linear regime exists corresponding to steady-state diffusion (Brownian motion), a straight line can be fit which has the slope $6D$, where $D$ is the diffusivity. Futhermore, an effective point defect migration energy $E_m$ can be obtained from $$D(T) = D_0\exp\left(-\frac{E_m}{k_\text{B}T}\right),$$ which requires computing the diffusivity at multiple temperatures and fitting a straight line to the plot of $\ln D(T) = a/T + b$, where $a=-E_m/k_\text{B}$.  

### Workflow

1. Sample a random, separated, or B2 ordered configuration, or load one from a dataset
2. Insert a point defect on the lattice site closest to the center of the simulation box
3. Minimize to 0K
4. Equilibrate with NPT to desired temperature
5. Run diffusion with NVT
6. Minimize snapshots captured between jumps to remove thermal displacements
7. Analyze snapshots with Wigner-Seitz analysis to obtain point defect trajectory
8. Repeat 1-7 for many configurations
9. Compute mean squared displacement for the point defect and each alloy species over all configurations
10. Fit steady state MSD curves with straight lines to obtain diffusivities
11. (Optional) Repeat 1-10 for multiple temperatures and compute the effective migration energy

### Input File

```yaml
name: <directory name of new study>
type: PDM
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

members: <number of independent configurations simulated per temperature>
temperature: <single value or a list of temperatures to run PD diffusion>
timestep: <timestep for equilibration and diffusion>
Tdamp: <equilibration temperature damping coefficient and diffusion Langevin friction coefficient>
Pdamp: <equilibration pressure damping coefficient>
processors: <number of MPI ranks for each independent simulation to be run in parallel>
minimize: <minimization criteria for initial minimization and quenching as a list [etol, ftol, maxiter, maxeval]>
snapshot: <frequency (in timesteps) of snapshots during diffusion>
equil: <number of equilibration timesteps>
diffusion: <number of diffusion timesteps>

defect: <vac, int>
int_type: <crowd, db; type of interstitial structure>
int_species: <element name of interstitial>
int_orientation: <crystal direction indices as a string (e.g., 111 is the <111> direction)>
```

### Example




## `CC`

Acronym: **C**ollision **C**ascade

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