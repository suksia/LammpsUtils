This package supports research on concentrated alloys conducted by the RDMAP research group within the Penn State Nuclear Engineering department.

Two study classes are currently available: `MCMD` for evaluating short range order (SRO) in bcc refractory alloys, and `PDI` for evaluating the distribution of point defect insertion energies due to variations in the local chemical environment.

For each study, a brief workflow is provided along with the input keyword argument pairs required to run a study.

## `MCMD`

__Scope:__ cubic concentrated alloys

This study is designed to perform a series of independent hybrid Monte Carlo with molecular dynamics (MCMD) simulations on unique starting configurations. The final configuration for each simulation is then quenched and saved as part of a reusable dataset. 

### Workflow:

1. Sample a random, separated, or B2 ordered configuration
2. Enthalpy minimize
3. Equilibrate to the target temperature in the NVT or NPT ensemble
4. Perform a MCMD run to sample from configuration space by minimizing the potential energy
5. Energy minimize the final configuration

### Input File: 

```yaml
name: <directory name of new study>
type: MCMD
dir: <parent or restart directory path>

lattice: <bcc>
lattice_const: <conventional cell length>
size: <list of supercell replications (e.g., [3, 3, 3])>
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

## `PDI`

__Scope:__ cubic concentrated alloys

This study is designed to insert a point defect into a series of independent configurations and evaluate the distribution of _insertion_ energy. Note, the insertion energy does not contain chemical, electrostatic, or finite-size corrections. Point defect formation energy can be computed from the insertion energy via $$E_\text{form} = E_\text{ins} \pm \mu + qE_F + E_\text{corr},$$
where $+\mu$ corresponds to a vacancy and $-\mu$ for a self-interstitial. Note that the 0K lattice constant is used for both pristine and defective cells.

### Workflow:

1. Sample a random, separated, or B2 ordered configuration, or load one from a dataset
2. Enthalpy minimize pristine cell
3. Insert a point defect on the lattice site closest to the center of the simulation box
4. Energy minimize the defective configuration
5. Compute the insertion energy $E_\text{ins} = E_\text{def} - E_\text{pris}$

### Input File: 

```yaml
name: <directory name of new study>
type: PDI
dir: <parent or restart directory path>

dataset: <path to directory containing LAMMPS data files>

OR

lattice: <bcc>
lattice_const: <conventional cell length>
size: <list of supercell replications (e.g., [3, 3, 3])>
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
int_orientation: <crystal direction indices as a list (e.g., [1, 1, 1] is the <111> direction)>
processors: <number of MPI ranks for each independent simulation to be run in parallel>

minimize: <minimization criteria for final quenching as a list [etol, ftol, maxiter, maxeval]>
```

## Notes

1. When calculating the Warren-Cowley parameters, the initial lattice constant is used rather than the version containing volume relaxations. If the change in lattice constant is small, then this note can be safely ignored, otherwise the wrong amount of neighbors will be counted for each shell. In bcc systems, the distances corresponding to the middle of the gap between nearest neighbor shells are $a_0\times[0, 0.933, 1.207, 1.536, 1.695, 1.866]$. Since the smallest gap is $0.159a_0$, the limit is when $\Delta a_0 < 0.159a_0$. For $a_0\approx 3.075\,\AA$, this is $\Delta a_0 = 0.488\,\AA$. A good initial lattice constant usually results relaxations on the order $\pm0.01-0.1 \AA$, which is far below the limit.
