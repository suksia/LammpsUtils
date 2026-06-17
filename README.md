The point of this package is to automate some common studies performed in LAMMPS, primarily focused on metallic systems.

## `GenerateConfigurations`

Scope: cubic binary concentrated alloys

This study is designed to generate a dataset of ready-to-use configurations in an ensemble representing a concentrated substitutional alloy. It consists of the the following workflow
1. Sample a random configuration
2. Energy minimize until internal pressure is relieved
3. Equilibrate to the target temperature
4. Perform a hybrid Monte Carlo + molecular dynamics run to approach a real configuration
5. Quench the final configuration

## `Diffusion`

Scope: cubic elemental metals

This study has the following workflow:
1. Equilibriate the pristine lattice
2. Run MD to allow for diffusion
3. Compute the total MSD
4. 
4. Remove thermal motion by energy minimizing snapshots from diffusion
5. Perform a Wigner-Seitz analysis to identify the vacancy position in each snapshot
6. Calculate the mean squared displacement and obtain the diffusion constant
7. Repeat 1-6 for different temperatures
8. Fit an Arrhenius plot for D(T) and obtain the activation energy

## `VacancyDiffusion`

Scope: cubic elemental metals

This study has the following workflow:
1. Equilibriate the pristine lattice
2. Insert a vacancy and perform a quick re-equilbriation
3. Run MD to allow the vacancy to diffuse
4. Remove thermal motion by energy minimizing snapshots from diffusion
5. Perform a Wigner-Seitz analysis to identify the vacancy position in each snapshot
6. Calculate the mean squared displacement and obtain the diffusion constant
7. Repeat 1-6 for different temperatures
8. Fit an Arrhenius plot for D(T) and obtain the activation energy







NOTE: ELEMENTS SHOULD BE SPECIFIED IN THE ORDER GIVEN BY THE POTENTIAL


state
|-- sim0
|   |- mem0
|   |   |- input_files -> all files required to run LAMMPS for member
|   |   |- status -> 0 = ready, 1 = running, 2 = finished
|   |   |- dir -> directory
|   |   |- any other kwargs that are associated with this member



|   |- mem1
|-- sim1
|
...