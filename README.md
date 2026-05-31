The point of this package is to automate some common studies performed in LAMMPS, primarily focused on metallic systems.

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