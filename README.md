# ElectronSpinDynamics

Schulten-Wolynes and Semiclassical electronic spin dynamics implemented by Julia

[![Build Status](https://github.com/KenHino/ElectronSpinDynamics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KenHino/ElectronSpinDynamics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KenHino/ElectronSpinDynamics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KenHino/ElectronSpinDynamics.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

[`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable/) is used for ensemble semiclassical simulation.

## Installation

1. Install Julia

See [julialang.org](https://julialang.org/) for installation instructions.
I personally recommend using [Juliaup](https://github.com/JuliaLang/juliaup) to install Julia.

2. Clone the repository

```bash
git clone https://github.com/KenHino/ElectronSpinDynamics.jl.git
cd ElectronSpinDynamics.jl
```

3. Install the package

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

4. Run the tests

```bash
julia --project=. -e 'using Pkg; Pkg.test("ElectronSpinDynamics")'
```

5. Examples

```bash
cd example
julia --project=.. --threads 4 tutorial.jl
```

6. (optional) You can perform simulation without input file. See function such `SC` and `SW`.

7. (optional) The results are exported in HDF format. You can access the data by Python (needless to say, Julia as well).

```python
# Access results by Python
import h5py
f = h5py.File('example/SC/results.h5', 'r')
B005_SC_Tp = f['B=0.05']['T+'][:]
B005_SC_T0 = f['B=0.05']['T0'][:]
B005_SC_S = f['B=0.05']['S'][:]
B005_SC_Tm = f['B=0.05']['T-'][:]
B005_SC_time = f['B=0.05']['time_ns'][:]
```

Input file is [example/input.ini](example/input.ini).

## Input file

Hamiltonian
<img width="1090" height="866" alt="image" src="https://github.com/user-attachments/assets/f0348962-a38e-4626-ab37-b2a482624280" />


- `J`: Exchange coupling constant in mT (divided by absolute value of γe)
- `D`: D tensor in mT
- `kS`: Singlet rate constant in μs-1
- `kT`: Triplet rate constant in μs-1
- `I`: Multiplicity of the nuclei (not quantum numbers) when I=3, nucleus is nitrogen, when I=2, nucleus is hydrogen. other nuclei are not supported yet.
- `An`: Nuclear hyperfine coupling constants in mT (asymmetric is not supported yet)
- `out`: Output folder
- `B`: Magnetic field in mT
- `simulation_time`: Simulation time in ns (**not μs**)
- `dt`: Time step in ns
- `N_samples`: Number of samples

**Caution: The other parameters are not used currently! But for compatibility, I left them here.**

```ini
[system variables]
J = 0.224
D = -0.2533333333333333 -0.0 -0.0 -0.0 -0.2533333333333333 -0.0 -0.0 -0.0 +0.5066666666666666
kS = 1.0
kT = 1.0
[electron 1]
g = 2.0023193
I = 3 2 2 2 2 2 2 2 2 2 3
N_I = 1 1 1 1 1 1 1 1 1 1 1
A1 = 0.5141406139911681 0.0 0.0 0.0 0.5141406139911681 0.0 0.0 0.0 0.5141406139911681
A2 = -0.13706792618414612 -0.0 -0.0 -0.0 -0.13706792618414612 -0.0 -0.0 -0.0 -0.13706792618414612
A3 = -0.13706792618414612 -0.0 -0.0 -0.0 -0.13706792618414612 -0.0 -0.0 -0.0 -0.13706792618414612
A4 = -0.13706792618414612 -0.0 -0.0 -0.0 -0.13706792618414612 -0.0 -0.0 -0.0 -0.13706792618414612
A5 = -0.44033852832217035 -0.0 -0.0 -0.0 -0.44033852832217035 -0.0 -0.0 -0.0 -0.44033852832217035
A6 = 0.4546400686867858 0.0 0.0 0.0 0.4546400686867858 0.0 0.0 0.0 0.4546400686867858
A7 = 0.4546400686867858 0.0 0.0 0.0 0.4546400686867858 0.0 0.0 0.0 0.4546400686867858
A8 = 0.4546400686867858 0.0 0.0 0.0 0.4546400686867858 0.0 0.0 0.0 0.4546400686867858
A9 = 0.4262605982027767 0.0 0.0 0.0 0.4262605982027767 0.0 0.0 0.0 0.4262605982027767
A10 = 0.4233203613613487 0.0 0.0 0.0 0.4233203613613487 0.0 0.0 0.0 0.4233203613613487
A11 = 0.1784350286060594 0.0 0.0 0.0 0.1784350286060594 0.0 0.0 0.0 0.1784350286060594
[electron 2]
g = 2.0023193
I = 2 3 3 2 2 2 2
N_I = 1 1 1 1 1 1 1
A1 = 1.6045 0.0 0.0 0.0 1.6045 0.0 0.0 0.0 1.6045
A2 = 0.32156666666666667 0.0 0.0 0.0 0.32156666666666667 0.0 0.0 0.0 0.32156666666666667
A3 = 0.1465 0.0 0.0 0.0 0.1465 0.0 0.0 0.0 0.1465
A4 = -0.278 -0.0 -0.0 -0.0 -0.278 -0.0 -0.0 -0.0 -0.278
A5 = -0.3634 -0.0 -0.0 -0.0 -0.3634 -0.0 -0.0 -0.0 -0.3634
A6 = -0.4879 -0.0 -0.0 -0.0 -0.4879 -0.0 -0.0 -0.0 -0.4879
A7 = -0.5983 -0.0 -0.0 -0.0 -0.5983 -0.0 -0.0 -0.0 -0.5983
[simulation parameters]
simulation_type = SW
output_folder = out
seed = 42 99
B = 0.05
initial_state = singlet
simulation_time = 201.0
dt = 1.0
N_krylov = 7
integrator_tolerance = 1e-08
N_samples = 1000000
```

## References
- SW theory:
  `Schulten, Klaus, and Peter G. Wolynes. "Semiclassical description of electron spin motion in radicals including the effect of electron hopping." The Journal of Chemical Physics 68.7 (1978): 3292-3297.`
- SC theory:
  - w/o D and J:
    `Manolopoulos, David E., and P. J. Hore. "An improved semiclassical theory of radical pair recombination reactions." The Journal of chemical physics 139.12 (2013).`
  - with kS != kT:
    `Lewis, Alan M., David E. Manolopoulos, and P. J. Hore. "Asymmetric recombination and electron spin relaxation in the semiclassical theory of radical pair reactions." The Journal of Chemical Physics 141.4 (2014).`
  - with D and J:
    `Fay, Thomas P., et al. "How quantum is radical pair magnetoreception?." Faraday discussions 221 (2020): 77-91.`
