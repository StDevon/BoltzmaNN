# BoltzmaNN
The package that solves the phase-space Boltzmann equation for new comological species using physics-informed neural networks.

The network is constructed using PyTorch objects and methods and defined in 'boltzmann.py' and 'pinn.py'. The parameters of an example model (axions from lepton-flvaour violating tau lepton decays) are specified in 'parameters.py'. Particle physics processes and comological processes are implemented in 'physics.py'. 'cubic_spline.py' contains the cubic spline implementation for PyTorch tensors to treat the evolution of the entropy degrees of freedom. Notebook 'example.ipynb' provides an example of the network implementation to solve the Boltzmann equation for a specific model. We also use [PyBolt](https://github.com/Maxim-Laletin/PyBolt) as a classical numerical solver for the phase-space Boltzmann equation to compare the results. 

Authors: Michał Łukawski, Maxim Laletin 
