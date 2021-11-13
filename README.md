# Kinetic Polyelectrolyte gel

Codes supporting the publication "A kinetic model of a polyelectrolyte gel undergoing phase separation".

1. Implementation of the dynamic simulation for a costrained gel both for the electroneutral and non-electroneutral scenarios; the code uses finite element implementation in [FEniCS](https://www.https://fenicsproject.org/) (DynamicSim.py). 

2. Computation of the non homogeneous steady states for the a constrained non-electroneutral model (folder Equilibria): 
	- BifDiagram: main jupyter notebook in Julia based on the [BifurcationKit](https://github.com/rveltz/BifurcationKit.jl) package for the continuation of the equilibrium solution; the code automatically stores the solution, create plots and gif.
	- FunGel.jl: Julia module which contains the functional defining the equilibrium solution and other useful function.
	- PlotDiagram: Jupyter notebook in python for plotting the bifurcation diagram and creating a gif.

