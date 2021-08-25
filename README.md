# ModellingPaper
DynamicSim.py : dynamic simulation with an initially electro-neutral gel driven to change state by changing the concentration of ions in the bath. Implemented in Fenics

Equilibria: Folder containing the script for the computation of the bifurcation diagrams

	- BifDiagram: main jupyter notebook in julia for the continuation of the equilibrium solution; the code automatically stores the solution, create plots and 				gif.
	- FunGel.jl: julia module which contains the functional defining the equilibrium solution and other useful function.
	- PlotDiagram: jupyter notebook in python for plotting the bifurcation diagram and creating a gif.
