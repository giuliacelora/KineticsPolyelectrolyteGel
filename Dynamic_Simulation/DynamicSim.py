#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:57:42 2021

@author: celora
"""

import sys  

from dolfin import *
import csv
from numpy import pi,min, inf,empty,zeros,savetxt,array
import numpy as np
from ufl import tanh,sinh,cosh
import matplotlib.pyplot as plt
import pdb
import os
import importlib.util

snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "gmres",
                                          "maximum_iterations": 50,
                                          "absolute_tolerance": 1.0e-14,
                                         # "relative_tolerance":0,
                                          "report": False,
                                          "error_on_nonconvergence": False}}
#%%
""""
FIGURE 4 & 5 
""""

#define the model parameters
chi=0.95
alf=0.1
beta=0
G=1e-4
omega=0.025
Dm=10

Toll=5e-3
dtmax=20.0
dtmin=1e-8
timestep=1e-3
set_log_active(False)
tend=80000

lam=20.
tc=0
tc2=tend+2 
c0new=c0
# uncomment the following line for Figure 4
c0min=10**(-2.49)
c0=10**(-2.0)
# uncomment the following line for Figure 5 route 1
c0=10**(-2.49)
c0min=10**(-2.375)
# uncomment the following line for Figure 5 route 2
c0min=10**(-2.0)
c0=10**(-2.49)
#%%%
""""
FIGURE 10
""""
c0=10**(-3.1) # maximum concentration in the bath (initial)
c0min=10**(-4.) # minimum concentration in the bath
chi=0.8
alf=0.05
beta=4e-3
G=5e-4
omega=0.008
Dm=10

lam=0.01 # rate at which the concentration of ions in the bath changes 
tc=5000 # time at which the the concentration of ions in the bath drops from c0 -> c0min
tc2=14000 # time at which the the concentration of ions in the bath increases from c0min -> c0
Toll=8e-2 # tolerance for the time-stepping
dtmax=20.0 # maximum time-step
dtmin=1e-8 # minimum time-step
timestep=1e-3 # current time step
tend=1e4 # final time for the simulation

lam=0.01
tc=2.25e3
tc2=tc+1250

dt = Constant(timestep) # saves the current time step
dt2= Constant(1.0) # time step for the more accurate approximation for time-stepping dt2=dt/2

#%%%
# define the mesh, with restriction to the boundary in contact with the bath
gel=4
bath=10
inlet=20
interface=30
z0=1.0


mesh=IntervalMesh(500,0,1)

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
class Gel(SubDomain):
    def inside(self,x,on_boundary):
        return between(x[0],(0,z0))

class Interface(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],z0)   


Interface().mark(boundaries, inlet)

#%%%%% Initialisation of the solution with the homogeneous solution

def InitialConditions(cb,mesh):
    mus=ln(1-2*cb)
    mup=ln(cb)
    mum=ln(cb)

    Vmu=VectorElement("R",mesh.ufl_cell(),0,dim =2)

    QMu = FunctionSpace(mesh,Vmu)
    u=Function(QMu)
    u.interpolate(Constant((0.3,3)))
    v=TestFunction(QMu)
    psG,J=split(u)

    pmG=alf/J*1/2+sqrt((alf/J)**2*1/4+psG**2*cb**2/(1-cb)**2*exp(2*chi/J))
    vs,vp=split(v)
    M1 = (psG-(1-2*cb)*exp(-chi*(1.0-psG)/J-1/J-G*(J**2-1)/J))*vs
    M2 = (1-pmG*2+alf/J-psG-1/J)*vp
    M=(M1+M2)*dx
    solve(M==0,u)
    el=u(0.1,)
    psGel=el[0] 
    Jel=el[1] 
    pmGel=pmG((0.1,))
    ppGel=1-psGel-1/Jel-pmGel
    Phiel=1/2*ln(pmGel/ppGel)
    return [mus,mup,mum,psGel,ppGel,pmGel,Phiel,1/Jel]
    

#%% Definition of the measures
dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

#%% Definition of Functional space and functions
Vel=FiniteElement("Lagrange", mesh.ufl_cell(),1)
V=FunctionSpace(mesh,Vel)

Rel=FiniteElement("R",mesh.ufl_cell(),0)
VR=FunctionSpace(mesh,Rel)
#                [mus, mup, mum, phis, phip, phim, Phi, H^(-1)]
Qel=MixedElement([Vel, Vel, Vel, Vel,  Vel,  Vel,  Vel, Rel])
Hh = FunctionSpace(mesh,Qel)  

unew = Function(Hh) # new iterates with time step dt
usave = Function(Hh) # dummy function for saving the solution with no conflic
uhigh = Function(Hh) # more accurate solution for the time-adaptive stepping with time step dt/2
uold = Function(Hh) # contain solution at the previous time-step
dul = TrialFunction(Hh)
v= TestFunction(Hh)

#%% Dirichlet boundary conditions for the chemical potential, strongly imposed
bcs1 = DirichletBC(Hh.sub(0), Constant(ln(1-2*c0)), boundaries, inlet)
bcs2 = DirichletBC(Hh.sub(1), Constant(ln(c0)), boundaries, inlet)
bcs3 = DirichletBC(Hh.sub(2), Constant(ln(c0)), boundaries, inlet)
bcs=[bcs1,bcs2,bcs3]

#%%%%% computation of the initial conditions

uinit=InitialConditions(c0,mesh=mesh)
uinitold=InitialConditions(c0/2,mesh=mesh)

unew.interpolate(Constant(uinit))
uold.interpolate(Constant(uinitold))
uhigh.assign(unew)
usave.assign(unew)

#%% extra needed function

Vg=Expression(("x[0]",), degree=2) # needed for the extra contribution when rescaling from a moving to a fixed domain

#%%

def residual(u,uprev,dt):

    (mus,mup,mum,ps,pp,pm,Phi,invH)=split(u) #access the new solution

    (_,_,_,psprev,ppprev,pmprev,_,invHprev)=split(uprev) #access the previous solution

    (vs,vp,vm,vsmu,vpmu,vmmu,vPhi,vH)=split(v)   #access the trial functions

    bt=beta*invHprev #rescaled beta
    om=omega*invHprev #rescaled beta
    
    #volume fraction in the gel
    pn=1-ps-pm-pp
    pnprev=1-psprev-pmprev-ppprev

    #change in the lenght of the gel (inverse)
    dH=invH-invHprev
    
    #definition of the fluxes
    jsG= -(psprev*grad(mus)+ppprev*grad(mup)+pmprev*grad(mum))*invHprev
    jpG= - Dm* (ppprev*grad(mup))*invHprev+ppprev/psprev*jsG
    jmG= - Dm* (pmprev*grad(mum))*invHprev+pmprev/psprev*jsG
    Vn = -jsG -jpG - jmG
    JP=jpG+ppprev*Vn
    JM=jmG+pmprev*Vn
    JS=jsG+psprev*Vn
    # definition of the variational form for time dependent evolution
    # (the second term arises from the rescaling of the domain from [0,H] to [0,1])
    L1 = ( (ps-psprev)*vs+ 
          +dot(Vg,grad(ps))*dH/invH*vs
          -dot(JS,grad(vs))*invHprev*dt)*dx 
    L2= ( (pp-ppprev)*vp
          +dot(Vg,grad(pp))*dH/invH*vp
          -dot(JP,grad(vp))*dt*invHprev)*dx 
    
    L3 = ( (pm-pmprev)*vm+ 
          +dot(Vg,grad(pm))*dH/invH*vm
          -dot(JM,grad(vm))*invHprev*dt)*dx
    
    # definition of the electric problem
    Robcond=ln(1+2*c0*(cosh(Phi)-1)) # contribution from "Robin-type" BC for the bath
    L4 = (-bt**2*dot(grad(Phi),grad(vPhi))+(pp-pm+alf*pn)*vPhi)*dx -bt*(2*Robcond)**(1/2)*vPhi*ds(inlet)
    
    # implicit definition of the volume fraction in terms of the chemical potential:
    p=bt**2*dot(grad(Phi),grad(Phi))/2-3*om**2*dot(grad(ps),grad(ps))/2+G*(1-pn**2)/pn
    L5 = ((ln(ps)-mus+p+chi*(1-ps)*pn+pn)*vsmu-om**2*(ps-1)*dot(grad(ps),grad(vsmu)))*dx # ps=ps(mus,mup,mum)
    L6 = ((ln(pp)-mup+mus-ln(ps)-chi*pn+Phi)*vpmu-om**2*dot(grad(ps),grad(vpmu)))*dx # pp=pp(mus,mup,mum)
    L7 = (ln(pm)-mum+mup-ln(pp)-2*Phi)*vmmu*dx # pm=pm(mus,mup,mum)
    
    # integral condition for the definition of the size of the gel 
    L8=(invH-pn)*vH*dx

    return L1+L2+L3+L4+L5+L6+L7+L8 # full residual


#%% initialisation of the simulation
t=0.0 # current timepoint
set_log_active(False)


c0t=c0 # concentration of ions in the bath at time t

#%% Definition of the variational problems
F=residual(unew,uold,dt)
J=derivative(F,unew,dul)
P=NonlinearVariationalProblem(F,unew,bcs,J);
solverLow= NonlinearVariationalSolver(P) # solver for the new solution with time-step dt

Fhigh=residual(uhigh,uold,dt2)
Jhigh=derivative(Fhigh,uhigh,dul)
Phigh=NonlinearVariationalProblem(Fhigh,uhigh,bcs,Jhigh);
solverHigh= NonlinearVariationalSolver(Phigh) # solver for the more accurate solution with time-step dt/2


#%% variables to save the solution
cbath=[]
tvec=[]
Hvec=[]
variables=['mus','mup','mum','ps','pp','pm','Phi']

solution=usave.vector()[:]
Hvec.append(1/unew.sub(n)((0.1,)))
tvec.append(0)

#%% Time loop with time-step controll
while t<tend:
    dt.assign(timestep)        
    dt2.assign(timestep/2)
    if t>tc:
        if t<tc2:
            c0new=(c0-c0min)*exp(-lam*(t-tc))+c0min # increase in c0
        else:
            c0new=(c0min-c0)*exp(-lam*(t-tc2))+c0 # decrease in c0
        # as c0 changes we need to update the boundary conditions
        bcs1 = DirichletBC(Hh.sub(0), Constant(ln(1-2*c0new)), boundaries, inlet)
        bcs2 = DirichletBC(Hh.sub(1), Constant(ln(c0new)), boundaries, inlet)
        bcs3 = DirichletBC(Hh.sub(2), Constant(ln(c0new)), boundaries, inlet)
        bcs=[bcs1,bcs2,bcs3]
        F=residual(unew,uold,dt)
        J=derivative(F,unew,dul)
        P=NonlinearVariationalProblem(F,unew,bcs,J);
        solverLow= NonlinearVariationalSolver(P)
        
        Fhigh=residual(uhigh,uold,dt2)
        Jhigh=derivative(Fhigh,uhigh,dul)
        Phigh=NonlinearVariationalProblem(Fhigh,uhigh,bcs,Jhigh);
        solverHigh= NonlinearVariationalSolver(Phigh)
    try:
        solverLow.solve()   
        Conv=True
    except:
        Conv=False
    if Conv:
        try:
            solverHigh.solve()  
            Conv=True
        except:
            Conv=False
    if Conv:

        if n>0:
            eta=np.sqrt(assemble(dot(uhigh-unew,uhigh-unew)*dx))
        else:               
            eta=Toll/2
        
        if (eta<Toll) or timestep==dtmin:
            t=t+timestep
    
            timestep=np.max([np.min([dtmax,Toll/2/eta*timestep]),dtmin])   
            assign(uold, unew) # update with the new soltion
            assign(uhigh,unew)

            n+=1 
            print('Successful:',timestep)
            print('Size:',1/unew.sub(len(variables))((0.1,)))
     
            assign(usave,unew)
            solution=np.column_stack((solution,usave.vector()[:]))
            Hvec.append(1/unew.sub(len(variables))((0.1,)))
            tvec.append(t)
            cbath.append(c0new)
        else:
            timestep=np.max([np.min([dtmax,Toll/2/eta*timestep]),dtmin])   
            
            assign(unew, uold) # new step failed, update with the previous solution
            assign(uhigh,uold)
    else:
        if timestep==dtmin:
            print("Problem: Minimum Time Step reached")
            break
        else:
            timestep=np.max([np.min([dtmax,timestep/2]),dtmin])   
            assign(unew, uold)
            assign(uhigh,uold)

tvec=array(tvec)
Hvec=array(Hvec)
cbath=array(cbath)  
#%% saving solution
variables=['mus','mup','mum','ps','pp','pm','Phi'] 
dof={}
sol={}
for ind in range(len(variables)):
    dof[variables[ind]] = Hh.sub(ind).dofmap().dofs()[:]
    sol[variables[ind]]= solution[dof[variables[ind]],:]

import csv
direct='Simulation/sim1/'
try:
    os.makedirs(direct)
    os.makedirs(direct+'gif/')
    print('Folder created! Saving the new solution')
except:
    print('Folder already exist! Saving the new solution')
    
for nam in variables:
    with open(direct+nam+'.csv', 'w') as csvfile:
         csv.writer(csvfile, delimiter=' ').writerows(sol[nam])
    csvfile.close() 
with open(direct+'time.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerow(tvec)
csvfile.close() 
with open(direct+'phi0.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerow(cbath)
csvfile.close() 
with open(direct+'H.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerow(Hvec)
csvfile.close() 

#%% Plot of the soltution for gif
import matplotlib
matplotlib.rc('axes',linewidth=1.25,labelsize=20)
matplotlib.rcParams['text.usetex'] = True
fig_vec=[]
direct='Simulation/sim1/'
C1=[255/256,205/256,0,1]
C2=[0/256,102/256,204/256,1]
C3=[0/256,0/256,102/256,1]
xx=np.linspace(0,1,501)
t1=4000
t2=30000
LL=np.linspace(t1,t2,80)
Nsave=0
for el in LL:
    nim=np.where(tvec>el)[0][0]
    fig=plt.figure(figsize=(12,6),dpi=250)
    ps=sol['ps'][-1::-1,nim]
    pm=sol['pm'][-1::-1,nim]
    pp=sol['pp'][-1::-1,nim]
    Phi=sol['Phi'][-1::-1,nim]
    pn=1-ps-pp-pm
    q=alf*pn-pm+pp
    grid = plt.GridSpec(2, 3)    
    plt.subplot(grid[0,0:2])

    plt.fill_between(xx*Hvec[nim],ps*0,ps,color=C3)
    plt.fill_between(xx*Hvec[nim],ps,ps+pn,color=C2)
    plt.fill_between(xx*Hvec[nim],ps+pn,ps/ps,color=C1)
   
    plt.yticks([0.25,0.50,0.75])
    plt.tick_params(direction="in",length=8,pad=2)
    plt.ylim((0,1))
    plt.xlim((0,5))

    plt.subplot(grid[1,0])
    plt.plot(xx,q,linewidth=2.5,color='indigo')
    plt.tick_params(direction='in')
    plt.ylim(-0.05,0.05)
    plt.xlim(0,1)
    plt.xlabel('$x$')

    plt.title('$q(x,t)$',size=20)
    
    plt.subplot(grid[1,1])
    plt.plot(xx,Phi,linewidth=2.5,color='darkgreen')
    plt.tick_params(direction='in')
    plt.title('$\\Phi(x,t)$',size=20)
    plt.xlim(0,1)
    plt.ylim(0,5.5)
    plt.xlabel('$x$')
    plt.subplot(grid[0,2])
    plt.plot(tvec[:nim+1],cbath[:nim+1],linewidth=2.,color='k')
    plt.plot(tvec[nim],cbath[nim],'o',color='gold',markersize=10)

    plt.tick_params(direction='in')
    plt.title('$\phi_0(t)$',size=20)
    plt.xlim(t1,t2)
    plt.ylim(5e-5,5e-3)
    plt.yscale('log')        

    plt.xlabel('$x$')
    plt.subplot(grid[1,2])
    plt.plot(tvec[:nim+1],Hvec[:nim+1],linewidth=2.,color='k')
    plt.plot(tvec[nim],Hvec[nim],'o',color='gold',markersize=10)

    plt.tick_params(direction='in')
    plt.title('$h(t)$',size=20)
    plt.xlim(t1,t2)
    plt.ylim(3.0,4.75)


    plt.xlabel('$x$')
    plt.tight_layout(pad=0.02)

    plt.savefig(direct+'gif/im'+str(Nsave)+'.png', dpi=250,
            orientation='portrait',transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    Nsave+=1
    plt.show()
    
    
#%% Create GIF if package available 

package_name = 'imageio'

spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed. Gif can not be created")
else:
    import imageio
    frames=[]
    for A in range(Nsave):
        filename=direct+'gif/im'+str(A)+'.png'
        frames.append(imageio.imread(filename))
    kwargs_write={'fps':12.0,'quantizer':'wu'}
    imageio.mimsave(direct+'sim.gif',frames,'GIF-FI',**kwargs_write)
