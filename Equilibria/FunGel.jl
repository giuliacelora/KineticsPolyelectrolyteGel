module CPDE
export init_par, wrap_F,wrap_Jacobian, meanval 
using LinearAlgebra, SparseArrays


function wrap_F(ps,pn,Phi,B,invH,OpDxx,OpDx,par)

    bt =par["beta"]*invH
    om =par["omega"]*invH
    chi=par["chi"]
    alf=par["alf"]
    G  =par["G"]
    c0 =par["c0"]
    dx =par["dx"]
    nx =par["nx"]
    F=zeros((nx+1)*4+1)
    Robcond=(2*log(1+2*c0*(cosh(Phi[end])-1)))^(1/2)

    p=bt^2*(OpDx*Phi).^2/2+G*(1 .-pn.^2)./pn-om^2*(OpDx*ps).^2/2    

    p[1]=G*(1-pn[1]^2)./pn[1]
    p[end]=G*(1-pn[end]^2)./pn[end]+log(1.0+2*c0*(cosh(Phi[end])-1.0))
    F1=ps .-(1-2*c0)*exp.(-chi*(1 .-ps).*pn-pn-p-(ps .-1).*B)

    pm=ps*c0/(1-2*c0).*exp.(chi.*pn-B+Phi)
    pp=ps*c0/(1-2*c0).*exp.(chi.*pn-B-Phi)

    F2=(1 .-ps-pm-pp-pn)
    fac=2*ps*c0/(1-2*c0).*exp.(chi.*pn-B)

    F3=bt^2*OpDxx*Phi+(alf.*pn-fac.*sinh.(Phi))

    F3[end]=F3[end]-2*Robcond/dx*bt  # using ghost point to apply Robin-type condition from the bath

    
    F4=B-om^2*OpDxx*ps    

    v=ones(nx+1)*dx
    v[1]=dx/2
    v[end]=dx/2
    F5=(invH-v'*pn)
    F[1:nx+1]=F1
    F[nx+2:2*nx+2]=F2
    F[2*nx+3:3*nx+3]=F3
    F[3*nx+4:4*nx+4]=F4

    F[end]=F5

    return F
end


function init_par(nx)
    par=Dict()
    # number of lattice sites.
    par["nx"]=nx
    # spacing of the lattice.
    hx=1/nx; 
    par["dx"]=hx
    # spatial variable
    spx=range(0,stop=nx,length=nx+1)*hx;
    par["spx"]=spx

    # gel parameters 
    par["c0"]=1e-3
    par["alf"]=0.05
    par["G"]=5e-4
    par["chi"]=0.8
    par["omega"] = 1e-2;
    par["beta"]=1e-3;
    OpDxx,OpDx=init_diff(par,nx) # generate the differential operators
    return par,OpDxx,OpDx
end

function init_diff(par,nx)
        hx=1/nx
        D = -2*sparse(I, nx+1, nx+1);
        E = spdiagm(1 => ones(nx),-1=>ones(nx))
        A = (E+D)/hx^2;
        # Laplacian operator with neumann condition applied by ghost point
        OpDxx=A 
        OpDxx[1,2]=2/hx^2;
        OpDxx[end,end-1]=2/hx^2;
        # First Order Differential Operator (centered difference scheme)
        OpDx = spdiagm(1 => ones(nx),-1=>-ones(nx))
        OpDx = OpDx/(2*hx); 
        # Note that due to the Neumann Condition the value at the boundary is knonw
        # so this is strongly imposed
        OpDx[1,2]=0
        OpDx[end,end-1]=0
        return OpDxx,OpDx
end

function formInitialGuess(par,nx)
 
    K1=ones(nx+1)*0.2
    K2=ones(nx+1)/2.
    K3=ones(nx+1)*10.0
    K4=ones(nx+1)*0.005
    K5=2.
    Kvec=zeros(4*(nx+1)+1)
    Kvec[1:nx+1].=K1
    Kvec[nx+2:2*nx+2].=K2
    Kvec[2*nx+3:3*nx+3].=K3
    Kvec[3*nx+4:4*nx+4].=K4

    Kvec[end]=1/K5

    return Kvec
end


end
