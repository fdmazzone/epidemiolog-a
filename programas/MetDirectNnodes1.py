#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def SIRMnn(N,gamma,R0,sigmaT,T,I0,ptos_grilla, metodo="global"):
    """
    Parameters
    ----------
    N : 1 dimensional np.array float64 n elementos
        Cantidad habitantes en cada nodo
    I0 : 1 dimensional np.array float64 n elementos
        Cantidad inicial de infectados.
    gamma : float64
        1/gamma = periodo infecciosidad.
    R0 : 2 dimensional np.array float64 nxn
         Matriz de R0.
    sigmaT : python función
             provisión total de vacunas.
    T : float64
        Tiempo final.
    n : int
        cantidad de puntos que dividiremos [0,T].

    Returns
    -------
    Graficos

    DESCRIPCIÓN
    -----------
    
    Resuelve el problema de control de minimizar

                  J(σ)=γ∫(I_1+I_2+...+I_n)dt (1)
                  
    σ=(σ₁,σ₂....,σₙ), Sᵢ Iᵢ , σ₁,σ₂,...,σₙ resuelven:  

    S'ᵢ(t)=−Sᵢ(t) ∑ⱼ₌₁ⁿ βᵢⱼ Iⱼ(t)− σᵢ Sᵢ(t)
    I'ᵢ(t)=Sᵢ(t) ∑ⱼ₌₁ⁿ βᵢⱼ Iⱼ(t)−γ Iᵢ(t)

    y estan sujetos a la condicion:
        σ₁≥0,  σ₂≥0,..., σₙ≥0. σ₁+σ₂+...+σₙ=1   
    Ver artículo "Optimal control on a vaccine metapopulation SIR model"

    Se discretiza el tiempo en valores almacenados en la variable t_int. 
    Se toma el vector U=(σ₁(t) | t ∈ t_int) como la variable de un problema de 
    minimización     Se define la función  J(U)=J(σ), donde σ es un 
    interpoolante lineal por pedazos     para U.  

    Finalmente se resuelve el problema de  minimizar J(U) por el metodo de 
    evolución diferencial. No se usan restricciones solo cotas, se parametriza 
    el conjunto de controles admisibles.


    @author: Fernando Mazzone
    """
   
    
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint 
    import numpy as np
    from scipy import optimize
    import time
    from scipy import interpolate
    import itertools

   
    start_time = time.time()
    
    
    
    global J, SIR2p, SimplexMap

    
    n=len(N)
    sigmaT = np.vectorize(sigmaT)
    x0=np.concatenate((N,I0))
    x0=np.concatenate((x0,[np.sum(x0[n:])]))
    beta=R0/np.tile(N,[n,1])   # beta = R0/N
    t_int=np.linspace(0,T,ptos_grilla)    
    nro_t=len(t_int)
    
 
    
    

    ########### PARAMETRIZACION CONJUNTO CONTROLES ADMISIBLES #########
    
    def SimplexMap(x):
        A=np.cumprod(x,axis=1)
        C=np.concatenate(( np.ones_like(x[:,[0]]), A[:,:-1]),axis=1)
        B=C-A
        return np.concatenate(( B,A[:,[-1]]   ),axis=1  )
    
    
    ########### ECUACIONES ESTADO ##############################
    def SIR2p(x,t,u):
        S,I=x[:n],x[n:2*n]
        Incidencia=S*beta.dot(I)
        U=u(t)
        dS=-S*U-Incidencia
        dI=-gamma*I+Incidencia
        dJ=gamma*np.sum(I)
        return np.concatenate((dS,dI,[dJ]))
    
    ################### FUNCIÒN OBJETIVO #######################   
 
    def J(U,t):
        W=U.reshape([nro_t,n-1])
        V=np.diag(sigmaT(t_int)).dot(SimplexMap(W))
        u_func=interpolate.interp1d(t_int, V.T,fill_value="extrapolate")
        sol = odeint(SIR2p,x0 ,t,args=(u_func,))
        return sol[-1,-1]
    
    
    
    ############# COTAS #######################
    bounds=[(0,1) for i in range(nro_t*(n-1))] 
    
 
    ############ OPTIMIZADOR ########################################
    
    t=np.linspace(0,T,2)
    if metodo=="global":
        opt=optimize.differential_evolution(J,bounds,args=(t,),workers=8)
        mje='Evolución Diferencial (Global)'
        U=opt["x"]
    elif metodo=="local":
        sigma0=np.random.rand(nro_t*(n-1))
        opt=optimize.minimize(J, sigma0, args=(t,), bounds=bounds)
        U=opt.x
        mje='Minimización Local'
    #U=np.random.rand(nro_t*(n-1)) #BORRAR  ES PARA VER SI NO HAY CONTROL
    
    
    
    W=U.reshape([nro_t,n-1])
    V=np.diag(sigmaT(t_int)).dot(SimplexMap(W))
    u_func=interpolate.interp1d(t_int, V.T,fill_value="extrapolate")
      
    
    
    
    
    
    
    ################ SOLUCIÒN ESTADO ÒPTIMO ##################
    
    t=np.linspace(0,T,200)
    sol = odeint(SIR2p,x0 ,t,args=(u_func,))
    S=sol[:,:n]
    I=sol[:,n:2*n]
    J_val=sol[:,2*n]
    x_func=interpolate.interp1d(t, sol.T,fill_value="extrapolate")    
    
    
    ################### PROBLEMA ADJUNTO  ###################33
    
    

    
    
    def ADJ(lam,t):
        betaI=beta.dot(x_func(T-t)[n:2*n])
        U=u_func(T-t)
        dlam1=betaI*(lam[:n]-lam[n:])+U*lam[:n]
        dlam2=(beta.T).dot(x_func(T-t)[:n]*(lam[:n]-lam[n:]))+gamma*lam[n:]+gamma*np.ones(n)
        return np.concatenate((-dlam1,-dlam2)) 
    
    sol_adj = odeint(ADJ, np.zeros(2*n) ,t)
    lam=sol_adj[::-1,:]
    
    lam_func=interpolate.interp1d(t, lam.T,fill_value="extrapolate")


    ########### HAMILTONIANO ###########################################
    def H(t,x,lam,U):
        u=lambda t: U
        f=SIR2p(x,t,u)
        return f[:-1].dot(lam)-gamma*sum(x[n:2*n])

    
    H_dim=lambda t: H(t,x_func(t),lam_func(t),u_func(t))
    H_dim = np.vectorize(H_dim)
    
    
    
    
    ########### GRAFICOS ###########################################
    fig,  ((ax1, ax2, ax3),( ax4, ax5, ax6),( ax7, ax8, ax9))=plt.subplots(3,3)
    ax1.plot(t,S) #susceptibles
    leyenda= ['$S_%s$'%i for i in range(1,n+1)]
    ax1.legend(leyenda)
    ax2.plot(t,I)#infeccionsos
    leyenda= ['$I_%s$'%i for i in range(1,n+1)]
    ax2.legend(leyenda)
    ax3.plot(t,np.sum(I,1)) #suma infecciosos
    ax3.legend((r'$\sum_{j=1}^nI_j$',))
    #control total y u1  u2
    
    marker = itertools.cycle(('^', 's', 'o', '>','<', '*')) 
    ax4.plot(t,u_func(t).T)#
    ax4.plot(t,sigmaT(t))#
    for l, ms in zip(ax4.lines, marker):
        l.set_marker(ms)
    plt.show()
    leyenda= ['$\sigma_%s(t)$'%i for i in range(1,n+1)]
    ax4.legend(leyenda)
    ax5.plot(t,lam[:,:n]*S)
    leyenda= ['$\lambda_%sS_%s$'%(i,i) for i in range(1,n+1)]
    ax5.legend(leyenda)
    ax6.plot(t,H_dim(t))
    ax6.legend((r'$H(t,x(t),\lambda(t))$',))
    ax7.plot(t,J_val)
    ax7.legend((r'$\gamma\int_0^T\sum_{j=1}^nI_jdt$',))
    ax8.plot(t,lam)
    leyenda= ['$\lambda_%s$'%i for i in range(1,2*n+1)]
    ax8.legend(leyenda)
    
    ax9.plot(t,sigmaT(t))#infeccionsos
    ax9.legend((r'$\sigma_T$',))
    
    
    Deltat=(time.time() - start_time)/3600
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    fig.text(.2,.97,r'Experimento método directo, minimización método '+metodo+'.  Nro ptos grilla:'+str(nro_t),fontsize=14)
    fig.text(.2,.89,r'$R_0=$   '+str(R0),fontsize=12)
    
    fig.text(.4,.92,'N='+str(N)+'  Tiempo computo: '+str('%5.2f'%Deltat)+'h',fontsize=12)
    
    fig.text(.4,.9,'I(0)='+str(x0[n:2*n]),fontsize=12)
    fig.text(.6,.9,'#Total infectados='+str(int(J_val[-1])),fontsize=12)
    
    print("--- %s seconds ---" % (Deltat*3600))
    
    
    
    
    
    