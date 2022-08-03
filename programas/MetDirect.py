#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def SIRM2n(N,gamma,R0,sigmaT,T,I0,ptos_grilla,metodo="global"):
    """
    Parameters
    ----------
    N : 1 dimensional np.array float64 2 elementos
        Cantidad habitantes en cada nodo
    I0 : 1 dimensional np.array float64 2 elementos
        Cantidad inicial de infectados.
    gamma : float64
        1/gamma = periodo infecciosidad.
    R0 : 2 dimensional np.array float64 2x2
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

    Resuelve el problema de control de minimizar

                  J(σ)=γ∫(I_1+I_2)dt (1)
                  
    σ=(σ₁,σ₂), Sᵢ Iᵢ , σ₁,σ₂ resuelven:  

    S'ᵢ(t)=−Sᵢ(t) ∑ⱼ₌₁ⁿ βᵢⱼ Iⱼ(t)− σᵢ Sᵢ(t)
    I'ᵢ(t)=Sᵢ(t) ∑ⱼ₌₁ⁿ βᵢⱼ Iⱼ(t)−γ Iᵢ(t)

    y estan sujetos a la condicion:
        σ₁≥0,  σ₂≥0. σ₁+σ₂=1   
    Ver artículo "Optimal control on a vaccine metapopulation SIR model"

    Se discretiza el tiempo en valores almacenados en la variable t_int. 
    Se toma el vector U=(σ₁(t) | t ∈ t_int) como la variable de un problema de 
    minimización     Se define la función  J(U)=J(σ), donde σ es un 
    interpoolante lineal por pedazos     para U.  

    Finalmente se resuelve el problema de  minimizar J(U) por el metodo de 
    evolución diferencial.


    @author: Fernando Mazzone
    """
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint 
    from scipy import optimize
    import time
    from scipy.interpolate import UnivariateSpline as spline
    import numpy as np
    
    
    start_time = time.time()
    global J, SIR2p

    
    n=len(N)
    sigmaT = np.vectorize(sigmaT)
    x0=np.concatenate((N,I0))
    x0=np.concatenate((x0,[np.sum(x0[n:])]))
    beta=R0/np.tile(N,[n,1])   # beta = R0/N
    t_int=np.linspace(0,T,ptos_grilla)

 
    
    
    ########### ECUACIONES ESTADO ##############################
    def SIR2p(x,t,u):
        S,I=x[:2],x[2:4]
        Incidencia=S*beta.dot(I)
        U=np.array([u(t),sigmaT(t)-u(t)])
        dS=-S*U-Incidencia
        dI=-gamma*I+Incidencia
        dJ=gamma*np.sum(I)
        return np.concatenate((dS,dI,[dJ]))
    
    
    ################### FUNCIÒN OBJETIVO #######################
    def J(U,t):
        u_func=spline(t_int, U,s=0, k=1)
        sol = odeint(SIR2p,x0 ,t,args=(u_func,))[-1,4]
        return sol
    
    
    
    ############# RESOLVEMOS MINIMIZACIÒN #######################
    bounds=[(0,sigmaT(i)) for i in t_int]
    t=np.linspace(0,T,2)

    
        
    if metodo=="global":
        opt=optimize.differential_evolution(J,bounds,args=(t,),workers=8)
        mje='Evolución Diferencial (Global)'
        U=opt["x"]
    elif metodo=="local":
        nro_t=len(t_int)
        sigma0=np.zeros(nro_t*(n-1))
        opt=optimize.minimize(J, sigma0, args=(t,),bounds=bounds)
        mje='Minimización Local'
        U=opt.x
    # U=np.random.rand(len(t_int))*sigmaT(t_int) #BORRAR  ES PARA VER SI NO HAY CONTROL
    # mje="Control al azar"              #BORRAR  ES PARA VER SI NO HAY CONTROL
    
    
    u_func=spline(t_int, U,s=0, k=1)
    
    
    ############  RESOLVEMOS ECUACIÓON DE ESTADO ###############
    
    t=np.linspace(0,T,200)
    sol = odeint(SIR2p,x0 ,t,args=(u_func,))
    S=sol[:,:2]
    I=sol[:,2:4]
    J_val=sol[:,4]
    
    S1=spline(t,S[:,0],s=0)
    S2=spline(t,S[:,1],s=0)
    I1=spline(t,I[:,0],s=0)
    I2=spline(t,I[:,1],s=0)
    
    S_func=lambda t: np.array([S1(t),S2(t)])
    I_func=lambda t: np.array([I1(t),I2(t)])
    
    x_func=lambda t: np.array([S1(t),S2(t),I1(t),I2(t)])

    ################### PROBLEMA ADJUNTO  ###################    
    def ADJ(lam,t):
        betaI=beta.dot(I_func(T-t))
        U=np.array([u_func(T-t),sigmaT(T-t)-u_func(T-t)])  
        dlam1=betaI*(lam[:2]-lam[2:])+U*lam[:2]
        dlam2=(beta.T).dot(S_func(T-t)*(lam[:2]-lam[2:]))+gamma*lam[2:]+gamma*np.ones(2)
        return np.concatenate((-dlam1,-dlam2)) 
    
    sol = odeint(ADJ, np.zeros(4) ,t)
    lam=sol[::-1,:]
     
    
    lam1=spline(t,lam[:,0],s=0)
    lam2=spline(t,lam[:,1],s=0)
    lam3=spline(t,lam[:,2],s=0)
    lam4=spline(t,lam[:,3],s=0)
    
    lam_func=lambda t: np.array([lam1(t),lam2(t),lam3(t),lam4(t)])
    
    
    ###########  HAMILTONIANO  ###########################
    def H(t,x,lam,U):
        u=lambda t: U
        f=SIR2p(x,t,u)
        return f[:4].dot(lam)-x[2]-x[3]
    
      
    H_dim=lambda t: H(t,x_func(t),lam_func(t),u_func(t))
    H_dim = np.vectorize(H_dim)    
    
    
    ################GRAFICAMOS SOLUCION ##################
    
    

    
    
    
    
    
    fig,  ((ax1, ax2, ax3),( ax4, ax5, ax6),( ax7, ax8, ax9))=plt.subplots(3,3)
    ax1.plot(t,S) #susceptibles
    ax1.legend((r'$S_1$',r'$S_2$'))
    ax2.plot(t,I)#infeccionsos
    ax2.legend((r'$I_1$',r'$I_2$'))
    ax3.plot(t,np.sum(I,1)) #suma infecciosos
    ax3.legend((r'$I_1+I_2$',))
    #control total y u1  u2
    ax4.plot(t,np.array([u_func(t),sigmaT(t)-u_func(t),sigmaT(t)]).T)#
    ax4.legend((r'$\sigma_1(t)$',r'$\sigma_2(t)$',r'$\sigma_T(t)$'))
    
    

    
    ax5.plot(t,lam2(t)*S2(t)-lam1(t)*S1(t))
    ax5.legend((r'$\lambda_2(t)S_2(t)-\lambda_1(t)S_1(t)$',))
    ax6.plot(t,H_dim(t))
    ax6.legend((r'$H(t,x(t),\lambda(t))$',))
    
    ax7.plot(t,J_val)
    ax7.legend((r'$\int_0^T(I_1+I_2)dt$',))
    
    ax8.plot(t,lam)#infeccionsos
    ax8.legend((r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$',r'$\lambda_4$'))
    
    ax9.plot(t,sigmaT(t))#infeccionsos
    ax9.legend((r'$\sigma_T$',))
    
    Deltat=(time.time() - start_time)/3600
    nro_t=len(t_int)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    fig.text(.2,.97,r'Experimento método directo, '+mje+'.  Nro ptos grilla:'+str(nro_t),fontsize=14)
    fig.text(.2,.89,r'$R_0=$   '+str(R0),fontsize=12)
    #fig.text(.4,.95,r'$\gamma=1,\quad R_0=$'+str(R0),   fontsize=12)
    
    
    fig.text(.4,.92,'N='+str(N)+'  Tiempo computo: '+str('%5.2f'%Deltat)+'h',fontsize=12)
    
    fig.text(.4,.9,'I(0)='+str(x0[n:2*n]),fontsize=12)
    
    fig.text(.6,.9,'#Total infectados='+str(int(J_val[-1])),fontsize=12)
    
    print("--- %s seconds ---" % (time.time() - start_time))