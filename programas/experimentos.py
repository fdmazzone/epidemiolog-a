#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:09:16 2022

@author: fernando
"""
import numpy as np


############ EXPERIMENTO CON TRES NODOS ###############
N= np.array([100000,100000, 100000, 100000])  #  Población en los nodos
gamma=1
R0=np.array([[1.2,0,0,1E-3],[1E-3,2.5,0,0],[0,1E-3,2,0],[0,0,1E-3,1.5]])
sigmaT=lambda t: .05  
T=60.0 
I0=np.array([0,0,0,1])
ptos_grilla=10
######################################################

############ EXPERIMENTO CON TRES NODOS ###############
# N= np.array([100000,100000, 100000])  #  Población en los nodos
# gamma=1
# R0=np.array([[2,1E-3,1E-3],[1E-3,2,1E-3],[1E-3,1E-3,2]])
# sigmaT=lambda t: .05  
# T=40.0 
# I0=np.array([0,0,1])
# ptos_grilla=10
######################################################

############ EXPERIMENTO CON DOS NODOS ###############
# N= np.array([80000,120000])  #  Población en los nodos
# gamma=1
# R0=np.array([[1.8,0.001],[.001,1.8]])
# sigmaT=lambda t: .05  
# T=40.0 
# I0=np.array([0,10])
# ptos_grilla=10
######################################################
     
# from MetDirect import SIRM2n
# SIRM2n(N,gamma,R0,sigmaT,T,I0,ptos_grilla, metodo="global")


from MetDirectNnodes1 import SIRMnn
SIRMnn(N,gamma,R0,sigmaT,T,I0,ptos_grilla,metodo="local")

# from MetDirectNnodes import SIRMnnLC
# SIRMnnLC(N,gamma,R0,sigmaT,T,I0,ptos_grilla)






