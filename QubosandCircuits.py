import matplotlib
import matplotlib.pyplot as plt
from qiskit import*
from qiskit import IBMQ
import numpy as np
import scipy 
import math
import networkx as nx
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # Makes the images look nice")
from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.tools.visualization import plot_histogram
from docplex.mp.model import Model
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import CplexOptimizer
from scipy.linalg import expm  
from qiskit.optimization.converters import (
    InequalityToEquality,     # converts inequality constraints to equality constraints by adding slack variables
    LinearEqualityToPenalty,  # converts linear equality constraints to quadratic penalty terms 
    IntegerToBinary,          # converts integer variables to binary variables
    QuadraticProgramToQubo    # combines the previous three converters
)
from  qiskit.aqua.operators.list_ops import *
from   qiskit.aqua.operators.primitive_ops import *
from   qiskit.quantum_info import *
import random
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import QAOA
from qiskit.extensions import UnitaryGate
from  qiskit.circuit.library.generalized_gates import MCMT
from qiskit.circuit.library.standard_gates import RXGate


#plot the graph 
def plot_result(G, x,n):
    colors = ['hotpink' if x[i] == 0 else 'b' for i in range(n)]
    pos, default_axes = nx.circular_layout(G,scale = 3), plt.axes(frameon=False)
    nx.draw_networkx(G, node_color=colors, node_size=700,  pos=pos,arrowsize=5)
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
    
    
#The costfunction of QUBO1 
def cost1(x,edges,cedges):
        n = len(x)
        q = (len(x) - 1)*(len(x))/2
        cost = 0
        for index in edges:
            e1 = index[ 0]
            e2 = index[ 1]
            cost = cost + x[n - 1 - e1]* x[n - 1 - e2]   
        for index in cedges:
            e1 = index[0]
            e2 = index[1]
            cost = cost - q*x[n - 1 - e1]* x[n - 1 - e2]          
        return cost

    
#NP-matrix of qubo1
#returns the np-matrix of the hamiltonian
def makeNPMatrix1(edges,cedges,n):
    #QUBO 
    mdl = Model('MaxClique')
    x = mdl.binary_var_list('x{}'.format(i) for i in range(n))
    q = (n-1)*n/2
    objective = mdl.sum([ (x[i]*x[j]) for (i, j) in edges]+[- q* (x[i]*x[j]) for (i, j) in cedges]) 
    mdl.maximize(objective)
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    #to matrix 
    H, offset = qp.to_ising()
    H_matrix = np.real(H.to_matrix())
    return H   

#The costfunction of QUBO3 and qubo3
def cost2(x,edges,cedges):
        n = len(x)
        cost = 0
        for i in range(n):
            cost = cost + x[n - 1 - i]  
        for index in cedges:
            e1 = index[0]
            e2 = index[1]
            cost = cost - 2*x[n - 1 - e1]* x[n - 1 - e2]          
        return cost


#The costfunction of Mixer
def cost4(x,edges,cedges):
        n = len(x)
        cost = 0
        for i in range(n):
            cost = cost + x[i]
        return cost


#NP-matrix of qubo2
#returns the np-matrix of the hamiltonian
def makeNPMatrix2(edges,cedges,n):
    #QUBO 
    mdl = Model('MaxClique')
    x = mdl.binary_var_list('x{}'.format(i) for i in range(n))
    q = (n-1)*n/2
    objective = mdl.sum([ (x[i]) for i  in range(n)]+[ -2* (x[i]*x[j]) for (i, j) in cedges]) 
    mdl.maximize(objective)
    mdl.prettyprint()
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    #to matrix 
    H, offset = qp.to_ising()
    H_matrix = np.real(H.to_matrix())
    return H 

def makeNPMatrix4(edges,cedges,n):
    #QUBO 
    mdl = Model('MaxClique')
    x = mdl.binary_var_list('x{}'.format(i) for i in range(n))
    objective = mdl.sum([ (x[i]) for i  in range(n)]) 
    mdl.maximize(objective)
    mdl.prettyprint()
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    #to matrix 
    H, offset = qp.to_ising()
    H_matrix = np.real(H.to_matrix())
    return H 

#qubo2
def makeCircuit2(n,edges,cedges,gammasbetas,p):
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    for i in range(n):
        QAOA.h(i)
    for j in range(p):
        for i in range(n):
            QAOA.rz(0.5*gammasbetas[j], i ) 
        
        for edge in cedges:
            k = edge[0]
            l = edge[1]
            QAOA.rz(-0.5*gammasbetas[j], k ) 
            QAOA.rz(-0.5*gammasbetas[j], l ) 
            QAOA.cx(k, l)
            QAOA.rz(0.5*gammasbetas[j],l) 
            QAOA.cx(k, l)

        QAOA.rx(gammasbetas[p+j],range(n))

    return QAOA

#qubo2
def makeCircuit1(n,edges,cedges,gammasbetas,p):
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    for i in range(n):
        QAOA.h(i)
        
    for j in range(p):
        for i in range(n):
            QAOA.rz(0.5*gammasbetas[j], i ) 
        
        for edge in cedges:
            k = edge[0]
            l = edge[1]
            QAOA.rz(-0.5*gammasbetas[j], k ) 
            QAOA.rz(-0.5*gammasbetas[j], l ) 
            QAOA.cx(k, l)
            QAOA.rz(0.5*gammasbetas[j],l) 
            QAOA.cx(k, l)

        QAOA.rx(gammasbetas[p+j],range(n))
    return QAOA

#qubo3
def makeCircuit3( n,edges,cedges,gammasbetas,p):
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    
    for i in range(n):
        QAOA.h(i)
    QAOA.barrier()
    for j in range(p):
        for i in range(n):
            QAOA.u1(-gammasbetas[j], i )
        for edge in cedges:
            k = edge[0]
            l = edge[1]
            #warum scheint das vorzeichen hier egal zu sein? 
            QAOA.cu1(2*gammasbetas[j],l,k) 
    #warum beta und nicht 2*beta? 
        QAOA.barrier()
        QAOA.rx(-2*gammasbetas[p+j],range(n))
        QAOA.barrier()

    return QAOA

#mixer
def makeCircuit4(graph,n,edges,cedges,gammasbetas,p):
    cgraph = nx.complement(graph)
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    for j in range(p):
        for i in range(n):
            QAOA.u1(-gammasbetas[j], i )  
        for i in range(n):
            nbrs = 0
            nbs = []
            for nbr in cgraph[i]:
                QAOA.x(nbr)
                nbrs += 1
                nbs.append(nbr)

            nbs.append(i)
            if(nbrs != 0):
                gate = MCMT(RXGate(-2*gammasbetas[p+j]),nbrs ,1)
                QAOA.append(gate,nbs )
            else:
                QAOA.rx(-2*gammasbetas[p+j],i)
            for nbr in cgraph[i]:
                QAOA.x(n - 1 - nbr)
       
    return QAOA


#to calculate dicke states 
def ccr(alpha,theta,c1,c2,u,qc,n): # control-control-rotation gate gate   
    pi = np.pi
    # alpha is x,y,z. u is qubit that u acts on. c_1,c_2 are the control locations
    if alpha == 'x':
        qc.cu3(theta/2,-pi/2,pi/2,c2,u)
        qc.cx(c1,c2)
        qc.cu3(-theta/2,-pi/2,pi/2,c2,u)
        qc.cx(c1,c2)
        qc.cu3(theta/2,-pi/2,pi/2,c1,u)
    elif alpha == 'y':
        qc.cu3(theta/2,0,0,c2,u)
        qc.cx(c1,c2)
        qc.cu3(-theta/2,0,0,c2,u)
        qc.cx(c1,c2)
        qc.cu3(theta/2,0,0,c1,u)
    else:
        qc.cu1(theta/2,c2,u)
        qc.cx(c1,c2)
        qc.cu1(-theta/2,c2,u)
        qc.cx(c1,c2)
        qc.cu1(theta/2,c1,u)
        
#to calculate dicke states     
def scs(x,y,qc,n): # s is starting qubit, qc is quantum circuit
    for i in range(1,y+1):
        if i == 1:
            qc.cx(x-1-i,x-1)
            qc.cu3(2*math.acos(math.sqrt(i/x)),0,0,x-1,x-1-i)
            qc.cx(x-1-i,x-1)
        else:
            qc.cx(x-1-i,x-1)
            ccr('x',2*math.acos(math.sqrt(i/x)),x-1,x-i,x-1-i,qc,n)
            qc.cx(x-1-i,x-1)
            
#mixerdicke
def makeCircuit5(graph,n,edges,cedges,gammasbetas,p):
    cgraph = nx.complement(graph)
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    k = 1
    #dickestate
    for i in range(n-1,n-k-1,-1):
        QAOA.x(i)
    for l in range(n,k,-1):
        scs(l,k,QAOA,n)
    for l in range(k,1,-1):
        scs(l,l-1,QAOA,n) 
    #mixer and phaseseparation 
    for j in range(p):
        for i in range(n):
            QAOA.u1(-gammasbetas[j], i )       
        for i in range(n):
            nbrs = 0
            nbs = []
            for nbr in cgraph[i]:
                QAOA.x(nbr)
                nbrs += 1
                nbs.append(nbr)

            nbs.append(i)
            if(nbrs != 0):
                gate = MCMT(RXGate(gammasbetas[p+j]),nbrs ,1)
                QAOA.append(gate,nbs )
            #else:
            #warum habe ich das ausgeklammert?  Testen ob es besser ist? 
                #QAOA.rx(gammasbetas[p+j],i)
            for nbr in cgraph[i]:
                QAOA.x(nbr)

    return QAOA


#mixersim  trotterized with t = 2 
def makeCircuit6(graph,n,edges,cedges,gammasbetas,p):
    cgraph = nx.complement(graph)
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    r = 2
        
    for j in range(p):
        
        for i in range(n):
            QAOA.u1(-gammasbetas[j], i )
        
        for l in range(r):       
            for i in range(n):
                nbrs = 0
                nbs = []
                for nbr in cgraph[i]:
                    QAOA.x(nbr)
                    nbrs += 1
                    nbs.append(nbr)

                nbs.append(i)
                if(nbrs != 0):
                    #gate = MCMT(RXGate(gammasbetas[p+j]/r),nbrs ,1)
                    gate = MCMT(RXGate(2*gammasbetas[l+j]/r),nbrs ,1)
                    QAOA.append(gate,nbs )
                else:
                    #QAOA.rx(gammasbetas[p+j]/r,i)
                    QAOA.rx(2*gammasbetas[l+j]/r,i)
                for nbr in cgraph[i]:
                    QAOA.x(nbr)
       
    return QAOA

#mixerrand
#orderingsis a list of p lists of a random permutation of 0 to n - 1 
def makeCircuit7(graph,n,edges,cedges,gammasbetas,p,orderings):
    cgraph = nx.complement(graph)
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    for j in range(p):
        for i in range(n):
            QAOA.u1(-gammasbetas[j], i ) 
        for i in orderings[j]:
            nbrs = 0
            nbs = []
            for nbr in cgraph[i]:
                QAOA.x(nbr)
                nbrs += 1
                nbs.append(nbr)
            nbs.append(i)
            if(nbrs != 0):
                gate = MCMT(RXGate(-2*gammasbetas[p+j]),nbrs ,1)
                QAOA.append(gate,nbs )
            else:
                QAOA.rx(-2*gammasbetas[p+j], i)
            for nbr in cgraph[i]:
                QAOA.x(nbr)
       
    return QAOA

#mixerrandminus
#orderingsis a list of p lists of a random permutation of 0 to n - 1 
#gammasbetas is p =(p-1) gamma + p beta
def makeCircuit9(graph,n,edges,cedges,gammasbetas,p,orderings):
    cgraph = nx.complement(graph)
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    for j in range(p):
        if(j!=0): 

            for i in range(n):

                QAOA.u1(-gammasbetas[j - 1], i ) 
                QAOA.barrier()

        for i in orderings[j]:

            nbrs = 0
            nbs = []
            for nbr in cgraph[i]:
                QAOA.x( nbr)
                nbrs += 1
                nbs.append( nbr)

            nbs.append(i)
            if(nbrs != 0):
                gate = MCMT(RXGate(2*gammasbetas[p - 1 +j]),nbrs ,1)
                QAOA.append(gate,nbs )
            else:
                QAOA.rx(2*gammasbetas[p - 1 +j],i)
            for nbr in cgraph[i]:
                QAOA.x(nbr) 
            QAOA.barrier()
    return QAOA

#mixerdickerand
#orderingsis a list of p lists of a random permutation of 0 to n - 1 
def makeCircuit8(graph,n,edges,cedges,gammasbetas,p,orderings):
    cgraph = nx.complement(graph)
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA = QuantumCircuit(sol,ans)
    
    k = 1
    #dickestate
    for i in range(n-1,n-k-1,-1):
        QAOA.x(i)
    for l in range(n,k,-1):
        scs(l,k,QAOA,n)
    for l in range(k,1,-1):
        scs(l,l-1,QAOA,n) 
        
    for j in range(p):
        for i in range(n):
            QAOA.u1(-gammasbetas[j], i ) 

        for i in orderings[j]:

            nbrs = 0
            nbs = []
            for nbr in cgraph[i]:
                QAOA.x(nbr)
                nbrs += 1
                nbs.append(nbr)

            nbs.append(i)
            if(nbrs != 0):
                gate = MCMT(RXGate(gammasbetas[p+j]),nbrs ,1)
                QAOA.append(gate,nbs )
            else:
                QAOA.rx(gammasbetas[p+j],i)
            for nbr in cgraph[i]:
                QAOA.x(nbr)
       
    return QAOA
