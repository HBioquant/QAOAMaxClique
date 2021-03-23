import matplotlib.pyplot as plt
from qiskit import*
import numpy as np
import scipy 
import math
import copy
import networkx as nx
from   matplotlib import cm
#from   matplotlib.ticker import LinearLocator, FormatStrFormatter
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # Makes the images look nice")
from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.tools.visualization import plot_histogram
from docplex.mp.model import Model
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import CplexOptimizer
from scipy.linalg import expm
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor   
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
from QubosandCircuits  import *





#global variables for database use 
#counts the steps of an optimization
steps = 0
#needed because a list cant be constatns 
orderingsglobal  = [] 
 

def QAOAforMaximumClique(graph = False,
                                                 qubo = False,#(qubo1,qubo2,qubo3,mixer,mixerdicke,mixersim,mixerrand,mixerrandnoise, mixerrandminus
                                                 backend = False, #
                                                 #shots, When shots is false n*400 is used (look for a beter version) 
                                                 shots = False,
                                                 #if p = False then p is 1 
                                                 p = False ,
                                                 #enable to use the qiskit library
                                                 library = False, #True to enable
                                                 #the method for finding the starting 
                                                 #points of the minimazation                                
                                                 angleselection = False, #(monte-carlo, basin-hopping)
                                                 #need for monte-carlo
                                                 guesses = False, #if False guesses = n                                               
                                                 #only when library is false 
                                                 optimizer = False, #(gradientdescent,python,stochastic-gradient-descent            
                                                 #enable to plot the graph
                                                 plotgraph = False,
                                                 #enable to plot the transpiled version of the circuit
                                                 transpiled = False, #todo
                                                 #enable to plot the circuit
                                                 plotcircuit = False,
                                                 #enable to plot the results histogram
                                                 histogram = False,
                                                 #to make a histogramof  the output propability of the expacation value 
                                                 startangles = [],
                                                 # a list for all guesses of lists for for all p with permuations of 0 to n - 1
                                                 orderings = [],
                                                 #epsilon used to calculate the gradient numarically 
                                                 eps = False,
                                                 #alpha is the step size of adams algorithm 
                                                 alpha = False,
                                                 #the treshold of how near exact the gradient decsent will be 
                                                 threshold = False,
                                                 #beta2 from adams algorithm 
                                                 beta1 = False,
                                                 #beta2 from adams algorithm        
                                                 beta2 = False,
                                                 #hoe many steps in gradient decsent 
                                                 optisteps = False,
                                                 #set true to simulate noise
                                                 noise=False,
                                                 #to draw a historgramm of each single guess 
                                                 executeallguesses = False,
                                                 #plottheoptimalangles in a graph 
                                                 plotoptangles = False,
                                                 #the provider is neede for the noise 
                                                 provider = False
                                                 ): 
    
    #these are only the parameters of gradientdecsten with adams algorihm 
    if(optimizer == 'gradientdescent'):
        if(eps == False):
            eps = 0.001
        if(alpha == False):
            alpha = 0.1
        if(threshold == False):
            threshold = 0.3
        if(beta1 == False):
            beta1 = 0.9
        if(beta2 == False):
            beta2 = 0.99
        if(optisteps == False):
            optisteps = 20
      
    #get data fromm graphs
    edges = graph.edges()
    cgraph = nx.complement(graph)
    cedges = cgraph.edges()
    nodes = graph.nodes()
    n = len(nodes)
       
    #set initial default paramters if they are not already set 
    if(guesses == 0):
        guesses = 0 
    if(guesses == False):
        guesses = 0
    if (p == False):
        p = 1  
    if(shots == False):
        shots = n*400       
    ranges = [0,2*math.pi,0,math.pi]
    if(angleselection == False):
        angleselection = "monte-carlo"
    
    #plot the graoh 
    if(plotgraph):
        plot_result(graph, [0]*n,n)
    #guess orderings if qubo = mixerrand of mixerrandminus 
    orderings = guessorderings(qubo,p,guesses,orderings,n)
    
    if(library):
        qc, optgammasbetas = libraryoptimize(qubo,edges,cedges,n,p,plotoptangles=plotoptangles)
        result = quantumsimulation(qc,n,shots,backend,provider=provider,noise=noise,plotcircuit=plotcircuit)
        if(histogram):
            display (plot_histogram(result.get_counts(),figsize = (8,6),bar_labels = False))    
        propability,maxcliques  = propabilityofsucess(qubo,result,edges,cedges,n,shots)
        return 
    
    #for database
    guessedgammaandbetas = []
    optimizedgammaandbetas = []
    optimiedexpectationvalues = []
    numberofsteps = [] 
    probabiliiesofsuccesses = [] 
    #each guess goes individually
    if(executeallguesses):
        guesses2 = guesses
        guesses3 = 1
        
    else:
        guesses2 = 1
        guesses3 = guesses
    for i in range(guesses2):
    #I have it only for monte carlo implemeted
        if(angleselection == "monte-carlo"):
            if(startangles==[] and orderings==[]):
                qc,numberofsimulationsin, randomgammasbetas, gammasbetas, exp, steps = montecarlodata(backend,optimizer,guesses3,graph,qubo,n,edges,cedges,p,ranges,shots,startangles,orderings,provider=provider,eps=eps,alpha=alpha,threshold=threshold,beta1=beta1,beta2=beta2,optisteps=optisteps,noise=noise,plotoptangles = plotoptangles ) 
            if(startangles==[] and orderings!=[]):
                qc,numberofsimulationsin, randomgammasbetas, gammasbetas, exp, steps = montecarlodata(backend,optimizer,guesses3,graph,qubo,n,edges,cedges,p,ranges,shots,startangles,orderings[i],provider=provider,eps=eps,alpha=alpha,threshold=threshold,beta1=beta1,beta2=beta2,optisteps=optisteps ,noise=noise,plotoptangles = plotoptangles) 
            if(startangles!=[] and orderings==[]):
                qc,numberofsimulationsin, randomgammasbetas, gammasbetas, exp, steps = montecarlodata(backend,optimizer,guesses3,graph,qubo,n,edges,cedges,p,ranges,shots,startangles[i],orderings,provider=provider,eps=eps,alpha=alpha,threshold=threshold,beta1=beta1,beta2=beta2,optisteps=optisteps ,noise=noise,plotoptangles = plotoptangles ) 
            if(startangles!=[] and orderings!=[]):
                qc,numberofsimulationsin, randomgammasbetas, gammasbetas, exp, steps = montecarlodata(backend,optimizer,guesses3,graph,qubo,n,edges,cedges,p,ranges,shots,startangles[i],orderings[i],provider=provider,eps=eps,alpha=alpha,threshold=threshold,beta1=beta1,beta2=beta2 ,optisteps=optisteps,noise=noise,plotoptangles = plotoptangles) 
                
        if(angleselection == "one"):
            if(qubo == "qubo3"):
                qc = makeCircuit3(n,edges,cedges,startangles[0],p)
            if(qubo == "mixerrandminus"):
                qc = makeCircuit9(graph,n,edges,cedges,startangles[0],p,orderings[0])
            randomgammasbetas = startangles
            gammasbetas = startangles
            exp = [0]
            steps = 1
            mean = [0]
            stdofeachstep = [0]
            rangeofeachstep = [0]
            propability = 0
                
             
        #nicht fertig 
        if(angleselection == "basin-hopping"):
            qc = basinhopping(optimizer,1,graph,qubo,n,edges,cedges,p,shots) 
           

        result = quantumsimulation(qc,n,shots,backend,plotcircuit=plotcircuit,cedges = cedges,depthweightcnots = False,provider=provider,noise=noise)
        cnots = 0
        depth = 0
        if(transpiled):   
            result,cnots,depth = quantumsimulation(qc,n,shots,backend,transpiled,cedges = cedges,depthweightcnots = True,provider=provider,noise=noise)  
        if(histogram):
            display (plot_histogram(result.get_counts(),figsize = (8,6),bar_labels = False))
        
        propability,maxcliques  = propabilityofsucess(qubo,result,edges,cedges,n,shots)
        #begin of  database
     
        guessedgammaandbetas.append(randomgammasbetas)      
        optimizedgammaandbetas.append(list(gammasbetas))
        optimiedexpectationvalues = optimiedexpectationvalues + exp
        numberofsteps.append(steps)
        probabiliiesofsuccesses.append(propability)
        #end of database
 
    
    output = []
    #edges of graph
    output.append(list(edges))
    
    # list[float]: [probabiliiesofsuccesses]
    output.append(probabiliiesofsuccesses)
    
    #list[float]: for each guess[optimiedexpectationvalues]
    output.append(optimiedexpectationvalues) 
    
    #list[int]: [for each guess numberofsteps]
    output.append(numberofsteps)  
       
    # depthaftertranspiled
    output.append(depth)
    # cnots after transpiled 
    output.append(cnots)     
    #n
    output.append(n)
    #[maxcliques]  
    output.append(maxcliques)      
    #number of iterations  p
    output.append(p)      
    # qubo
    output.append(qubo)   
    #optmizer
    output.append(optimizer)
    # angleselection
    output.append(angleselection)
    # number of guesses
    output.append(guesses)   
    #guessingrange[fromgamma,untilgamma,frombeta,unitlbeta]
    output.append(ranges)      
    # shots
    output.append(shots)   
    #list[list[float]]: [for each guess guessed[gamma1, ..., gammap,beta1,...betap]]
    output.append(guessedgammaandbetas)  
    # list[list[float]]:  [for each guess optimized [gamma1, ..., gammap,beta1,...betap]]
    output.append(optimizedgammaandbetas) 
    #orderings
    output.append(orderings)
    output.append(backend)
    #eps
    output.append(eps)
    #alpha
    output.append(alpha)
    #threshold
    output.append(threshold)
    #beta1
    output.append(beta1)
    #beta2
    output.append(beta2)
    #optisteps
    output.append(optisteps)
    #noise
    output.append(noise)
    return output

#guesstheorderings of the partitiond mixer 
def guessorderings(qubo,p,guesses,orderings,n):
    if(((qubo == "mixerrand")or (qubo == "mixerdickerand") or (qubo == "mixerrandminus")or (qubo == "mixerrandnoise")) and (orderings == [])):
        orderings = [] 
        for k in range(guesses):
            orderingforoneguess = [] 
            for j in range(p):
                #ordeing for one p  
                orderingforoneguess.append(list(np.random.permutation(n)))
            orderings.append(orderingforoneguess)               
        #Todo orderings make 
   # print(orderings)
    if((qubo != "mixerrand")and (qubo != "mixerdickerand") and (qubo != "mixerrandminus") and (qubo != "mixerrandnoise") and (orderings == [])):
        orderings = [] 
        for i in range(guesses):
            orderings.append(0)
    return orderings

#use the library from qiskit 
def libraryoptimize(qubo,edges,cedges,n,p,plotoptangles=False,modulo=False):       
    if(qubo == "qubo1"):
        H = makeNPMatrix1(edges,cedges,n)       
    if(qubo == "qubo2"):
        H = makeNPMatrix2(edges,cedges,n)       
    if(qubo == "qubo3"):
        H = makeNPMatrix2(edges,cedges,n)
        
    optimizer = COBYLA()
    qaoa_mes = QAOA(H, p=p, optimizer=optimizer, quantum_instance=Aer.get_backend('statevector_simulator'))
    results = qaoa_mes.run()
    qc1 = qaoa_mes.get_optimal_circuit()
    print(type(results.optimal_parameters))
    print(type(results.optimal_parameters.items()))
    i = 0
    for key, value in results.optimal_parameters.items():
        if(i == 0):
            gamma = value
            i += 1
        else:
            beta = value
    print("beta",beta)
    print("gamma",gamma)
    print('optimal params:      ', results.optimal_parameters)
    print('optimal value:       ', results.optimal_value)
    qcl = qaoa_mes.get_optimal_circuit()
    optgammasbetas = []
    isgamma = 0
    for key, value in results.optimal_parameters.items():
        if(modulo):
            if(isgamma < p):
                optgammasbetas.append(value%(2*math.pi))
                isgamma += 1
            else:
                optgammasbetas.append(value%(math.pi))
        else:
            optgammasbetas.append(value)
                  
    if(plotoptangles):
        plottheoptangles(optgammasbetas,p)
    if(qubo == "qubo1"):
        qc = makeCircuit1(n,edges,cedges,optgammasbetas,p)  
    if(qubo == "qubo2"):
        qc = makeCircuit2(n,edges,cedges,optgammasbetas,p)
    if(qubo == "qubo3"):
        qc = makeCircuit3(n,edges,cedges,optgammasbetas,p)
        
    ans = ClassicalRegister(n)
    sol = QuantumRegister(n)
    QAOA2 = QuantumCircuit(sol,ans)
    QAOA2.append(qc1,range(n))
    return qc, optgammasbetas

#only used for  1 geuss //the data function 
def montecarlodata(backend,optimizer,guesses,graph,qubo,n,edges,cedges,p,ranges,shots,startangles=False,orderings=False,provider=False,eps=False,alpha=False,threshold=False,beta1=False,beta2=False,optisteps=False,noise=False,plotoptangles = False ):
    numberofsimulationsin = 0
    optgammasbetas = []
    
    expopt = float("inf")
    #the orderings with the optimal permutation 
    orderingopt = orderings
    orderingguess = orderings
    #database 
    randomgammabetaslist = []
    optgammasbetaslist = [] 
    explist = []
    for i in range(guesses):     
        #when no angels are given take random ones 
        if(startangles==[]):
            gammasbetas = [] 
            
            for j in range(p): 
                if((j!=0) or (qubo != "mixerrandminus")):
                    gammasbetas.append(random.uniform(ranges[0], ranges[1]))
            for j in range(p):              
                gammasbetas.append(random.uniform(ranges[2], ranges[3]))
        #else use the given angles 
        else:
            gammasbetas = startangles        
    
        randomgammabetas = copy.copy(gammasbetas)
        randomgammabetaslist.append(randomgammabetas)
     
              
        if(optimizer == "python"):
            gammasbetas,exp,steps  = minimizewithpython(graph,backend,gammasbetas,qubo,n,edges,cedges,p,shots,orderingguess,provider,noise=noise)
        if(optimizer == "gradientdescent"):
            gammasbetas,exp,steps  = minimizewithgradientdescent(graph,backend,gammasbetas,qubo,n,edges,cedges,p,shots,orderingguess,provider,eps,alpha,threshold,beta1,beta2,optisteps,noise)
        #database
        optgammasbetaslist.append(list(gammasbetas))
        explist.append( - exp)
        #wir minimieren hier 
        if(exp < expopt ):
            expopt = exp
            optgammasbetas = gammasbetas
   
    if(qubo == "qubo1"):
        qc = makeCircuit1(n,edges,cedges,optgammasbetas,p)  
    if(qubo == "qubo2"):
        qc = makeCircuit2(n,edges,cedges,optgammasbetas,p)
    if(qubo == "qubo3"):
        qc = makeCircuit3(n,edges,cedges,optgammasbetas,p)
    if(qubo == "mixer"):
        qc = makeCircuit4(graph,n,edges,cedges,optgammasbetas,p)
    if(qubo == "mixerdicke"):
        qc = makeCircuit5(graph,n,edges,cedges,optgammasbetas,p)
    if(qubo == "mixersim"):
        qc = makeCircuit6(graph,n,edges,cedges,optgammasbetas,p)
    if((qubo == "mixerrand" ) or (qubo == "mixerrandnoise" )):
        qc = makeCircuit7(graph,n,edges,cedges,optgammasbetas,p,orderingopt)
    if(qubo == "mixerdickerand"):
        qc = makeCircuit8(graph,n,edges,cedges,optgammasbetas,p,orderingopt)
    if(qubo == "mixerrandminus"):
        qc = makeCircuit9(graph,n,edges,cedges,optgammasbetas,p,orderingopt)
    if(plotoptangles):
        plottheoptangles(optgammasbetas,p)
    return qc,numberofsimulationsin,randomgammabetaslist,optgammasbetaslist,explist, steps

#plot the anlges in a histogram 
def plottheoptangles(optgammasbetas,p):
    ps = []
    betas = [] 
    gammas = []
    for i in range(p):
        ps.append(str(i+1))
        betas.append(optgammasbetas[i+p])
        gammas.append(optgammasbetas[i])
    plt.ylim(0.0,6.2832)
    plt.plot(ps,gammas,color='b')
    plt.ylabel('gamma')
    plt.show()
    plt.ylim(0.0,3.1416)
    plt.plot(ps,betas,color='r')
    plt.ylabel('beta')
    plt.show()



#minimeze with python library scipy 
def minimizewithpython(graph,backend,gammasbetas,qubo,n,edges,cedges,p,shots,orderingguess= False,provider=False,noise=
False): 
    variational = gammasbetas
    constants = [qubo,n,edges,cedges,graph,p,shots,backend,provider,noise]     
    global orderingsglobal
    orderingsglobal = orderingguess
    #x_list = []
    def callback(x):
        print(x)
    global steps  
    steps = 0
    #use the scipy minimize functions 
    opt = scipy.optimize.minimize(expectationValueObj, x0=variational, args=constants, method = 'COBYLA')   
    exp = expectationValueObj(opt.x,constants) 
    return opt.x ,exp, steps



def minimizewithgradientdescent(graph,backend,gammasbetas,qubo,n,edges,cedges,p,shots,orderingguess,provider,eps,alpha,threshold,beta1,beta2,optisteps,noise):     
    constants = [qubo,n,edges,cedges,graph,p,shots,backend,provider,noise] 
    
    #database
    global orderingsglobal
    orderingsglobal = orderingguess       
    global steps  
    steps = 0
    
    
    #isualization? 
    gammabetacounts = []
    for i in range(len(gammasbetas)):
        gammabetacounts.append([gammasbetas[i]])
    
    #visualization 
    count = []

    exps = []  
    #startingpoint 
    count.append(0)
    exps.append( expectationValueObj2(gammasbetas,constants))  
  
    m = [0]*(2*p)
    v = [0]*(2*p)
  
    gradients = [] 
    for i in range(len(gammasbetas)):
        temp = gammasbetas[i]
        gammasbetas[i] += eps
        firtsterm = expectationValueObj2(gammasbetas,constants)      
        gammasbetas[i] -= 2*eps        
        secondterm = expectationValueObj2(gammasbetas,constants)
        partialgrad = (firtsterm - secondterm)/(2*eps)       
        gradients.append(partialgrad)      
        gammasbetas[i] += eps
    
    gammasbetasopt = copy.copy(gammasbetas)
    
    #calculating the norm of the gradient 
    normofgrad = getnormofgradient(gradients)
    # for counting 
    c = 0

    #going through all ps 
    flag = 0 
    t = 1
    #while(normofgrad > 2*threshold):
    
    while(c < optisteps):              
        gradients[flag] = getpartialgradient(flag,gammasbetas,eps,constants)
        
        
        #to get the angles before they are overwritten 
        gammasbetasopt = copy.copy(gammasbetas)
        
        #adams algorithm
        m[flag] = beta1*m[flag] + (1 - beta1)*partialgrad
        v[flag] = beta2*v[flag] + (1 - beta2)*partialgrad*partialgrad
        m2 = m[flag]/(1 - beta1**t)
        v2 = v[flag]/(1 - beta2**t)
        
        #update angles
        if(flag < p):
            temp = gammasbetas[flag]
            gammasbetas[flag] =  (gammasbetas[flag] + alpha* m2*(math.sqrt(v2)+eps)) #%(2*np.pi)
            if((gammasbetas[flag] > (2*np.pi)) or (gammasbetas[flag] < 0)):
                gammasbetas[flag] = temp          
            
            #gammasbetas[flag] =  (gammasbetas[flag] + alpha*(partialgrad))%(2*np.pi)
        else:
            temp = gammasbetas[flag]
            gammasbetas[flag] =  (gammasbetas[flag] + alpha* m2*(math.sqrt(v2)+eps)) #%(np.pi)
            if((gammasbetas[flag] > (np.pi)) or (gammasbetas[flag] < 0)):
                gammasbetas[flag] = temp
            #gammasbetas[flag] =  (gammasbetas[flag] + alpha*(partialgrad))%(np.pi)
        
        if(flag == (len(gradients) - 1)):
            t = t+1 
        flag = (flag + 1 )%(len(gammasbetas))
        
        #calculate new norm 
        normofgrad = getnormofgradient(gradients)
        
        #visualization
        for i in range(len(gammasbetas)):          
            gammabetacounts[i].append(gammasbetas[i])           
        count.append(c+1)     
        exps.append(expectationValueObj2(gammasbetas,constants))
        c = c + 1
   
    #just for plotting 
    for i in range(len(gammasbetas)):
        plt.plot(count,gammabetacounts[i],'r')
    plt.xlabel('steps')
    plt.ylabel('gammas = blau and betas = rot')
    plt.show()
    plt.plot(count,exps)
    plt.xlabel('steps')
    plt.ylabel('expectationvalue')
    plt.show() 
    
    stepsdiff = (steps - 2*(len(gammasbetas)) - 1 )/3
    steps = steps - stepsdiff - 1 
    
    return gammasbetasopt ,-exps[c-1], steps

def getpartialgradient(flag,gammasbetas,eps,constants):
    temp = gammasbetas[flag]
    gammasbetas[flag] += eps
    firstterm = expectationValueObj2(gammasbetas,constants)
    gammasbetas[flag] -= 2*eps
    secondterm = expectationValueObj2(gammasbetas,constants)
    gammasbetas[flag] += eps
    partialgrad = (firstterm - secondterm)/(2*eps)   
    return partialgrad

def getnormofgradient(gradients):
    normofgrad = 0 
    for i in range(len(gradients)):
        normofgrad += pow(abs(gradients[i]), 2)
    normofgrad = math.sqrt(normofgrad)    
    return normofgrad

#minusexp weil wir minimieren benutzen wir fuer pythonminimize 
def expectationValueObj(gammabeta,constants):
    qubo = constants[0]
    n = constants[1]
    edges = constants[2]
    cedges = constants[3]
    graph = constants[4]
    p = constants[5]
    shots = constants[6]
    backend = constants[7]
    provider=constants[8]
    noise = constants[9]
    
    global orderingsglobal
    
    if(qubo == "qubo1"):
        qc = makeCircuit1(n,edges,cedges,gammabeta,p)   
    if(qubo == "qubo2"):
        qc = makeCircuit2(n,edges,cedges,gammabeta,p)
    if(qubo == "qubo3"):
        qc = makeCircuit3(n,edges,cedges,gammabeta,p)
    if(qubo == "mixer"):
        qc = makeCircuit4(graph,n,edges,cedges,gammabeta,p)
    if(qubo == "mixerdicke"):
        qc = makeCircuit5(graph,n,edges,cedges,gammabeta,p)
    if(qubo == "mixersim"):
        qc = makeCircuit6(graph,n,edges,cedges,gammabeta,p)
    if((qubo == "mixerrand" ) or (qubo == "mixerrandnoise" )):
        qc = makeCircuit7(graph,n,edges,cedges,gammabeta,p,orderingsglobal)
    if(qubo == "mixerdickerand"):
        qc = makeCircuit8(graph,n,edges,cedges,gammabeta,p,orderingsglobal)
    if(qubo == "mixerrandminus"):
        qc = makeCircuit9(graph,n,edges,cedges,gammabeta,p,orderingsglobal)
          
    result = quantumsimulation(qc,n,shots,backend,provider=provider,noise = noise)
    global steps  
    steps += 1 
    dictofresults = result.get_counts().items()

    variance = 0
    exp = 0
    maxval =  - float("inf")
    minval =  float("inf")
    for key, value in dictofresults:
        x         = [int(num) for num in list(key)]
        if((qubo == "qubo3") or (qubo == "mixerrandnoise")):
            val = cost2(x,edges,cedges)
        if((qubo == "mixer" ) or (qubo == "mixerdicke") or (qubo == "mixerrand") or (qubo == "mixerdickerand") or (qubo == "mixerrandminus") ):
            val = cost4(x,edges,cedges)
        exp = exp + (value/shots) * val 
        if(val > maxval):
            maxval = val 
        if(val < minval):
            minval = val
    return  - exp


#the expecation value only used for gradient deccesnt  
def expectationValueObj2(gammabeta,constants):
    qubo = constants[0]
    n = constants[1]
    edges = constants[2]
    cedges = constants[3]
    graph = constants[4]
    p = constants[5]
    shots = constants[6]
    backend = constants[7]
    provider = constants[8]
    noise = constants[9]
    
    global orderingsglobal
    if(qubo == "qubo1"):
        qc = makeCircuit1(n,edges,cedges,gammabeta,p)   
    if(qubo == "qubo2"):
        qc = makeCircuit2(n,edges,cedges,gammabeta,p)
    if(qubo == "qubo3"):
        qc = makeCircuit3(n,edges,cedges,gammabeta,p)
    if(qubo == "mixer"):
        qc = makeCircuit4(graph,n,edges,cedges,gammabeta,p)
    if(qubo == "mixerdicke"):
        qc = makeCircuit5(graph,n,edges,cedges,gammabeta,p)
    if(qubo == "mixersim"):
        qc = makeCircuit6(graph,n,edges,cedges,gammabeta,p)
    if((qubo == "mixerrand" ) or (qubo == "mixerrandnoise" )):
        qc = makeCircuit7(graph,n,edges,cedges,gammabeta,p,orderingsglobal)
    if(qubo == "mixerdickerand"):
        qc = makeCircuit8(graph,n,edges,cedges,gammabeta,p,orderingsglobal)
    if(qubo == "mixerrandminus"):
        qc = makeCircuit9(graph,n,edges,cedges,gammabeta,p,orderingsglobal)
          
    result = quantumsimulation(qc,n,shots,backend,provider=provider,noise = noise )
    global steps  
    steps += 1 
    dictofresults = result.get_counts().items()
   
    exp = 0
    maxval =  - float("inf")
    minval =  float("inf")
    for key, value in dictofresults:
        x         = [int(num) for num in list(key)]
        if((qubo == "qubo3") or (qubo == "mixerrandnoise")):
            val = cost2(x,edges,cedges)
        if((qubo == "mixer" ) or (qubo == "mixerdicke") or (qubo == "mixerrand") or (qubo == "mixerdickerand") or (qubo == "mixerrandminus")):
            val = cost4(x,edges,cedges)
        exp = exp + (value/shots) * val 
        if(val > maxval):
            maxval = val 
        if(val < minval):
            minval = val


    #return  math.log(exp)
    return -exp


#gets a quantumgate and simulates it and then returns the output 
#what is depthweightcnots?
def quantumsimulation(qc1,n,shots,backend,plotcircuit = False,cedges = False,depthweightcnots = False,provider = False,noise = False):
    if(depthweightcnots):
        qc1 = transpile(qc1, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
        if(len(cedges) == 0):
            cnots = 0
        else:
            #cnots = 0
            cnots = qc1.count_ops()['cx']
        depth = qc1.depth()

    qc1.measure(range(n),range(n))
    if(plotcircuit):
        display(qc1.draw('mpl'))
        
    if(str(backend) == "qasm_simulator" ):
        if(noise):
            backend2 = provider.get_backend('ibmq_santiago')
            noise_model = NoiseModel.from_backend(backend2)
            coupling_map = backend2.configuration().coupling_map
            basis_gates = noise_model.basis_gates      
            simulate     = execute(qc1, backend=backend, shots=shots,coupling_map=coupling_map,
                                        basis_gates=basis_gates,
                                        noise_model=noise_model)
        else:
            simulate     = execute(qc1, backend=backend, shots=shots)
            
        
    else:
        simulate     = execute(qc1, backend=backend, shots=shots)

    result = simulate.result()
       
    if(depthweightcnots):
        return result,cnots,depth
    else:
        return result 
    
#calculate the probability of sucess 
def propabilityofsucess(qubo,result,edges,cedges,n,shots,histogram=False):
    maxcliques = []
    #fix this the max value must be 
    values = []    
    maxvalue = 0
    propability = 0
    keys = allpossiblekeys(n)
    hist = {}
    for key in keys:
        x         = [int(num) for num in list(key)]
        if(qubo == "qubo1"):
            costofx = cost1(x,edges,cedges)
        else:
            costofx = cost2(x,edges,cedges)          
        if(costofx == maxvalue):
            maxcliques.append(x)
        if(costofx > maxvalue):
            maxcliques = [x]
            maxvalue = costofx
          
    for key, value in result.get_counts().items():     
        x         = [int(num) for num in list(key)]
        if(qubo == "qubo1"):
            costofx = cost1(x,edges,cedges)
        else:
            costofx = cost2(x,edges,cedges)
            
        if costofx in hist.keys():
            hist[costofx] = hist[costofx]  +  value
        else: 
            hist[costofx] = value  
           
        if(costofx == maxvalue):
            propability += value
    propability = propability/shots
    if(histogram):
        display(plot_histogram(hist,figsize = (8,6),bar_labels = False))

    return propability,maxcliques

#get all possible n-bitstrings
def allpossiblekeys(n):
    keys = []
    if(n==0):
        return keys
    if(n == 1):
        return ['0','1']
    else:
        keys2 = allpossiblekeys(n-1)
        for i in keys2:
            keys.append('0' + i)
        for i in keys2:
            keys.append('1' + i)
    return keys

