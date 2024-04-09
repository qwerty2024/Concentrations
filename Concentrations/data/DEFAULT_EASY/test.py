
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time

m = int(input())
n = int(input())

Xs = np.empty(m * n)
Ys = np.empty(m * n)

counter = 0;
for i in range(m * n): 
    if i % n == 0:
        counter = 0 
    Xs[i] = counter     
    counter = counter + 1

counter = m - 1
for i in range(m * n):    
    if i % m == 0 and (i != 0):
        counter = counter - 1
    Ys[i] = counter 

index = 1;

file = str(index) + ".txt"
data = np.genfromtxt(file, delimiter=' ', dtype=np.float64)

cs = []

for i in range(m * n):
    if data[i][0] == 0:
        cs.append([1.0,1.0,1.0])

    if data[i][0] == 1:
        cs.append([1.0,0.0,0.0,data[i][1]])
                                     
    if data[i][0] == 2:              
        cs.append([0.0,1.0,0.0,data[i][1]])
                                     
    if data[i][0] == 3:              
        cs.append([0.0,0.0,1.0,data[i][1]])
                                     
    if data[i][0] == 4:              
        cs.append([0.0,1.0,1.0,data[i][1]])
                                     
    if data[i][0] == 5:              
        cs.append([1.0,0.0,1.0,data[i][1]])
                                     
    if data[i][0] == 6:              
        cs.append([1.0,1.0,0.0,data[i][1]])

plt.scatter(Xs,Ys,s=10,color=cs)
plt.show()



while(True): 
    
    index = index + 1
    
    file = str(index) + ".txt"
    if (os.path.exists(file) == False):
        break;
    
    data = np.genfromtxt(file, delimiter=' ', dtype=np.float64)

    cs.clear()

    for i in range(m * n):
        if data[i][0] == 0:
            cs.append([1.0,1.0,1.0])
    
        if data[i][0] == 1:
            cs.append([1.0,0.0,0.0,data[i][1]])
                                        
        if data[i][0] == 2:              
            cs.append([0.0,1.0,0.0,data[i][1]])
                                        
        if data[i][0] == 3:              
            cs.append([0.0,0.0,1.0,data[i][1]])
                                        
        if data[i][0] == 4:              
            cs.append([0.0,1.0,1.0,data[i][1]])
                                        
        if data[i][0] == 5:              
            cs.append([1.0,0.0,1.0,data[i][1]])
                                        
        if data[i][0] == 6:              
            cs.append([1.0,1.0,0.0,data[i][1]])          


    plt.scatter(Xs,Ys,s=10,color=cs)
    
    #plt.clf()
    
    plt.show()
    
    #time.sleep(1.5)
    

    


