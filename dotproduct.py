# -*- coding: utf-8 -*-
"""
Low-Power Artificial Neural Network Perceptron Based on Monolayer MoS2

Guilherme Migliato Marega 1,2, Zhenyu Wang 1,2, Maksym Paliy 4, Gino Giusi 5, Sebastiano Strangio 4, Francesco Castiglione 6, Christian Callegari 6, Mukesh Tripathi 1,2, Aleksandra Radenovic 3, Giuseppe Iannaccone 4,6*, Andras Kis 1,2* 

1 - Institute of Electrical and Microengineering, École Polytechnique Fédérale de Lausanne (EPFL),  
CH-1015 Lausanne, Switzerland
2 - Institute of Materials Science and Engineering, École Polytechnique Fédérale de Lausanne (EPFL), CH-1015 Lausanne, Switzerland
3 - Institute of Bioengineering, École Polytechnique Fédérale de Lausanne (EPFL),
CH-1015 Lausanne, Switzerland
4 - Department of Information Engineering, University of Pisa, I-56122 Pisa, Italy
5 -  Engineering Department, University of Messina, I-98166 Messina, Italy
6 - Quantavis s.r.l., Largo Padre Renzo Spadoni snc, I-56123 Pisa, Italy

*Correspondence should be addressed to Andras Kis, andras.kis@epfl.ch and Giuseppe Iannaccone, giuseppe.iannaccone@unipi.it

"""

"""
Function: To post-processing experimental results in order to apply the nonlinear function at 
each neuron output, analyse accuracy in function of input signal noise and plot the experimental results.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


# Parameters
w_step = 0.20; # Pre-Processing : data generation 
gain = 10 # Digital Gain

# Generating Weight Vectors
w1 = 1 + w_step;
w2 = 0 - w_step;

W = np.zeros((7,10));

interator = 0;
while(w1 >= w_step):
    w1 -= w_step;
    w2 += w_step; 
    
    W[0][interator] = w1;
    W[1][interator] = w2; 
    interator+=1
    
    
xdata = np.zeros((200,7));

input_max = 14
input_step = 1/input_max

interator = 0;
for i in range(0,14):
    for j in range(0,14):
        xdata[interator][0] = (j+1)*input_step
        xdata[interator][1] = (i+1)*input_step           
        interator+=1
        
# Saving generated data

np.savetxt("dataset_W_dotproduct.csv", W, delimiter=";")     
np.savetxt("dataset_x_dotproduct.csv", xdata, delimiter=";")     

        
'''Post-Processing : Dot Product'''

# Loading Experimental Data
number = []
for i in range(0,10):
    with open('./Number'+str(i)+'/FeedFoward') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            row = [float(x) for x in row]
            number.append(np.array(row))
    
number = np.array(number)
number = np.transpose(number)
number = np.reshape(number, (202, 10))

''' Cleaning Experimental Data'''
number = gain*number    # Applying scaling factor to experimental values
number = number[2:,:]   # Cleaning dummy values in the vector

# Creating XY plane grid
x = np.arange(input_step, 1+input_step, input_step)
y = np.arange(input_step, 1+input_step, input_step)
X, Y = np.meshgrid(x, y)
XX = X.flatten()
YY = Y.flatten()


# Ploting Data
plt.rcParams["figure.figsize"] = (20,20)
for i in range(0,6):
    
    # Slicing and reshaping data
    sliced_data = np.array([number[j][i] for j in range(0,196)]) 
    Z = np.reshape(sliced_data, (14,14))
    # data_wrap = np.c_[x,y,number[:,i]]
    # A = np.c_[data_wrap[:,0], data_wrap[:,1], np.ones(data_wrap.shape[0])]
    # C,_,_,_ = scipy.linalg.lstsq(A, data_wrap[:,2])    # coefficients
    # Zfit = C[0]*X + C[1]*Y + C[2]
    
    
    # http://inversionlabs.com/2016/03/21/best-fit-surfaces-for-3-dimensional-data.html
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(30, 210)
    fig.set_dpi(100)
    ax.grid(False)
    
    #ax.plot_surface(X, Y, Zfit, rstride=1, cstride=1, alpha=0.1)
    
    #ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap=cm.plasma, alpha=0.8)
    #surf = ax.scatter3D(Y, X, Z, alpha = 1, marker = "." , color = "#a39d8b" , s = 100)
    dy = dx = (1/14)*np.ones_like(YY)
    dz = np.zeros_like(YY)
    cmap = cm.get_cmap('plasma')
    norm = Normalize(vmin=min(Z.flatten()), vmax=max(Z.flatten()))
    colors = cmap(norm(Z.flatten()))

    ax.bar3d(YY, XX, dz, dx, dy, Z.flatten(), color=colors)

    plt.rcParams.update({'font.size': 36})
    ax.set_zlim(-0.05, 1.05)
    ax.axis('tight')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()
    
    fig.savefig('DotProduct_'+str(i)+'.svg', format='svg')
