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

# Import useful libraries
import numpy as np # For Matrix Operations
import matplotlib.pyplot as plt # For Ploting
import csv # For reading csv files 

# Definition of softmax modified from : 
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x, nnumbers = 3):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/ np.transpose(np.tile(np.sum(np.exp(x),axis = 1), (nnumbers,1)))

# Dataset Parameters
nnumbers = 3    # Number of classification outputs
nsamples = 5    # Number of Aquisitions per parameter
size = 202      # Single dataset size

# Parameters
gain = 10       # Digital Gain - defined during experimental to scale conductance values 
                # to the correct abstract values in close range of [0-error,1+error]
accuracy = []   # Initializing Accuracy Matrix
count = -1

''' Interating over each noise value measurement staring from 0.1 std to 0.5 std.'''
for sigma in ['0p1','0p25','0p5']: 
    
    '''Loading base dataset'''
    # Loading Theoretical Weights Values
    w = []
    with open('./Dataset/weights_'+sigma+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            row = [float(x) for x in row]
            w.append(np.array(row))
            
    w = np.array(w)
    w = np.transpose(w)
    
    # Loading Input Dataset
    inputs = []
    with open('./Dataset/dataset_xtest_'+sigma+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            row = [float(x) for x in row]
            inputs.append(np.array(row))
            
    inputs = np.array(inputs)
    inputs = np.transpose(inputs)
    
    # Loading Output Dataset
    y = []
    with open('./Dataset/dataset_ytest_'+sigma+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            row = [float(x) for x in row]
            y.append(row)
            
    y = np.array(y)
    
    # Generating theoretical outputs from the matrix multiplication: y = W.x
    out = np.matmul(w,inputs)
    out = np.transpose(out)  
    
    count += 1;
    accuracy.append([]);
    
    ''' Reading each measurement with defined sigma'''
    for meas in range(1,nsamples+1): 
        # Experimental Data loading
        number = []
        for i in range(0,nnumbers):
            with open('./Transfer of Learning '+str(meas)+'/'+sigma+' - sigma/Number'+str(i)+'/FeedFoward') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                line_count = 0
                for row in csv_reader:
                    row = [float(x) for x in row]
                    number.append(np.array(row))
            
        number = np.array(number)
        number = np.transpose(number)
        number = np.reshape(number, (size, nnumbers))
        
        # Application of digital gain on the output experimental values.
        number = gain*number
        # Removing two first data points which are dummies acquisitions to synchronise
        # the DAC and ADC.
        number = number[2:,:]
        
        '''Visualization of Experimental Outputs of each Neuron'''
        # Comparison between experimental and theorerical output
        for i in range(0,nnumbers):
            fig, ax = plt.subplots()
            # Uncomment if you want to compare predicted output and experimental output 
            # rects1 = ax.bar(np.arange(200),out[:,i],1, color = '#C7CEEA',alpha=1, label = 'theory')
            rects2 = ax.bar(np.arange(200),number[:,i],1, color = '#FF9AA2',alpha=0.50, label = 'experimental')
            fig.tight_layout()
            ax.set_ylabel('Signal internsity (a.u.)')
            ax.set_xlabel('Test Number')
            ax.set_title('DotProduct - Number '+str(i))
            ax.legend(loc='upper right', frameon=False)
            plt.savefig('./Images/comparison_number'+str(i)+'_'+sigma+'_'+str(meas)+'.svg')
            plt.show()
               
        '''Calculation of Output Classification Number'''
        # Applying softmax function to the Experimental Output
        softmax_realout = softmax(number)
        # Classification of Experimental 
        classification_realout = np.argmax(number ,axis = 1)
        
        # Applying softmax function to the Theoretical Output
        softmax_out = softmax(out)
        # Classification of Experimental 
        classification_out = np.argmax(out ,axis = 1)
        
        '''Calculation of Theoretical and Experimental Accuracy'''
        # Experimental Accuracy for defined measurement at sigma
        accuracy_real = np.sum(1*(classification_realout[i] == y[i]) for i in range(0,200))/200
        # Theoretical Accuracy for defined sigma 
        accuracy_theory =  np.sum(1*(classification_out[i] == y[i]) for i in range(0,200))/200
        
        print('# Measurement '+str(meas)+' - sigma:'+sigma);
        print('Real Accuracy: ', accuracy_real);
        print('Theory Accuracy: ', accuracy_theory);
        accuracy[-1].append(accuracy_real);
        
        '''Comparison of Predict Output and Experimental Output'''
        fig, ax = plt.subplots()
        # Uncomment if you want to compare predicted output and experimental output 
        plt.plot(np.arange(200),classification_out,'bo',label = 'Theory')
        plt.plot(np.arange(200),classification_realout,'r.', label = 'Experimental')
        ax.set_ylabel('Classification Number')
        ax.set_xlabel('Test Number')
        ax.set_title('Output Visualization')
        ax.legend(loc='upper right', frameon=True)
        plt.savefig('./Images/output_understanding_'+sigma+'_'+str(meas)+'.svg')
        plt.show()
        
        '''Loading Programming and Weight Matrix Experimental Values'''
        # Experimental Programming loading
        # The digital gain is already accounted in the weight matrix
        Programming = []    # Initializing Programming Matrix of each memory 
        Pulses = []         # Initializing Pulsing Matrix from previous programming matrix
        for i in range(0,nnumbers):
            Programming.append([])
            Pulses.append([])
            for j in range(0,7):  
                with open('./Transfer of Learning '+str(meas)+'/'+sigma+' - sigma/Number'+str(i)+'/Programming_'+str(j)) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter='\t')
                    counter=0
                    for row in csv_reader:
                        if(counter == 0):
                            prog = [float(x) for x in row]
                            Programming[i].append(np.array(prog))
                        if(counter == 1):
                            puls = [float(x) for x in row]
                            Pulses[i].append(np.array(puls))
                        counter+=1
                        
            # Extracting Convergence Curves of Each Memory
            Programming[i] = np.array(Programming[i])
            Programming[i] = np.transpose(Programming[i])
            Pulses[i] = np.array(Pulses[i])
            Pulses[i] = np.transpose(Pulses[i])
            
        # Ploting convergence plots for each memory
        for i in range(0,nnumbers):
            for j in range(0,7):
                N = len(Programming[i][j])
                fig, ax1 = plt.subplots()
                
                ax1.set_ylabel('Weight',color="#FF9AA2",fontsize=14)
                ax1.set_xlabel('Number os Pulses')
                ax1.set_title('Convergence '+sigma+' - Measurement #'+str(meas)+'- Memory'+str(i)+str(j))
                ax1.plot(np.arange(N),Programming[i][j],'o-',1, color = '#FF9AA2',alpha=1, label = 'experimental')
                #ax1.ylim([0,1.5])
                ax1.tick_params(axis='y', labelcolor='#FF9AA2')
                
                ax2=ax1.twinx()  
                ax2.set_ylabel("Vpulse (V)",color="#C7CEEA",fontsize=14)
                ax2.bar(np.arange(N),Pulses[i][j],1, color = '#C7CEEA',alpha=0.50, label = 'experimental')
                ax2.tick_params(axis='y', labelcolor='#C7CEEA')
                ax2.legend(loc='upper right', frameon=True)
                
                
                plt.savefig('./Images/ConvergenceProgramming_'+sigma+'_'+str(meas)+'_'+str(i)+str(j)+'.svg')
                fig.tight_layout()
                plt.show()     
                
                # fig, ax = plt.subplots()
                # rects2 = ax.bar(np.arange(N),Pulses[i][j],1, color = '#C7CEEA',alpha=0.50, label = 'experimental')
                # ax.set_ylabel('Weight')
                # ax.set_ylabel('V_Pulses')
                # ax.set_xlabel('Number os Pulses')
                # ax.set_title('Convergence '+sigma+' - Measurement #'+str(meas)+'- Memory'+str(i)+str(j))
                # ax.legend(loc='upper right', frameon=True)
                # plt.savefig('./Images/ConvergencePulses_'+sigma+'_'+str(meas)+'_'+str(i)+str(j)+'.svg')
                # plt.show()  
            
        '''Weight Maps - Experimental Maps'''
        # Extracting weight maps fromt the convergence curves
        w_experimental = []
        for i in range(0,nnumbers):
            w_experimental.append([])
            for j in range(0,7):
                w_experimental[i].append(Programming[i][j][-1])

        w_experimental = np.array(w_experimental)     
                
        # Ploting weight matrix - comparison between theoretical and experimental
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        
        im = axes.flat[0].imshow(w, vmin=0, vmax=1.3)
        axes.flat[1].imshow(w_experimental, vmin=0, vmax=1.3)
        
        axes.flat[1].set_xlabel('Display pixel (a-g)')
        axes.flat[1].set_ylabel('Output Neuron')
        axes.flat[1].set_title('Weight Matrix - Experimental')
        
        axes.flat[0].set_xlabel('Display pixel (a-g)')
        axes.flat[0].set_ylabel('Output Neuron')
        axes.flat[0].set_title('Weight Matrix - Theory')
                
        fig.colorbar(im, ax=axes.flat[1])
        
        fig.tight_layout(pad=6.0)
        plt.savefig('./Images/WMaps_'+sigma+'_'+str(meas)+'.svg');
        plt.show()
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                