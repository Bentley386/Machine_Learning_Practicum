import numpy as np
import matplotlib.pyplot as plt
import random


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras

def loadData(path,lamb,noise):
    """
    Loads the simulated data (Ks, theta0s and Intensities)

    Path: String, containing a path to the relevant .npy files
    lamb: The wavelength used in the simulation
    noise: the amount of white noise added to the intensities
    """
    
    K = np.load(path+"Kvalues.npy") #shape = (100k) 
    theta0 = np.load(path+"theta0.npy") #shape (100k,200)
    intensity = np.load(f"{path}intensity{lamb}noise{noise}.npy")
    return {"K" : K, "theta0" : theta0, "intensity" : intensity}
    
def visualizeSelectedFew(inPath,outPath):
    """ 
    Make visualization plot for report
    
    inPath : string, path to the relevant data (.npy)
    outPath: string, path to the output directory
    """
    
    basic = loadData(inPath,500,0)
    diffLamb = loadData(inPath,700,0)
    
    fig, ax = plt.subplots(3,3, constrained_layout=True,sharey="row")
    
    x = list(range(1,201))
    t = list(range(1,401))
    for i in range(3):
        K = "{:.2e}".format(float(basic['K'][i]))
        theta0 = basic['theta0'][i]
        ax[0][i].plot(x,theta0)
        ax[0][i].set_title(f"K: {K}, $\\theta_0$")
        if i==0:
            ax[0][i].set_ylabel(r"$\theta_0$")
        ax[0][i].set_xlabel(r"$z$")
        
        ax[1][i].plot(t,basic['intensity'][i])
        ax[1][i].set_title(f"K: {K}, $I(t)$, $\\lambda=500$")
        if i==0:
            ax[1][i].set_ylabel(r"$I$")
        ax[1][i].set_xlabel(r"$t$")
        
        ax[2][i].plot(t,diffLamb['intensity'][i])
        ax[2][i].set_title(f"K: {K}, $I(t)$, $\\lambda=700$")
        if i==0:
            ax[2][i].set_ylabel(r"$I$")
        ax[2][i].set_xlabel(r"$t$")
            

    plt.savefig(outPath+"data.png",dpi=700)
    
class myNetwork:
    
    def __init__(self):
        self.model = Sequential()
        
    def setup(self, parameters):
        
        activation = parameters['activation']
        optimizer = parameters['optimizer']
        loss = parameters['loss']
        metrics = parameters['metrics']
        kernel_init = parameters['kernel_init']
        neurons = parameters['neurons']
        self.layers = parameters['layers']

        self.model.add(Dense(neurons, activation= activation, kernel_initializer=kernel_init, input_shape=(400,)))
        for i in range(1,self.layers):
            self.model.add(Dense(neurons, activation=activation, kernel_initializer=kernel_init))
        self.model.add(Dense(1,activation= activation))
        self.model.compile(optimizer= optimizer,loss= loss, metrics= metrics)
    
    def train(self,X,Y, Xtest, Ytest, batch, epochs):
        
        self.history = self.model.fit(X,Y,batch_size=batch, epochs=epochs, validation_data = (Xtest,Ytest))
        
    def plotHistory(self, outFile):
        plt.plot(self.history.history['loss'],label="Training data")
        plt.plot(self.history.history['val_loss'],label="Validation data")
        plt.ylabel("Mean squared error")
        plt.xlabel("Epoch")
        plt.legend()
        #plt.tight_layout()
        plt.title(f"{self.layers} hidden layers")
        plt.savefig(outFile + f"test{self.layers}Layers.png")
parameters = {"activation" : "relu", "optimizer" : "adam", "loss" : "mean_squared_error", "metrics" : ["mean_absolute_error"], "kernel_init" : "he_uniform" , "neurons" : 200, "layers" : 1}


plt.style.use("seaborn-v0_8")

PATH = "DataK/"
lambs = ["500","505","700","900"]
data = [loadData(PATH,lamb,"0") for lamb in lambs]
Ks = [d['K'] for d in data]
Ks = np.concatenate(Ks)
Is = [d['intensity'] for d in data]
Is = np.vstack(Is)


shuffle = np.random.permutation(len(Ks))
Ks = Ks[shuffle]
Ks = Ks*0.5*1e11
Is = Is[shuffle,:]

trainNum = int(0.95*len(Ks))
Xtrain = Is[:trainNum,:]
Ytrain = Ks[:trainNum]

Xtest = Is[trainNum:,:]
Ytest = Ks[trainNum:]

net = myNetwork()
net.setup(parameters)
net.train(Xtrain, Ytrain, Xtest, Ytest, 32, 50)
net.plotHistory("Figures/")
#visualizeSelectedFew("DataK/","Figures/")


    