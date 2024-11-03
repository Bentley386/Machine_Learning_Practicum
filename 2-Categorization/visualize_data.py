import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Perceptron

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import DBSCAN

class DataLoader:
    """Loads AND stores the data 
    (violating some OOP principles, but this is just a project...)"""
    
    def __init__(self,path,num=5750):
        """Path - path to the folder with ALL the relevant data
           num - Total number of different specters"""
        
        self.path = path
        self.num = num
    
    def loadAllSpectra(self):        
        self.wav = np.loadtxt(self.path + "val.dat", comments="#")
        spectra = []
        for i in range(1,self.num):
            flux = np.loadtxt(self.path + f"{i}.dat", comments="#")
            spectra.append(flux)
        self.spectra = np.array(spectra)
    
    def loadLabeledData(self):
        labeledData = np.loadtxt(self.path + "ucni_set_tipi.txt", delimiter=' ', dtype=str)
        self.types = {"CMP":[],"TRI":[],"HAE":[],"MAB":[],"DIB":[],"BIN":[],"HFR":[]}
        #self.types = {key : [] for key in set(labeledData[:,1])}
        for v, k in labeledData:
            self.types[k].append(int(v)-1) #integer n is n-1-th position in spectra array - confusing but will do for project
        self.labels = list(self.types.keys())
        
    def loadAll(self):
        self.loadAllSpectra()
        self.loadLabeledData()
    
    

class Plotter:
    """Just a collection of (non general) plotting functions, could be a module?"""
    
    def __init__(self, data, output):
        """Data is an initialized DataLoader object (no checks)"""

        self.data = data
        self.output = output

    def plotLabeledSpectra(self):
        fig, ax = plt.subplots(4,2,sharex=True,sharey=True, constrained_layout=True)
        fig.delaxes(ax[3][1]) #we have 7 labels
        for i, ax in enumerate(ax.ravel()[:-1]):
            if i in [0,2,4,6]:
                ax.set_ylabel("Flux")
            if i in [5,6]:
                ax.set_xlabel(r"$\lambda$")
                ax.tick_params(labelbottom=True)
    
            for spec in self.data.types[self.data.labels[i]]:
                ax.plot(self.data.wav,self.data.spectra[spec])
            ax.set_title(self.data.labels[i])
            ax.grid()
        fig.subplots_adjust(hspace=0.1)
        plt.savefig(self.output+"labeled.png",dpi=800)
        
    def plotPCA(self, dimensions):
        dimRed = DimRed(self.data)
        reduced = dimRed.pca(dimensions)
        if dimensions==2:
            fig, ax = plt.subplots()
            ax.scatter(reduced[:,0],reduced[:,1],s=0.1,c='k')
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1],s=5,label=k)
            ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
        elif dimensions==3:
            fig, ax = plt.subplots(subplot_kw={"projection" :"3d"})
            ax.scatter(reduced[:,0],reduced[:,1],reduced[:,2],s=0.1,c='k')
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1],reduced[v,2],s=5,label=k)
            ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        elif dimensions==4:
            fig, ax = plt.subplots(subplot_kw={"projection" :"3d"})
            scat = ax.scatter(reduced[:,0],reduced[:,1],reduced[:,2],s=0.1,c=reduced[:,3])
            cbar = plt.colorbar(scat, ax=ax)
            cbar.set_ticks([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        else:
            fig, ax = plt.subplots(dimensions,dimensions)
            for i in range(dimensions):
                for j in range(dimensions):
                    ax[i][j].scatter(reduced[:,i],reduced[:,j],color="k",s=0.1)
                    for k,v in self.data.types.items():
                        ax[i][j].scatter(reduced[v,i],reduced[v,j],s=1,label=k)
                        ax[i][j].set_xticks([])
                        ax[i][j].set_yticks([])
                    if j>=i:
                        fig.delaxes(ax[i][j])
        plt.tight_layout()
        plt.savefig(self.output + f"PCA{dimensions}D.png",dpi=800)
        
    def plotKernels(self):
        dimRed = DimRed(self.data)
        kernels = ["linear","cosine","rbf","sigmoid"]
        fig, ax = plt.subplots(2,2)
        for i,ax in enumerate(ax.ravel()):
            reduced = dimRed.kpca(2,kernels[i])
            ax.scatter(reduced[:,0],reduced[:,1],s=0.1,c='k')
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1],s=10,label=k)
            if i==0:
                ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(kernels[i])
        plt.savefig(self.output + "kernelPCA.png",dpi=800)
        
    def plotLLE(self):
        dimRed = DimRed(self.data)
        neighbours = [3+3*i for i in range(9)]
        fig, ax = plt.subplots(3,3, constrained_layout=True)
        for i,ax in enumerate(ax.ravel()):
            reduced = dimRed.lle(neighbours[i])
            ax.scatter(reduced[:,0],reduced[:,1],s=0.1,c='k')
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1],s=10,label=k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"$n={neighbours[i]}$")
        plt.savefig(self.output + "LLE.png", dpi=800)

    def plotTSNE(self):
        dimRed = DimRed(self.data)
        perp = [5+5*i for i in range(9)]
        fig, ax = plt.subplots(3,3, constrained_layout=True)
        for i,ax in enumerate(ax.ravel()):
            reduced = dimRed.tsne(perp[i])
            ax.scatter(reduced[:,0],reduced[:,1],s=0.1,c='k')
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1],s=10,label=k)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"$P={perp[i]}$")
        plt.savefig(self.output + "tsne.png", dpi=800)
        
    def plotLogisticRegressor(self,dimension, cv=False):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        if cv:
            lab = clas.logisticRegressionCV(dimension)
        else:
            lab = clas.logisticRegression(dimension)

        fig, ax = plt.subplots()
        for i, (k,v) in enumerate(self.data.types.items()):
            ax.scatter(reduced[(lab==i+1),0],reduced[(lab==i+1),1], s=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_prop_cycle(None)
        for k,v in self.data.types.items():
            print(k)
            ax.scatter(reduced[v,0],reduced[v,1], s=50, marker="x", label=k)
        ax.legend(ncols=2)
        if cv:
            plt.savefig(self.output + f"logisticCV{dimension}.png", dpi=800)
        else:
            plt.savefig(self.output + f"logistic{dimension}.png", dpi=800)
            
    def plotPerceptron(self):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        lab = clas.perceptron()
        fig, ax = plt.subplots()
        for i, (k,v) in enumerate(self.data.types.items()):
            ax.scatter(reduced[(lab==i+1),0],reduced[(lab==i+1),1], s=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_prop_cycle(None)
        for k,v in self.data.types.items():
            print(k)
            ax.scatter(reduced[v,0],reduced[v,1], s=50, marker="x", label=k)
        ax.legend(ncols=2)
        plt.savefig(self.output + f"perceptron.png", dpi=800)
        
    def plotSVC(self):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        fig, ax = plt.subplots(2,2)
        kernels = ["linear","poly","rbf","sigmoid"]
        for i,ax in enumerate(ax.ravel()):
            lab = clas.svc(kernels[i])
            for j, (k,v) in enumerate(self.data.types.items()):
                ax.scatter(reduced[(lab==j+1),0],reduced[(lab==j+1),1], s=0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_prop_cycle(None)
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1], s=50, marker="x", label=k)
            #ax.legend()
            ax.set_title(kernels[i])
        plt.savefig(self.output + "svc.png",dpi=800)
        
    def plotTree(self):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        fig, ax = plt.subplots(2,2)
        depths = [2,3,5,10]
        for i,ax in enumerate(ax.ravel()):
            lab = clas.tree(depths[i])
            for j, (k,v) in enumerate(self.data.types.items()):
                ax.scatter(reduced[(lab==j+1),0],reduced[(lab==j+1),1], s=0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_prop_cycle(None)
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1], s=50, marker="x", label=k)
            #ax.legend()
            ax.set_title(f"Depth: {depths[i]}")
        plt.savefig(self.output + "tree.png",dpi=800)

    def plotGradTree(self):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        fig, ax = plt.subplots(2,2)
        params = [(50,1),(100,1),(50,2),(100,2)]
        for i,ax in enumerate(ax.ravel()):
            lab = clas.gradTree(params[i][0],params[i][1])
            for j, (k,v) in enumerate(self.data.types.items()):
                ax.scatter(reduced[(lab==j+1),0],reduced[(lab==j+1),1], s=0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_prop_cycle(None)
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1], s=50, marker="x", label=k)
            #ax.legend()
            ax.set_title(f"N : {params[i][0]}, depth: {params[i][1]}")
        plt.savefig(self.output + "gradTree.png",dpi=800)
        
    def plotRandomTree(self):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        fig, ax = plt.subplots(2,2)
        params = [(50,1),(100,1),(50,2),(100,2)]
        for i,ax in enumerate(ax.ravel()):
            lab = clas.randomTree(params[i][0],params[i][1])
            for j, (k,v) in enumerate(self.data.types.items()):
                ax.scatter(reduced[(lab==j+1),0],reduced[(lab==j+1),1], s=0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_prop_cycle(None)
            for k,v in self.data.types.items():
                ax.scatter(reduced[v,0],reduced[v,1], s=50, marker="x", label=k)
            #ax.legend()
            ax.set_title(f"N : {params[i][0]}, depth: {params[i][1]}")
        plt.savefig(self.output + "randomTree.png",dpi=800)
        
    def plotNB(self):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        lab = clas.nb()
        fig, ax = plt.subplots()
        for i, (k,v) in enumerate(self.data.types.items()):
            ax.scatter(reduced[(lab==i+1),0],reduced[(lab==i+1),1], s=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_prop_cycle(None)
        for k,v in self.data.types.items():
            ax.scatter(reduced[v,0],reduced[v,1], s=50, marker="x", label=k)
        ax.legend(ncols=2)
        plt.savefig(self.output + f"nb.png", dpi=800)
      
    def plotDBSCAN(self):
        dimRed = DimRed(self.data)
        reduced = dimRed.tsne(15)
        clas = Classifier(reduced,self.data.labels,self.data.types)
        lab = clas.dbscan()
        fig, ax = plt.subplots()
        print(np.unique(lab))
        for i in np.unique(lab):
            print("wut")
            ax.scatter(reduced[(lab==i),0],reduced[(lab==i),1], s=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(self.output + f"dbscan.png", dpi=800)
        

class DimRed:
    
    def __init__(self,data):
        
        self.data = data
        spectra = self.data.spectra
        
        self.scaler = StandardScaler()
        self.scaler.fit(spectra)
        self.transformedSpectra = self.scaler.transform(spectra)
    
    def pca(self,dimensions):
        
        pca = PCA(n_components=dimensions)
        return pca.fit_transform(self.transformedSpectra)
    
    def kpca(self,dimensions,kernel):
        pca = KernelPCA(n_components=dimensions,kernel=kernel)
        return pca.fit_transform(self.transformedSpectra)
    
    def lle(self,neighbours,dimensions=2):
        lle = LocallyLinearEmbedding(n_neighbors=neighbours,n_components=dimensions)
        return lle.fit_transform(self.transformedSpectra)
    
    def tsne(self,perplexity):
        pca = PCA(n_components=100)
        data = pca.fit_transform(self.transformedSpectra)
        tsne = TSNE(perplexity=perplexity)
        results = []
        KLs = []
        for i in range(1):
            results.append(tsne.fit_transform(data))
            KLs.append(tsne.kl_divergence_)
        return results[np.argmin(KLs)]
    
class Classifier:
    
    def __init__(self, data, labels, types):
        self.data = data
        self.labels = labels
        self.labeled = types
        
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        
        self.transformedData = self.scaler.transform(data)
        
        self.Y = [i+1 for i,k in enumerate(self.labels) for _ in self.labeled[k]]
        self.X = [self.transformedData[i] for _,k in enumerate(self.labels) for i in self.labeled[k]]
        
    def logisticRegression(self,degree):
        if degree==1:
            log = LogisticRegression()
            log.fit(self.X,self.Y)
            return log.predict(self.transformedData)
        else:
            poly = PolynomialFeatures(degree=degree)
            X2 = poly.fit_transform(self.X)
            
            log = LogisticRegression()
            log.fit(X2,self.Y)

            transformed = poly.fit_transform(self.transformedData)
            return log.predict(transformed)

    def logisticRegressionCV(self,degree):

        if degree==1:
            log = LogisticRegressionCV()
            log.fit(self.X,self.Y)
            return log.predict(self.transformedData)
        else:
            poly = PolynomialFeatures(degree=degree)
            X2 = poly.fit_transform(self.X)
            
            log = LogisticRegressionCV()
            log.fit(X2,self.Y)

            transformed = poly.fit_transform(self.transformedData)
            return log.predict(transformed)
    def perceptron(self):
        perc = Perceptron(early_stopping=True,validation_fraction=0.2)
        perc.fit(self.X,self.Y)
        return perc.predict(self.transformedData)
    
    def svc(self,kernel):
        svc = SVC(kernel=kernel)
        svc.fit(self.X,self.Y)
        return svc.predict(self.transformedData)
        
    def tree(self, max_depth):
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(self.X,self.Y)
        return tree.predict(self.transformedData)
        
    def gradTree(self,estimators,depth):
        tree = GradientBoostingClassifier(n_estimators=estimators,max_depth=depth)
        tree.fit(self.X,self.Y)
        return tree.predict(self.transformedData)

    def randomTree(self,estimators,depth):
        tree = RandomForestClassifier(n_estimators=estimators,max_depth=depth)
        tree.fit(self.X,self.Y)
        return tree.predict(self.transformedData)
    
    def nb(self):
        nb = GaussianNB()
        nb.fit(self.X,self.Y)
        return nb.predict(self.transformedData)
    
    def dbscan(self):
        dbscan = DBSCAN()
        return dbscan.fit_predict(self.transformedData)
        

data = DataLoader("./spektri/")
data.loadAll()
plotter = Plotter(data,"./plots/")
#plotter.plotLabeledSpectra()
#plotter.plotPCA(2)
#plotter.plotPCA(3)
#plotter.plotPCA(6)
#plotter.plotKernels()
#plotter.plotLLE()
#plotter.plotTSNE()

#plotter.plotLogisticRegressor(1)
#plotter.plotLogisticRegressor(3)
#plotter.plotLogisticRegressor(3,True)
plotter.plotPerceptron()
#plotter.plotSVC()
#plotter.plotTree()
#plotter.plotGradTree()
#plotter.plotRandomTree()
#plotter.plotNB()