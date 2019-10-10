import numpy as np
from scipy.spatial import ConvexHull,Delaunay
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def ppp(f,i):
    l = np.zeros((f.shape[0]+1,f.shape[1]))
    a = ConvexHull(f)
    b = f[a.simplices[i]]
    c = np.mean(b,axis=0)
    d = np.cross(b[1]-b[0],b[2]-b[0])
    
    for k in range(f.shape[0]):
        l[k]=f[k]
    l[-1]=c + 0.001*d
    
    e = ConvexHull(l)
    g = f.shape[0] in list(e.vertices)
    gg = d[2]>0
    
    if gg + g ==2 or gg+g == 0:
        return True
    return False

def decompose(w):
    return (w+abs(w))/2,(-w+abs(w))/2


def Mink(l):
    a = sep(l[0])
    liste = l[1:]
    t = len(a)
    for i in liste:
        print(t,'t')
        if len(i.shape) == 1:
            p=[]
            for j in a:
                p.append(j+i)
                
            a = p[:]    
        else:
            p = []
            t = t*i.shape[0]
            for k in range(i.shape[0]):
                for j in a:
                    p.append(j+i[k])
            
            a = p[:] 
        
                
    return a


def sep(el):
    a=[]
    if len(el.shape)==1:
        return [el]
    else :
        for k in range(el.shape[0]):
            a.append(el[k])
        return a


def clean(l):
    a=[]
    for i in l:
        if np.count_nonzero(i)>0:
            a.append(i)
            
    return a



def hyp(simplices,points):
    
    e = [ppp(points,i)*1 for i in range(simplices.shape[0])]
    c=0
    for s in simplices:
        p = list(s)
        p.append(s[0])
        #if a[c]==0:
        #if d[c]==1:
        if e[c]==1:
            for k in range(3):
                x=[points[p[k]][0],points[p[k+1]][0]]
                y=[points[p[k]][1],points[p[k+1]][1]]
                plt.plot(x,y,marker = 'o')
        c+=1
    plt.xlim(-2,2.5)
    plt.ylim(-2,2.5)
    plt.show()
    
    
    
            
def hypersurface_one(poids,s):
    
    W1 = poids['f1.weight']
    
    b1 = poids['f1.bias'].reshape(-1,1)
    
    W2 = poids['f2.weight']
    
    b2 = poids['f2.bias']
    
    
    

    def decompose(w):
        return (w+abs(w))/2,(-w+abs(w))/2


    PG1 = np.concatenate((decompose(W1)[1],0*b1.reshape(-1,1)),axis=1)
    PH1 = np.concatenate((decompose(W1)[0],b1.reshape(-1,1)),axis=1)
    PF1 = np.stack((PG1,PH1),axis=1)


    H2_PF1 = np.zeros((5,2,3)) 
    for i in range(5): 
        H2_PF1[i]= PF1[i] * decompose(W2)[1][0][i]
    H2_PG1 = np.zeros((5,2,3)) 
    for i in range(5): 
        H2_PG1[i]= PG1[i] * decompose(W2)[0][0][i]
        
    G2_PF1 = np.zeros((5,2,3)) 
    for i in range(5): 
        G2_PF1[i]= PF1[i] * decompose(W2)[0][0][i]
    G2_PG1 = np.zeros((5,2,3)) 
    for i in range(5): 
        G2_PG1[i]= PG1[i] * decompose(W2)[1][0][i]


    H2 = []
    for i in range(5):
        if decompose(W2)[1][0][i] !=0:
            H2.append(PF1[i] * decompose(W2)[1][0][i])
            print('f',i+1,decompose(W2)[1][0][i])
        else:
            H2.append(PG1[i] * decompose(W2)[0][0][i])
            print('g',i+1,decompose(W2)[0][0][i])
            
            
    G2 = []
    for i in range(5):
        if decompose(W2)[1][0][i] !=0:
            G2.append(PG1[i] * decompose(W2)[1][0][i])
            print('g',i+1,decompose(W2)[1][0][i])
        else:
            G2.append(PF1[i] * decompose(W2)[0][0][i])
            print('f',i+1,decompose(W2)[0][0][i])
            
            
    F2 = Mink(clean(H2))+ Mink(clean(G2))
    
    if s =='f':
    
    
        FF = list(set(map(tuple, F2)))
        f3 = np.zeros((len(FF),3))
        for i in range(len(FF)):
            f3[i,0]=FF[i][0]
            f3[i,1]=FF[i][1]
            f3[i,2]=FF[i][2]
            
        print(f3,f3.shape)

        hull = ConvexHull(f3)
        
        return hyp(hull.simplices,f3)
    
    
    if s =='g':
    
    
        FF = list(set(map(tuple,  Mink(clean(G2)))))
        g3 = np.zeros((len(FF),3))
        for i in range(len(FF)):
            g3[i,0]=FF[i][0]
            g3[i,1]=FF[i][1]
            g3[i,2]=FF[i][2]
        print(g3,g3.shape)
        
        
        
    if s =='h':
    
    
        FF = list(set(map(tuple,  Mink(clean(H2)))))
        h3 = np.zeros((len(FF),3))
        for i in range(len(FF)):
            h3[i,0]=FF[i][0]
            h3[i,1]=FF[i][1]
            h3[i,2]=FF[i][2]
        print(h3,h3.shape)
        
        h3h = ConvexHull(h3)
    
        fig = plt.figure() # For plotting
        ax = fig.add_subplot(111, projection='3d')
        for s in h3h.simplices:
            p2 = h3[s]
            ax.plot_trisurf(p2[:,0], p2[:,1], p2[:,2],cmap='viridis', linewidth=0.3)
            print(s)
        ax.view_init(elev=0, azim=0)
        fig.tight_layout()
    
    
