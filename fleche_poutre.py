import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib import cm

def ls(Y,A):
    N = A.T @ A # Matrice normale
    Xhat = np.linalg.inv(N) @ A.T @ Y # Estimation des parametres
    return(Xhat)

def Quad_surf(x,y,a,b,c,d,e,f):
    return(a*(x**2) + b*(y**2) + c*x*y + d*x + e*y + f)


# Recuperation de donnée
poutres = ["poutrelle_vide","poutrelle_charge"]
poutre = poutres[0]
resol = 500 #nombre de noeud du maillage pour calculer le modèle suivant x ou y
data = np.loadtxt(poutre + ".txt")
x = data[:,0:1]
y = data[:,1:2]
z = data[:,2:3]

# remise sous forme de colonnes
x = np.reshape(x,(len(x),1))
y = np.reshape(y,(len(x),1))
z = np.reshape(z,(len(x),1))

# Construction de Y et A en fonction des données
Y = z
A = np.hstack((x**2,y**2,x*y,x,y,np.ones((len(y),1))))

# # Moindre carrée
Xhat = ls(Y,A)
Yhat = np.dot(A,Xhat) # on en déduit la modélisation
V = Y - Yhat # résidu
m_v = np.mean(V)
S_v = np.std(V)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# #ax.scatter(x,y,z,marker = ".",c="gray",alpha = 0.2)

x_s = np.linspace(min(x), max(x), resol)
y_s = np.linspace(min(y), max(y), resol)
x_s,y_s = np.meshgrid(x_s,y_s)
(a,b,c,d,e,f) = Xhat
a,b,c,d,e,f = a[0],b[0],c[0],d[0],e[0],f[0]
z_s_vide = a*(x_s**2) + b*(y_s**2) + c*x_s*y_s + d*x_s + e*y_s +f

poutre = poutres[1]
data = np.loadtxt(poutre + ".txt")
x = data[:,0:1]
y = data[:,1:2]
z = data[:,2:3]

# remise sous forme de colonnes
x = np.reshape(x,(len(x),1))
y = np.reshape(y,(len(x),1))
z = np.reshape(z,(len(x),1))

# Construction de Y et A en fonction des données
Y = z
A = np.hstack((x**2,y**2,x*y,x,y,np.ones((len(y),1))))

# # Moindre carrée
Xhat = ls(Y,A)
Yhat = np.dot(A,Xhat) # on en déduit la modélisation
V = Y - Yhat # résidu
m_v = np.mean(V)
S_v = np.std(V)

fig = plt.figure()
ax = fig.gca(projection='3d')
# #ax.scatter(x,y,z,marker = ".",c="gray",alpha = 0.2)

x_s = np.linspace(min(x), max(x), resol)
y_s = np.linspace(min(y), max(y), resol)
x_s,y_s = np.meshgrid(x_s,y_s)
(a,b,c,d,e,f) = Xhat
a,b,c,d,e,f = a[0],b[0],c[0],d[0],e[0],f[0]
z_s_charge = a*(x_s**2) + b*(y_s**2) + c*x_s*y_s + d*x_s + e*y_s +f

deplacement = z_s_charge - z_s_vide

surf = ax.plot_surface(x_s,y_s,deplacement,cmap = cm.coolwarm ,linewidth=0, antialiased=False)
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('déplacement(m)')

plt.title("déplacement de la poutre")

fleche = np.min(deplacement)
# plt.figure()
# plt.plot(1000*V)
# plt.title("Résidus " + poutre)
# plt.xlabel(" indice des points")
# plt.ylabel("résidu [mm]")









plt.legend()
plt.show()

