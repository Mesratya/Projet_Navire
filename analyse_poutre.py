import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


def ls(Y,A):
    N = A.T @ A # Matrice normale
    Xhat = np.linalg.inv(N) @ A.T @ Y # Estimation des parametres
    return(Xhat)

def Quad_surf(x,y,a,b,c,d,e,f):
    return(a*(x**2) + b*(y**2) + c*x*y + d*x + e*y + f)

# Recuperation de donnée
poutre = ["poutrelle_vide","poutrelle_charge"][1]
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

ax.scatter(x,y,z,marker = ".",c="gray",alpha = 0.2)

x_s = np.linspace(min(x), max(x), 30)
y_s = np.linspace(min(y), max(y), 30)
x_s,y_s = np.meshgrid(x_s,y_s)
(a,b,c,d,e,f) = Xhat
a,b,c,d,e,f = a[0],b[0],c[0],d[0],e[0],f[0]
z_s = a*(x_s**2) + b*(y_s**2) + c*x_s*y_s + d*x_s + e*y_s +f
ax.scatter(x_s,y_s,z_s,color = "red",alpha = 0.5,s = 0.5)

ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
plt.title(poutre)

plt.figure()
plt.plot(1000*V)
plt.title("Résidus " + poutre)
plt.xlabel(" indice des points")
plt.ylabel("résidu [mm]")




# # Recuperation de donnée
# data = np.loadtxt('poutrelle_charge.txt')
# x = data[:,0:1]
# y = data[:,1:2]
# z = data[:,2:3]
#
# # remise sous forme de colonnes
# x = np.reshape(x,(len(x),1))
# y = np.reshape(y,(len(x),1))
# z = np.reshape(z,(len(x),1))
#
#
#
# # Construction de Y et A en fonction des données
# Y = z
# A = np.hstack((x**2,y**2,x*y,x,y,np.ones((len(y),1))))
#
# # # Moindre carrée
# Xhat = ls(Y,A)
# Yhat_2 = np.dot(A,Xhat) # on en déduit la modélisation
# V = Y - Yhat_2 # résidu
# m_V = np.mean(V)
# S_V = np.std(V)
#
#
# #tracé
# y0 = np.linspace(0,max(y),1000)
# z_hat_2 = Xhat[0]*(y0**2) + Xhat[1]*y0 + Xhat[2]
#
# plt.scatter(y,z,c = "red",alpha = 0.5,label = "poutrelle chargée")
# plt.plot(y,Yhat_2,"black",linestyle = "--")
# # plt.plot(Yhat[0:len(x),0],Yhat[len(x):len(Yhat),0],'red')
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.scatter(x,y,z)
# ax.scatter(x,y,Yhat_2,marker = ".")
#
# ax.set_xlabel('x(m)')
# ax.set_ylabel('y(m)')
# ax.set_zlabel('z(m)')
# plt.title("poutrelle chargé")
#
#
#
#




plt.legend()
plt.show()

