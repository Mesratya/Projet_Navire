import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


def ls(Y,A):
    N = A.T @ A # Matrice normale
    Xhat = np.linalg.inv(N) @ A.T @ Y # Estimation des parametres
    return(Xhat)

# Recuperation de donnée
data = np.loadtxt('poutrelle_vide.txt')
x = data[:,0:1]
y = data[:,1:2]
z = data[:,2:3]

# remise sous forme de colonnes
x = np.reshape(x,(len(x),1))
y = np.reshape(y,(len(x),1))
z = np.reshape(z,(len(x),1))

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x,y,z,marker = ".")

ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
plt.title("poutrelle vide")



# Construction de Y et A en fonction des données
Y = z
A = np.hstack((y**2,y,np.ones((len(y),1))))

# # Moindre carrée
Xhat = ls(Y,A)
Yhat = np.dot(A,Xhat) # on en déduit la modélisation
V = Y - Yhat # résidu
m_V = np.mean(V)
S_V = np.std(V)

fig = plt.figure()
y0 = np.linspace(0,max(y),1000)
z_hat = Xhat[0]*(y0**2) + Xhat[1]*y0 + Xhat[2]
plt.scatter(y,z,c= "blue",alpha = 0.5,label = "poutrelle vide")
plt.plot(y0,z_hat,"black",linestyle = "--",label = "Moindres carrés polynomial (2nd°)" )

# Recuperation de donnée
data = np.loadtxt('poutrelle_charge.txt')
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
Yhat_2 = np.dot(A,Xhat) # on en déduit la modélisation
V = Y - Yhat_2 # résidu
m_V = np.mean(V)
S_V = np.std(V)


#tracé
y0 = np.linspace(0,max(y),1000)
z_hat_2 = Xhat[0]*(y0**2) + Xhat[1]*y0 + Xhat[2]

plt.scatter(y,z,c = "red",alpha = 0.5,label = "poutrelle chargée")
plt.plot(y,Yhat_2,"black",linestyle = "--")
# plt.plot(Yhat[0:len(x),0],Yhat[len(x):len(Yhat),0],'red')
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x,y,Yhat_2,marker = ".")

ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
plt.title("poutrelle vide")





# # petit test avec scipy.optimize.curvefit sur une surface quadratique !
#
# def quad(T,a,b,c,d,e,f):
#     x,y = T
#     return(a*(x**2) + b*(y**2) + c*x*y + d*x + e*y + f)
# T = (x,y)
# popt, pcov = curve_fit(quad,T,z)


plt.legend()
plt.show()

