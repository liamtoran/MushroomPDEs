import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import spsolve
import matplotlib.animation as animation

#Coéfficients physiques
K=.2 #coefficient diffusion
b=.2 # dtC=-b*rho*C
F0=1 # dtRho = Fo*Mu

#Paramêtres numériques 
n_t=501 #nombre de pas de temps
tf=30 # temps final de la simulation
xf = 100 #longeur de lasimulation
n_x =600 #nombres de points de la simulation

#Données initiales 
rho0=np.zeros(n_x) #rho initial	
mu0=np.zeros(n_x) #mu initial
mu0[(n_x//2):(n_x//2 +10)]=.01
c0=np.zeros(n_x)+1 #concentration initiale

def edp_1d_explicite(K, b, F0, rho0, mu0, c0, n_t , tf, xf, n_x):
	dt=tf/(n_t-1)
	dx=xf/(n_x-1)
	X=np.linspace(0,xf,n_x)
	T=np.linspace(0,tf,n_t)
	Mu=np.zeros((n_t,n_x))
	Rho=np.zeros((n_t,n_x))
	C=np.zeros((n_t,n_x))
	Mu[0]=mu0
	Rho[0]=rho0
	C[0]=c0
	#Résolution	du schema éxplicite
	for n in range(0,n_t-1):
		RHS=np.zeros(n_x)
		alpha=-C[n]*dt*(1+dt*F0)+dt*Rho[n]+1
		RHS[1:-1]= dt*((K/(dx**2))*(Mu[n,:-2]-2*Mu[n,1:-1]+Mu[n,2:])+C[n,1:-1]*Rho[n,1:-1])
		RHS[0]= dt*((K/(dx**2))*(-2*Mu[n,0]+Mu[n,1])+C[n,0]*Rho[n,0])
		RHS[-1]=dt*((K/(dx**2))*(-2*Mu[n,-1]+Mu[n,-2])+C[n,-1]*Rho[n,-1])
		Mu[n+1]=(1/alpha)*(Mu[n]+RHS)
		Rho[n+1]=Rho[n]+dt*F0*Mu[n+1]
		C[n+1]=C[n]/(1 + b*dt*Rho[n])
	return X,T,Mu,Rho,C

def edp_1d_semi_implicite_I(K, b, F0, rho0, mu0, c0, n_t , tf, xf, n_x):
	#Détermination des paramêtres numeriques deltat et deltax
	dt=tf/(n_t-1)
	dx=xf/(n_x-1)
	#Représentation de l'éspace et du temps
	X=np.linspace(0,xf,n_x)
	T=np.linspace(0,tf,n_t)
	#Initialisation
	Mu=np.zeros((n_t,n_x))
	Rho=np.zeros((n_t,n_x))
	C=np.zeros((n_t,n_x))
	Mu[0]=mu0
	Rho[0]=rho0
	C[0]=c0
	#Résolution	du schéma implicite-explicite I
	for n in range(0,n_t-1):
		alpha=-C[n]*dt*(1+dt*F0)+dt*Rho[n]+1
		A=np.diag(-np.ones(n_x-1),-1)+np.diag(2*np.ones(n_x),0)+np.diag(-np.ones(n_x-1),1)
		A=A*K*dt/(dx**2)
		A+=np.diag(alpha,0)
		Mu[n+1]= spsolve(A, Mu[n]+dt*C[n]*Rho[n])
		Rho[n+1]=Rho[n]+dt*F0*Mu[n+1]
		C[n+1]=C[n]/(1 + b*dt*Rho[n])
	return X,T,Mu,Rho,C

def edp_1d_semi_implicite_II(K, b, F0, rho0, mu0, c0, n_t , tf, xf, n_x):
	#Détermination des paramêtres numériques deltat et deltax
	dt=tf/(n_t-1)
	dx=xf/(n_x-1)
	#Représentation de l'éspace et du temps
	X=np.linspace(0,xf,n_x)
	T=np.linspace(0,tf,n_t)
	#Initialisation
	Mu=np.zeros((n_t,n_x))
	Rho=np.zeros((n_t,n_x))
	C=np.zeros((n_t,n_x))
	Mu[0]=mu0
	Rho[0]=rho0
	C[0]=c0
	#Résolution	du schéma implicite-explicite II
	for n in range(0,n_t-1):
		#Matrice du Laplacien
		A=np.diag(-np.ones(n_x-1),-1)+np.diag(2*np.ones(n_x),0)+np.diag(-np.ones(n_x-1),1)		
		A=A*K*dt/(dx**2) #Laplacien Numerique
		#Ajout des termes implicites
		alpha=-C[n]*dt*(1+dt*F0)+1
		A+=np.diag(alpha,0)
		#Résolution du systême implicite
		Mu[n+1]= spsolve(A, Mu[n]+dt*C[n]*Rho[n]-dt*Mu[n]*Rho[n])
		Rho[n+1]=Rho[n]+dt*F0*Mu[n+1]
		C[n+1]=C[n]/(1 + b*dt*Rho[n])
	return X,T,Mu,Rho,C
		
X,T,Mu,Rho,C= edp_1d_semi_implicite_I(K, b, F0, rho0, mu0, c0, n_t , tf, xf, n_x)

#Valeur de rho a l'infini
rho_inf = Rho[n_t-1,(n_x//2)]
print(rho_inf)


def speed(X,Rho):
    #Position du front
	argmed=np.zeros(n_t)
	for i in range(n_t):
		argmed[i]= X[(n_x//2)+np.min(np.where(np.append(Rho[i,(n_x//2):],[0])<rho_inf/2))]
	#Vitesse du front
	S = (argmed[(n_t//2)+1:]-argmed[(n_t//2):-1])*((n_t-1)/tf)
	s= np.average(S)
	return s
s=0
s = speed(X,Rho)
print(s)


#Animation
fig = plt.figure()

ax = plt.axes(xlim=(0, xf), ylim=(0, rho_inf+1))
line, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
line3, = ax.plot([], [], lw=2)
line4, = ax.plot([], [], lw=2)
time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)
legend_text = ax.text(0.80, 0.82, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    time_text.set_text('')
    legend_text.set_text('')
    return line,line2,line3,line4, time_text, legend_text


def animate(i):
    line.set_data(X, C[i])
    line2.set_data(X, Rho[i])
    line3.set_data(X, Mu[i])
    #line4.set_data(50+((i*s)*tf/(n_t-1)),np.linspace(0,rho_inf+1,10))
    time_text.set_text('time = {0:.1f}\n K={1}, b={2}, F0={3} '.format(T[i],K,b,F0))
    legend_text.set_text('Rho=Orange \nMu=Green \nC=Blue\ns={0:.3f}'.format(s))
    return line,line2, line3,line4, time_text, legend_text


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=(n_t-1), interval=(tf*200)/(n_t-1), blit=True)
                               


#anim.save('EDP_1D.gif',writer='imagemagick', fps=30)
plt.show()
