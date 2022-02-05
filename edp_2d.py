import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import spsolve
from scipy.sparse.linalg import (
    bicgstab,
    bicg,
    cg,
    cgs,
    gmres,
    lgmres,
    minres,
    qmr,
    gcrotmk,
)
from IPython.display import HTML
import time

start_time = time.time()

# Coéfficients physiques
K = 0.2  # coefficient diffusion
b = 0.2  # dtC=-b*rho*C
F0 = 1  # dtRho = Fo*Mu

physique = [K, b, F0]

# Paramêtres numériques
n_t = 500  # nombre de pas de temps
tf = 170  # temps final de la simulation
xf = 500  # longueur de la simulation
n_x = 1100  # nombres de points de la simulation
yf = xf
n_y = n_x
n_xy = n_x * n_y
numerique = [n_t, tf, xf, n_x, yf, n_y]

params = physique, numerique


# Données initiales
rho0 = np.zeros(n_xy)  # rho initial
mu0 = np.zeros(n_xy)  # mu initial
mu0[((n_xy + n_x) // 2) : ((n_xy + n_x) // 2) + 1] = 0.01
c0 = np.zeros(n_xy) + 1  # concentration initiale

xm = 200
xM = 300
ym = 120
yM = 220
im = (xm * n_x) // xf
iM = (xM * n_x) // xf
jm = (ym * n_y) // xf
jM = (yM * n_y) // xf
for i in range(im, iM):
    for j in range(jm, jM):
        c0[i + j * n_x] = 0.0


class EDP:
    def __init__(self, params):
        self.physique, self.numerique = params
        self.K, self.b, self.F0 = self.physique
        self.n_t, self.tf, self.xf, self.n_x, self.yf, self.n_y = self.numerique

        self.n_xy = self.n_x * self.n_y
        self.dt = self.tf / (self.n_t - 1)
        self.dx = self.xf / (self.n_x - 1)
        self.dy = self.yf / (self.n_y - 1)

        # self.X = np.linspace(0,self.xf,self.n_x)
        # self.Y = np.linspace(0,self.yf,self.n_y)
        # self.T = np.linspace(0,self.tf,self.n_t)

        # Matrice du Laplacien
        self.Lapl = sp.diags(-4 * np.ones(self.n_xy), 0)
        # Lapl += sp.diags(np.ones(n_xy-1),1)+sp.diags(np.ones(n_xy-1),-1)
        diagmod = np.ones(self.n_xy - 1)
        diagmod[np.arange(self.n_y - 1, self.n_xy - 1, self.n_y)] = np.zeros(
            self.n_y - 1
        )
        self.Lapl += sp.diags(diagmod, 1) + sp.diags(diagmod, -1)
        self.Lapl += sp.diags(np.ones(self.n_xy - self.n_y), self.n_y) + sp.diags(
            np.ones(self.n_xy - self.n_y), -self.n_y
        )
        self.Lapl = -self.K * self.dt / (self.dx**2) * self.Lapl
        self.Cond = sp.identity(self.n_xy)

    def array_to_2D(n_x, vect):
        return np.array(np.split(vect, n_x))

    def integrate(self, initial):
        mu, rho, c = initial
        alpha = -c * self.dt * (1 + self.dt * self.F0) + self.dt * rho + 1
        A = self.Lapl + sp.diags(alpha, 0)
        Target = mu + self.dt * c * rho

        # next_mu = spsolve(A,Target) #95.28 secondes d'execution
        # next_mu,check = bicg(A,Target) #3.38 secondes d'execution
        # next_mu,check = bicgstab(A,Target, x0=mu) #2.15 secondes d'execution
        # next_mu,check = cg(A,Target) #2.29 secondes d'execution
        # next_mu,check = cgs(A,Target) #2.36 secondes d'execution
        # next_mu,check = gmres(A,Target) #2.72 secondes d'execution
        # next_mu,check = lgmres(A,Target) #2.62 secondes d'execution
        next_mu, check = minres(
            A, Target, x0=mu, M=self.Cond
        )  # 2.15 secondes d'execution
        # next_mu,check = qmr(A,Target) #3.70 secondes d'execution
        # next_mu,check = gcrotmk(A,Target) #2.62 secondes d'execution
        next_rho = rho + self.dt * self.F0 * next_mu
        next_c = c / (1 + self.b * self.dt * next_rho)
        return next_mu, next_rho, next_c


Agent = EDP(params)
dt = Agent.dt

mu = mu0
rho = rho0
c = c0
Mu = [mu0]
Rho = [rho0]
C = [c0]
T = [0]
n = 0
step = 5
while n < n_t:
    mu, rho, c = Agent.integrate((mu, rho, c))
    if n % step == 0:
        Mu.append(mu)
        Rho.append(rho)
        C.append(c)
        T.append(n * dt)
    if n % 25 == 0:
        print(n, (time.time() - start_time))
    n += 1

print("--- %s seconds ---" % (time.time() - start_time))

tot = len(Mu)

Draw = "C"
f = C

fig = plt.figure()
im = plt.imshow(EDP.array_to_2D(n_x, f[2]), animated=True, cmap="PiYG")
time_text = plt.text(0, 0, "")

i = 2


def updatefig(*args):
    global i
    i += 1
    if i < tot - 1:
        im.set_array(EDP.array_to_2D(n_x, f[i]))
        time_text.set_text("time = {0:.1f}".format(T[i]))
    return im, time_text


ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True, repeat=True)
ani.save("EDP_2D_" + Draw + ".gif", writer="imagemagick", fps=30)

Draw = "Mu"
f = Mu

fig = plt.figure()
im = plt.imshow(EDP.array_to_2D(n_x, f[2]), animated=True, cmap="PiYG")
time_text = plt.text(0, 0, "test")
i = 2


def updatefig(*args):
    global i
    i += 1
    if i < tot - 1:
        im.set_array(EDP.array_to_2D(n_x, f[i]))
        time_text.set_text("time = {0:.1f}".format(T[i]))
    return im, time_text


ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True, repeat=True)
ani.save("EDP_2D_" + Draw + ".gif", writer="imagemagick", fps=30)


Draw = "Rho"
f = Rho

fig = plt.figure()
im = plt.imshow(EDP.array_to_2D(n_x, f[2]), animated=True, cmap="PiYG")
time_text = plt.text(0, 0, "")
i = 2


def updatefig(*args):
    global i
    i += 1
    if i < tot - 1:
        im.set_array(EDP.array_to_2D(n_x, f[i]))
        time_text.set_text("time = {0:.1f}".format(T[i]))
    return im, time_text


ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True, repeat=True)
ani.save("EDP_2D_" + Draw + ".gif", writer="imagemagick", fps=30)


plt.show()
