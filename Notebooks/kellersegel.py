import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import spsolve
import matplotlib.animation as animation

# Coéfficients physiques
K = 0.5  # coefficient diffusion
b = 0.2  # dtC=-b*rho*C
F0 = 1  # dtRho = Fo*Mu
ll = 0.1  # 1/Lambda

# Paramêtres numériques
n_t = 3001  # nombre de pas de temps
tf = 20  # temps final de la simulation
xf = 150  # longeur de la simulation
n_x = 601  # nombres de points de la simulation

# Données initiales
rho0 = np.zeros(n_x)  # rho initial
mu0 = np.zeros(n_x)  # mu initial
mu0[(n_x // 2) : (n_x // 2 + 10)] = 0.01
c0 = np.zeros(n_x) + 1  # concentration initiale


def KPP_mem_KS(K, ll, b, F0, rho0, mu0, c0, n_t, tf, xf, n_x):
    # Détermination des paramêtres numeriques deltat et deltax
    dt = tf / (n_t - 1)
    dx = xf / (n_x - 1)
    # Représentation de l'éspace et du temps
    X = np.linspace(0, xf, n_x)
    T = np.linspace(0, tf, n_t)
    # Initialisation
    Mu = np.zeros((n_t, n_x))
    Rho = np.zeros((n_t, n_x))
    C = np.zeros((n_t, n_x))
    Mu[0] = mu0
    Rho[0] = rho0
    C[0] = c0
    # Résolution	du schéma implicite-explicite I
    for n in range(0, n_t - 1):
        # Matrice de -Laplacien
        A = (
            np.diag(-np.ones(n_x - 1), -1)
            + np.diag(2 * np.ones(n_x), 0)
            + np.diag(-np.ones(n_x - 1), 1)
        )
        A *= 1 / (dx**2)

        # Matrice de la Divergence
        B = np.diag(-np.ones(n_x), 0) + np.diag(np.ones(n_x - 1), 1)
        B *= 1 / dx

        # Ajout des termes implicites
        M = K * dt * A  # Terme laplacien

        alpha = -C[n] * dt * (1 + dt * F0) + dt * Rho[n] + 1 + ll * dt * ((-A) @ (C[n]))
        M += np.diag(alpha, 0)  # termes scalaires en mu

        M += ll * dt * (B @ (C[n])) * B  # terme KS 2

        M = sp.csc_matrix(M)

        # Résolution du systême implicite
        Target = Mu[n] + dt * C[n] * Rho[n]
        Mu[n + 1] = spsolve(M, Target)
        Rho[n + 1] = Rho[n] + dt * F0 * Mu[n + 1]
        C[n + 1] = C[n] / (1 + b * dt * Rho[n])
    return X, T, Mu, Rho, C


X, T, Mu, Rho, C = KPP_mem_KS(K, ll, b, F0, rho0, mu0, c0, n_t, tf, xf, n_x)


def speed(X, Rho, rho_inf):
    # Position du front
    argmed = np.zeros(n_t)
    for i in range(n_t):
        argmed[i] = X[
            (n_x // 2)
            + np.min(np.where(np.append(Rho[i, (n_x // 2) :], [0]) < rho_inf / 2))
        ]
    # Vitesse du front
    s = ((n_t - 1) / tf) * (argmed[(n_t // 2) + 150] - argmed[(n_t // 2)]) / (150)
    return s


rho_inf = Rho[n_t - 1, (n_x // 2)]
s = speed(X, Rho, rho_inf)
print("La vitesse de propagation de la simulation est s=", s)


s_theorique = np.sqrt(
    K
    * ((18 * F0 + 4) + np.sqrt(((18 * F0 + 4) ** 2) + 108 * (1 + 4 * F0) * (F0**2)))
    / (2 * (1 + 4 * F0))
)
# Attention, ceci est pour C0=1
print("La vitesse théorique de propagation est s_theorique=", s_theorique)

# Animation
fig = plt.figure()

ax = plt.axes(xlim=(0, xf), ylim=(0, rho_inf + 1))
(line,) = ax.plot([], [], lw=2)
(line2,) = ax.plot([], [], lw=2)
(line3,) = ax.plot([], [], lw=2)
(line4,) = ax.plot([], [], lw=2)
time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)
legend_text = ax.text(0.80, 0.82, "", transform=ax.transAxes)


def init():
    line.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    time_text.set_text("")
    legend_text.set_text("")
    return line, line2, line3, line4, time_text, legend_text


def animate(i):
    line.set_data(X, C[i])
    line2.set_data(X, Rho[i])
    line3.set_data(X, Mu[i])
    line4.set_data(xf / 2 + ((i * s) * tf / (n_t - 1)), np.linspace(0, rho_inf + 1, 10))
    time_text.set_text("time = {0:.1f}\n K={1}, b={2}, F0={3} ".format(T[i], K, b, F0))
    legend_text.set_text("Rho=Orange \nMu=Green \nC=Blue\ns={0:.3f}".format(s))
    return line, line2, line3, line4, time_text, legend_text


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=(n_t - 1),
    interval=(tf * 200) / (n_t - 1),
    blit=True,
)


# anim.save('EDP_1D.gif',writer='imagemagick', fps=30)
plt.show()
