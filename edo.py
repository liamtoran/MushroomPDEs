import matplotlib.pyplot as plt
from scipy.integrate import ode
import numpy as np

b = 0.5  # dtC=-b*rho*C
F0 = 1  # dtRho = Fo*Mu
tf = 20  # temps final de la simulation
rho0 = 0  # rho initial
mu0 = 0.1  # mu initial
c0 = 1  # concentration initiale
n = 2000  # nombre de pas de temps

# Résolution du schéma éxplicite
def euler_explicite_edo(b, F0, tf, rho0, mu0, c0, n):
    # t0<t1 , temps etudies,
    # rho0, mu0,c0 reels positifs: condition initiale
    # n entier(nombre d'iterations)
    h = tf / n  # pas Deltat
    rho = rho0
    mu = mu0
    c = c0
    t = 0
    Rho = [rho0]
    Mu = [mu0]
    C = [c0]
    T = [t]
    for k in range(n):
        new_mu = mu + h * (c * (mu + rho) - mu * rho)
        new_rho = rho + h * F0 * mu
        new_c = c - h * b * rho * c
        mu = new_mu
        rho = new_rho
        c = new_c
        t = t + h
        Mu.append(new_mu)
        Rho.append(new_rho)
        C.append(new_c)
        T.append(t)
    return T, Mu, Rho, C


# Résolution du schéma semi- implicite I
def euler_semi_I_edo(b, F0, tf, rho0, mu0, c0, n):
    # t0<t1 , temps etudies,
    # rho0, mu0,c0 reels positifs: condition initiale
    # n entier(nombre d'iterations)
    h = tf / n  # pas Deltat
    rho = rho0
    mu = mu0
    c = c0
    t = 0
    Rho = [rho0]
    Mu = [mu0]
    C = [c0]
    T = [0]
    for k in range(n):
        new_mu = (mu + h * c * rho) / (1 + h * rho - h * c * (1 + h * F0))
        new_rho = rho + h * F0 * new_mu
        new_c = c / (1 + b * h * new_rho)
        mu = new_mu
        rho = new_rho
        c = new_c
        t = t + h
        Mu.append(new_mu)
        Rho.append(new_rho)
        C.append(new_c)
        T.append(t)
    return T, Mu, Rho, C


# Résolution du schéma semi- implicite II
def euler_semi_II_edo(b, F0, tf, rho0, mu0, c0, n):
    # t0<t1 , temps etudies,
    # rho0, mu0,c0 reels positifs: condition initiale
    # n entier(nombre d'iterations)
    t = 0
    h = tf / n  # pas Deltat
    rho = rho0
    mu = mu0
    c = c0
    Rho = [rho0]
    Mu = [mu0]
    C = [c0]
    T = [0]
    for k in range(n):
        new_mu = (mu + h * c * rho - h * rho * mu) / (1 - h * c * (1 + h * F0))
        new_rho = rho + h * F0 * new_mu
        new_c = c / (1 + b * h * new_rho)
        mu = new_mu
        rho = new_rho
        c = new_c
        t = t + h
        Mu.append(new_mu)
        Rho.append(new_rho)
        C.append(new_c)
        T.append(t)
    return T, Mu, Rho, C


# Programmation de la méthode de Newton-Raphson
def newton(f, gradf, newton_steps, x0):
    x = x0
    for k in range(newton_steps):
        x = x - f(x) / gradf(x)
    return x


# Résolution du schéma implicite
def euler_implicite_edo(b, F0, tf, rho0, mu0, c0, n):
    # t0<t1 , temps etudiés,
    # rho0, mu0,c0 reels positifs: conditions initiale
    # n entier(nombre d'itérations)
    t = 0
    newton_steps = 10  # nombre d'itérations de la méthode de Newton-Raphson pour le calcul implicite
    h = tf / n  # pas deltat
    rho = rho0
    mu = mu0
    c = c0
    Rho = [rho0]
    Mu = [mu0]
    C = [c0]
    T = [0]
    for k in range(n):
        # Calcul de new_mu par methode de Newton Raphson
        # coefficients du polynome d'ordre 3 en new_mu
        alpha = -(h**4) * F0**2 * b
        beta = -F0 * h**2 * (b + 1 + 2 * rho * b * h)
        gamma = (
            -(1 + b * h * rho)
            + b * h**2 * F0 * mu
            + h * (c * (1 + h * F0) - rho * (1 + b * h * rho))
        )
        delta = (1 + b * h * rho) * mu + h * c * rho

        def P(X):
            return alpha * X**3 + beta * X**2 + gamma * X + delta

        def gradP(X):
            return 3 * alpha * X**2 + 2 * beta * X + gamma

        new_mu = newton(P, gradP, newton_steps, mu)
        new_rho = rho + h * F0 * new_mu
        new_c = c / (1 + b * h * new_rho)
        mu = new_mu
        rho = new_rho
        c = new_c
        t = t + h
        Mu.append(new_mu)
        Rho.append(new_rho)
        C.append(new_c)
        T.append(t)
    return T, Mu, Rho, C


# Utilisation des libraries python (scipy) pour résoudre l'EDO
def black_box_edo(b, F0, tf, rho0, mu0, c0, n):
    def f(t, y, arg1, arg2):
        mu = y[0]
        rho = y[1]
        c = y[2]
        return [c * (rho + mu) - mu * rho, F0 * mu, -b * rho * c]

    r = ode(f).set_integrator("zvode", method="adams")
    r.set_initial_value([mu0, rho0, c0], 0).set_f_params(F0, b)
    dt = tf / (n - 1)
    Rho = [rho0]
    Mu = [mu0]
    C = [c0]
    t = 0
    T = [0]
    while r.t < tf:
        mu, rho, c = r.integrate(r.t + dt)
        Mu.append(mu)
        Rho.append(rho)
        C.append(c)
        T.append(r.t)
    return T, Mu, Rho, C


# Résolution
T, Mu, Rho, C = euler_semi_I_edo(b, F0, tf, rho0, mu0, c0, n)


rho_inf = Rho[n - 1]
# Étude Asymptotique
A = [np.log(mu) for mu in Mu]
B = [-min(1, b) * y * rho_inf for y in T]

# Tracé des solutions et de l'étude asymptotique
plt.subplot(221)
plt.plot(T, Mu)
plt.ylabel("mu")
plt.xlabel("t")
plt.subplot(222)
plt.plot(T, Rho)
plt.ylabel("rho")
plt.subplot(223)
plt.plot(T, C)
plt.ylabel("C")
plt.subplot(224)
plt.plot(T, A)
plt.plot(T, B)
plt.ylabel("log(mu), -b*rho_inf*t")
plt.show()
