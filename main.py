import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def diff4_segment(f):
    diff_f = np.zeros([2, N + 1])
    for i in range(N + 1):
        diff_f[0, i] = -g * (f[1, (i - 2 + N) % N] / 12 - (2 / 3) * f[1, (i - 1 + N) % N] + (2 / 3) * f[1, (i + 1 + N) % N] - f[1, (i + 2 + N) % N] / 12) / dx
        diff_f[1, i] = -h0 * (f[0, (i - 2 + N) % N] / 12 - (2 / 3) * f[0, (i - 1 + N) % N] + (2 / 3) * f[0, (i + 1 + N) % N] - f[0, (i + 2 + N) % N] / 12) / dx
    return diff_f


def animatewater(sol):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(a-1, b+1), ylim=(5, 17))
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    line, = ax.plot([], [], lw=2)
    time_template = 'time = %1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = x
        thisy = sol[i, 1, :]+h0
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i * dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, len(sol), interval=dt * 1000, blit=True, repeat=False)
    plt.show()
    return ani


def RK4(state):
    k1 = diff4_segment(state)
    k2 = diff4_segment(state + k1 * dt / 2)
    k3 = diff4_segment(state + k2 * dt / 2)
    k4 = diff4_segment(state + k3 * dt)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def matrix_diff(f):
    diff = ((D @ f.transpose()) / dx).transpose()
    diff = np.flip(diff, 0)
    diff[0, :] = -g * diff[0, :]
    diff[1, :] = -h0 * diff[1, :]
    diff[0, 0] = 0
    diff[0, -1] = 0
    return diff


def RK4D(state):
    k1 = matrix_diff(state)
    k2 = matrix_diff(state + k1 * dt / 2)
    k3 = matrix_diff(state + k2 * dt / 2)
    k4 = matrix_diff(state + k3 * dt)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solveEquation(method):
    sol = np.zeros([Nt + 1, 2, N + 1])
    sol[0, :, :] = ini_state
    for i in range(1, Nt + 1):
        sol[i, :, :] = method(sol[i - 1, :, :])
    return sol


def matrix_factory():
    G = np.zeros([N + 1, N + 1])
    H = np.eye(N + 1, N + 1)
    H[:4, :4] = [[17 / 48, 0, 0, 0], [0, 59 / 48, 0, 0], [0, 0, 43 / 48, 0], [0, 0, 0, 49 / 48]]
    H[-4:, -4:] = [[49 / 48, 0, 0, 0], [0, 43 / 48, 0, 0], [0, 0, 59 / 48, 0], [0, 0, 0, 17 / 48]]
    G[:4, :6] = [[-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0, 0], [-59 / 96, 0, 59 / 96, 0, 0, 0],
                 [1 / 12, -59 / 96, 0, 59 / 96, -1 / 12, 0],
                 [1 / 32, 0, -59 / 96, 0, 2 / 3, -1 / 12]]
    for i in range(6):
        for j in range(6):
            G[N - i, N - j] = -G[i, j]
    for i in range(4, N - 3):
        G[i, i - 2:i + 3] = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
    return np.linalg.inv(H) @ G, H

def upg_matrix_factory(order):
    BP = 4
    NES = 2
    xb = np.zeros(5)
    xb[0] = 0.0000000000000e+00
    xb[1] = 6.8764546205559e-01
    xb[2] = 1.8022115125776e+00
    xb[3] = 2.8022115125776e+00
    xb[4] = 3.8022115125776e+00
    for i in range(5 - (NES+1)):
        xb = xb[:-1]
    global dx, x
    dx = (b - a)/(2 * xb[-1] + N - 1 - 2*NES)
    x = dx * np.concatenate([xb, np.linspace(xb[-1]+1, (b - a)/dx - xb[-1]-1, N-2*(NES+1)).transpose(), (b - a)/dx - np.flip(xb, 0)])
    P = np.zeros(BP)
    P[0] = 2.1259737557798e-01
    P[1] = 1.0260290400758e+00
    P[2] = 1.0775123588954e+00
    P[3] = 9.8607273802835e-01
    for i in range(4 - BP):
        P = P[:-1]
    A = np.ones(N+1)
    A[0:BP] = P
    A[N-BP:-1] = np.flip(P, 0)
    H = np.zeros([N+1, N+1])
    np.fill_diagonal(H, A)
    Q = np.zeros([N+1, N+1])
    for i in range(4, N - 3):
        Q[i, i - 2:i + 3] = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
    
    print(Q)

def Energy():
    A = np.eye(N+1, N+1)
    nrg = np.zeros(Nt)
    A[0, 0] = A[N, N] = 1/2
    for i in range(Nt):
        u = np.array(solution[i, 0, :])
        h = np.array(solution[i, 1, :])
        nrg[i] = - h0 * g * (u @ (A @ (D @ h)) + h @ (A @ (D @ u)))
    plt.plot(np.linspace(0, Nt, Nt), nrg)
    plt.show()


"""configuration"""
g = 9.81
h0 = 10
a = -15
b = 15
N = 150
Nt = 1000
dx = (b - a) / N
D, H = matrix_factory()
x = a + dx * np.arange(0, N + 1)
dt = dx / (np.sqrt(g * h0) * 4)
ini_state = np.array([np.zeros(len(x)), np.exp((-0.5) * (x ** 2)) / np.sqrt(2 * np.pi)])
#ini_state = np.array([np.exp((-0.5) * (x ** 2)) / np.sqrt(2 * np.pi), np.exp((-0.5) * (x ** 2)) / np.sqrt(2 * np.pi)])
#ini_state = np.array([np.zeros(len(x)), np.zeros(len(x))])
#upg_matrix_factory(4)
solution = solveEquation(RK4D)
#Energy()
ani = animatewater(solution)
#ani.save('wall_water.gif', fps=25)
