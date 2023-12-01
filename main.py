import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import spdiags


class solve_typeI:
    def __init__(self, sp, N, Nt, t_ac, type_in):
        self.sp = sp
        self.N = N
        self.Nt = Nt
        self.t_ac = t_ac
        self.type_in = type_in
        self.dt = None
        self.x = None
        self.dx = None
        self.D = None
        self.sol = None
        self.H = None

    def RK4D(self, state, diff_method):
        k1 = diff_method(state)
        k2 = diff_method(state + k1 * self.dt / 2)
        k3 = diff_method(state + k2 * self.dt / 2)
        k4 = diff_method(state + k3 * self.dt)
        return state + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def matrix_diff(self, f):
        diff = (self.D @ f.transpose() / self.dx).transpose()
        diff = np.flip(diff, 0)
        diff[0, :] = -g * diff[0, :]
        diff[1, :] = -h0 * diff[1, :]
        diff[0, 0] = 0
        diff[0, -1] = 0
        return diff

    def matrix_factory(self):
        self.dx = (self.sp[1] - self.sp[0]) / self.N
        G = np.zeros([self.N + 1, self.N + 1])
        H = np.eye(self.N + 1, self.N + 1)
        H[:4, :4] = [[17 / 48, 0, 0, 0], [0, 59 / 48, 0, 0], [0, 0, 43 / 48, 0], [0, 0, 0, 49 / 48]]
        H[-4:, -4:] = [[49 / 48, 0, 0, 0], [0, 43 / 48, 0, 0], [0, 0, 59 / 48, 0], [0, 0, 0, 17 / 48]]
        G[:4, :6] = [[-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0, 0], [-59 / 96, 0, 59 / 96, 0, 0, 0],
                     [1 / 12, -59 / 96, 0, 59 / 96, -1 / 12, 0],
                     [1 / 32, 0, -59 / 96, 0, 2 / 3, -1 / 12]]
        for i in range(6):
            for j in range(6):
                G[self.N - i, self.N - j] = -G[i, j]
        for i in range(4, self.N - 3):
            G[i, i - 2:i + 3] = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
        self.D = np.linalg.inv(H) @ G
        self.H = H
        self.x = self.sp[0] + self.dx * np.arange(0, self.N + 1)

    def in_state(self):
        if self.type_in == 0:
            """ two-way wave """
            return np.array([np.zeros(self.x.size), np.exp((-0.5) * (self.x ** 2)) / np.sqrt(2 * np.pi)])
        elif self.type_in == 1:
            """ one-way wave """
            return np.array(
                [np.exp((-0.5) * (self.x ** 2)) / np.sqrt(2 * np.pi),
                 np.exp((-0.5) * (self.x ** 2)) / np.sqrt(2 * np.pi)])
        else:
            return np.array([np.zeros(self.x.size), np.zeros(self.x.size)])

    def solveEquation(self, method):
        global g, h0
        self.matrix_factory()
        self.dt = self.dx / (np.sqrt(g * h0) * self.t_ac)
        sol = np.zeros([self.Nt + 1, 2, self.N + 1])
        sol[0, :, :] = solve_typeI.in_state(self)
        for i in range(1, self.Nt + 1):
            sol[i, :, :] = method(sol[i - 1, :, :], self.matrix_diff)
        self.sol = sol


class solve_typeII(solve_typeI):
    def matrix_factory(self):
        BP = 4
        NES = 2
        xb = np.array([0, 0.68764546205559, 1.8022115125776, 2.8022115125776, 3.8022115125776])
        for i in range(5 - (NES + 1)):
            xb = xb[:-1]
        self.dx = (self.sp[1] - self.sp[0]) / (2 * xb[-1] + self.N - 2 * NES)
        self.x = self.sp[0] + self.dx * np.concatenate(
            [xb, np.linspace(xb[-1] + 1, (self.sp[1] - self.sp[0]) / self.dx - xb[-1] - 1, self.N + 1 - 2 * (NES + 1)).transpose(),
             (self.sp[1] - self.sp[0]) / self.dx - np.flip(xb, 0)])
        P = np.zeros(BP)
        P[0] = 2.1259737557798e-01
        P[1] = 1.0260290400758e+00
        P[2] = 1.0775123588954e+00
        P[3] = 9.8607273802835e-01
        for i in range(4 - BP):
            P = P[:-1]
        A = np.ones(self.N + 1)
        A[0:BP] = P
        A[self.N - BP + 1:self.N + 1] = np.flip(P, 0)
        H = np.zeros([self.N + 1, self.N + 1])
        np.fill_diagonal(H, self.dx * A)
        d = [[1 / 12], [-2 / 3], [0], [2 / 3], [-1 / 12]]
        G = np.zeros([self.N + 1, self.N + 1])
        for i in range(5):
            G += (spdiags(d[i] * (self.N + 1), i - 2, self.N + 1, self.N + 1).toarray())
        Bound = np.array([[-0.5, 6.5605279837843e-01, -1.9875859409017e-01, 4.2705795711740e-02, 0, 0],
                          [-6.5605279837843e-01, 0, 8.1236966439895e-01, -1.5631686602052e-01, 0, 0],
                          [1.9875859409017e-01, -8.1236966439895e-01, 0, 6.9694440364211e-01, -1 / 12, 0],
                          [-4.2705795711740e-02, 1.5631686602052e-01, -6.9694440364211e-01, 0, 2 / 3, -1 / 12]])
        for i in range(BP):
            for j in range(BP):
                G[i, j] = Bound[i, j]
                G[self.N - i, self.N - j] = -Bound[i, j]
        self.D = np.linalg.lstsq(H, G, rcond=None)[0]
        self.H = H

def animate_water(sol):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(sol.sp[0] - 1, sol.sp[1] + 1), ylim=(5, 17))
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    line, = ax.plot([], [], lw=2)
    time_template = 'time = %1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate1(i):
        thisx = sol.x
        thisy = sol.sol[i, 1, :] + h0
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i * sol.dt))
        return line, time_text

    frames = animation.FuncAnimation(fig, animate1, sol.Nt, interval=sol.dt * 1000, blit=True, repeat=False)
    plt.show()
    return frames


g = 9.81
h0 = 10
solution = solve_typeII([-15, 15], 150, 1000, 4, 1)
solution.solveEquation(solution.RK4D)
ani = animate_water(solution)
#ani.save('water.gif', fps=25)

"""
OUTDATED METHODS

def diff4_segment(f):
    diff_f = np.zeros([2, N + 1])
    for i in range(N + 1):
        diff_f[0, i] = -g * (
                f[1, (i - 2 + N) % N] / 12 - (2 / 3) * f[1, (i - 1 + N) % N] + (2 / 3) * f[1, (i + 1 + N) % N] -
                f[1, (i + 2 + N) % N] / 12) / dx
        diff_f[1, i] = -h0 * (
                f[0, (i - 2 + N) % N] / 12 - (2 / 3) * f[0, (i - 1 + N) % N] + (2 / 3) * f[0, (i + 1 + N) % N] -
                f[0, (i + 2 + N) % N] / 12) / dx
    return diff_f
"""
