import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import spdiags


class Segment_typeI:
    def __init__(self, posX, N, bordX, t_ac=None, type_in=None):
        self.posX = posX
        self.N = N
        self.bordX = bordX
        self.t_ac = t_ac
        self.dt = None
        self.x = None
        self.dx = None
        self.D = None
        self.H = None
        self.X_in = self.in_state(type_in)
        self.sol = np.zeros([1, 2, N + 1])
        self.sol[0] = self.X_in

    def in_state(self, type_in):
        self.matrix_factory()
        if self.t_ac is not None:
            global g, h0
            self.dt = self.dx / (np.sqrt(g * h0) * self.t_ac)
        if type_in == 0:
            """ two-way wave """
            return np.array([np.zeros(self.x.size),
                             np.exp((-0.5) * ((self.x - 0.5 * (self.posX[0] + self.posX[1])) ** 2)) / np.sqrt(
                                 2 * np.pi)])
        elif type_in == 1:
            """ one-way wave """
            return np.array(
                [np.exp((-0.5) * ((self.x - 0.5 * (self.posX[0] + self.posX[1])) ** 2)) / np.sqrt(2 * np.pi),
                 np.exp((-0.5) * ((self.x - 0.5 * (self.posX[0] + self.posX[1])) ** 2)) / np.sqrt(2 * np.pi)])
        else:
            return np.array([np.zeros(self.x.size), np.zeros(self.x.size)])

    def matrix_factory(self):
        self.dx = (self.posX[1] - self.posX[0]) / self.N
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
        self.D = (np.linalg.inv(H) @ G) / self.dx
        self.H = H
        self.x = self.posX[0] + self.dx * np.arange(0, self.N + 1)

    def segment_diff(self, f):
        diff = (self.D @ f.transpose()).transpose()
        diff = np.flip(diff, 0)
        diff[0, :] = -g * diff[0, :]
        diff[1, :] = -h0 * diff[1, :]
        if self.bordX[0] == 0:
            diff[0, 0] = 0
        if self.bordX[1] == 0:
            diff[0, -1] = 0
        return diff


class Segment_typeII(Segment_typeI):
    def matrix_factory(self):
        BP = 4
        NES = 2
        xb = np.array([0, 0.68764546205559, 1.8022115125776, 2.8022115125776, 3.8022115125776])
        for i in range(5 - (NES + 1)):
            xb = xb[:-1]
        self.dx = (self.posX[1] - self.posX[0]) / (2 * xb[-1] + self.N - 2 * NES)
        self.x = self.posX[0] + self.dx * np.concatenate(
            [xb, np.linspace(xb[-1] + 1, (self.posX[1] - self.posX[0]) / self.dx - xb[-1] - 1,
                             self.N + 1 - 2 * (NES + 1)).transpose(),
             (self.posX[1] - self.posX[0]) / self.dx - np.flip(xb, 0)])
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


def solving():
    for step in range(10000):
        storage = np.zeros([4, len(system), 2, 151])
        for i in system:
            storage[0, system.index(i)] = i.segment_diff(i.sol[-1])
        border(storage[0])
        for i in system:
            storage[1, system.index(i)] = i.segment_diff(i.sol[-1] + i.dt / 2 * storage[0, system.index(i)])
        border(storage[1])
        for i in system:
            storage[2, system.index(i)] = i.segment_diff(i.sol[-1] + i.dt / 2 * storage[1, system.index(i)])
        border(storage[2])
        for i in system:
            storage[3, system.index(i)] = i.segment_diff(i.sol[-1] + i.dt * storage[2, system.index(i)])
        border(storage[3])
        for i in system:
            m = system.index(i)
            i.sol = np.append(i.sol, [[np.zeros(i.N + 1), np.zeros(i.N + 1)]], 0)
            i.sol[-1] = i.sol[-2] + i.dt * (storage[0,m] + 2 * storage[1,m] + 2 * storage[2,m] + storage[3,m]) / 6



def border(storage):
    for check in system:
        id_check = system.index(check)
        if check.bordX[0] != 0 and id_check < check.bordX[0]:
            storage[id_check, :, 0] = (storage[id_check, :, 0] + storage[check.bordX[0] - 1, :, -1]) / 2
            storage[check.bordX[0] - 1, :, -1] = storage[id_check, :, 0]
        if check.bordX[1] != 0 and id_check < check.bordX[0]:
            storage[id_check, :, -1] = (storage[id_check, :, -1] + storage[check.bordX[1] - 1, :, 0]) / 2
            storage[check.bordX[1] - 1, :, 0] = storage[id_check, :, -1]


def animate_water(element):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(element.posX[0] - 1, element.posX[1] + 1), ylim=(5, 17))
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    line, = ax.plot([], [], lw=2)
    time_template = 'time = %1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate1(i):
        thisx = element.x
        thisy = element.sol[i, 1, :] + h0
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i * element.dt))
        return line, time_text

    frames = animation.FuncAnimation(fig, animate1, element.sol.shape[0], interval=element.dt * element.sol.shape[0],
                                     blit=True, repeat=False)
    return frames





g = 9.81
h0 = 10
system = [Segment_typeII([-20, 0], 150, [2, 2], 2, 3), Segment_typeII([0, 20], 150, [1, 1], 2, 1)]
solving()

ani1 = animate_water(system[0])
ani2 = animate_water(system[1])
# for j in system:
# ani = animate_water(j)
plt.show()
# ani = animate_water(system)
# ani1.save('water1.gif', fps=25)
# ani2.save('water2.gif', fps=25)
"""
def matrix_diff(segment):
    diff = (self.D @ f.transpose()).transpose()
    diff = np.flip(diff, 0)
    diff[0, :] = -g * diff[0, :]
    diff[1, :] = -h0 * diff[1, :]
    if self.bordX[0] == 0:
        diff[0, 0] = 0
    if self.bordX[1] == 0:
        diff[0, -1] = 0
    return diff


def RK4D(self):
    k1 = self.matrix_diff(self.sol[-1])
    k2 = self.matrix_diff(self.sol[-1] + k1 * self.dt / 2)
    k3 = self.matrix_diff(self.sol[-1] + k2 * self.dt / 2)
    k4 = self.matrix_diff(self.sol[-1] + k3 * self.dt)
    return self.sol[-1] + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def next_step(self):
    m = self.RK4D()
    self.sol = np.append(self.sol, [[np.zeros(self.N + 1), np.zeros(self.N + 1)]], 0)
    self.sol[-1] = m
"""