import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


class runge_kutta:
    def __init__(
            self, 
            a:float, 
            b:float, 
            N:int,
            cond_init:list):
        """TODO

        Args:
            a (float): Borne inférieure.
            b (float): Borne supérieure.
            N (int): Nombre de points.
            cond_init (list): Conditions initiales du problème.
        """
        self.a = a
        self.b = b 
        self.N = N 
        self.h = (self.b-self.a)/self.N
        self.cond_init = cond_init
        self.tpoints = np.arange(self.a,self.b,self.h)


    def f(self, cond_ini:list)->np:
        """
        TODO
        Returns:
            np: Array contenant les nouvelles conditions initiales.
        """
        masse_soleil = 1.98840987e+30
        r = np.sqrt(cond_ini[0]**2 + cond_ini[2]**2+ cond_ini[4]**2)

        f2x =  -(masse_soleil * gravitational_constant) *cond_ini[0]/(r**3)
        f1x = cond_ini[1]

        f2y =  -(masse_soleil * gravitational_constant) *cond_ini[2]/(r**3)
        f1y = cond_ini[3]

        f2z =  -(masse_soleil * gravitational_constant) *cond_ini[4]/(r**3)
        f1z = cond_ini[5]

        return np.array([f1x, f2x, f1y, f2y, f1z, f2z])



    # def h_prime(self, h, goal, i, j):
    #     #optimus prime
    #     return h * (30 * h * goal /abs(i-j))**0.25
    


    def runge_kutta(self, h):
        k1 = h * self.f(self.cond_init)
        k2 = h * self.f(self.cond_init+0.5*k1)
        k3 = h * self.f(self.cond_init+0.5*k2)
        k4 = h * self.f(self.cond_init+k3)
        return k1, k2, k3, k4
        


    def simulation(self):
        """Méthode de Runge-Kutta."""
        self.xpoints = []
        self.vxpoints = []
        self.ypoints = []
        self.vypoints = []
        self.zpoints = []
        self.vzpoints = []

        delta = 2000
        # self.cond_ini

        for t in tqdm(self.tpoints):
            self.xpoints.append(self.cond_init[0])
            self.vxpoints.append(self.cond_init[1])

            self.ypoints.append(self.cond_init[2])
            self.vypoints.append(self.cond_init[3])
            
            self.zpoints.append(self.cond_init[4])
            self.vzpoints.append(self.cond_init[5])

            k1, k2, k3, k4 = self.runge_kutta(self.h)
            # temp_cond_in_normal = self.cond_init + (k1+2*k2+2*k3+k4)/6
            self.cond_init += (k1+2*k2+2*k3+k4)/6

            # k1, k2, k3, k4 = self.runge_kutta(2*self.h)
            # temp_cond_in_doublé = self.cond_init + (k1+2*k2+2*k3+k4)/6


            # # Prendre l'élément 0, 2, 4 pour les positions
            # h_prime = self.h_prime(self.h, 2000, temp_cond_in_normal, temp_cond_in_doublé)






        return self.tpoints, self.xpoints, self.vxpoints, self.ypoints, self.vypoints, self.zpoints, self.vzpoints



if __name__ == "__main__":
    t, x, vx, y, vy, z, vz = runge_kutta(0, 2e9, 100000, [1.521E11, 0, 0, 29290, 0, 0]).simulation()
    # t, x, vx, y, vy, z, vz = runge_kutta(0, 2e9, 100000, [4e12, 0, 0, 500, 0, 0]).simulation()


    # fig = plt.figure()
    # ax = plt.axes()
    # # ax = plt.axes(projection='3d')

    # # ax.plot(t, x, label="x")
    # # ax.plot(t, y, label="y")
    # ax.plot(x, y)
    # ax.scatter(0, 0, c="red", label="Soleil")
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    # # ax.set_zlabel("z [m]")
    # plt.legend()
    # plt.show()


    fig, ax = plt.subplots(figsize=(5, 5))

    terre, = ax.plot([], [], 'k.', markersize=15)
    ax.plot(0, 0, 'X', markersize=5, color="yellow")        #Soleil
    ax.set_aspect("equal")
    
    # dist_Ter_Hal = np.linalg.norm([x_Hal_interp[0]-x_Ter_interp[0], 
    #                                y_Hal_interp[0]-y_Ter_interp[0], 
    #                                z_Hal_interp[0]-z_Ter_interp[0]])
    time_text = plt.text(0,13, f"Temps : 0 années")
    
    # On run deux fois la simulation, une fois pour tracer les ellipses 
    # non perturbées, et l'autre pour la dépendance temporelle. Ici, on traces 
    # les orbites non perturbées
    # ax.plot(x_and_v_points_non_perturb[0][0] / astronomical_unit, 
    #         x_and_v_points_non_perturb[0][2] / astronomical_unit, 
    #         'g-', label="Halley")
    # ax.plot(x_and_v_points_non_perturb[1][0] / astronomical_unit, 
    #         x_and_v_points_non_perturb[1][2] / astronomical_unit, 
    #         'r-', label="Jupiter")
    # ax.plot(x_and_v_points_non_perturb[2][0] / astronomical_unit, 
    #         x_and_v_points_non_perturb[2][2] / astronomical_unit, 
    #         'b-', label="Saturne")
    ax.plot(x, 
            y, 
            'k-', label="Terre")
    
    plt.legend()
    
    def animate(i):
        # halley.set_data(x_Hal_interp[i], y_Hal_interp[i])
        # jupiter.set_data(x_Jup_interp[i], y_Jup_interp[i])
        # saturne.set_data(x_Sat_interp[i], y_Sat_interp[i])
        terre.set_data(x[i], y[i])
        # dist_Ter_Hal = np.linalg.norm([x_Hal_interp[i]-x_Ter_interp[i], 
        #                            y_Hal_interp[i]-y_Ter_interp[i], 
        #                            z_Hal_interp[i]-z_Ter_interp[i]])
        time_text.set_text(f"Temps : {round(t[i]/(365.25*24*3600),3)} années")
        return terre
    
    anim = FuncAnimation(fig, animate, frames=10000, interval=40, repeat=False)
    plt.show()