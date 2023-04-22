import numpy as np
from scipy.constants import gravitational_constant
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import math


class runge_kutta:
    def __init__(
            self, 
            a:float, 
            b:float, 
            N:int,
            cond_init:list,
            delta:int):
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
        self.og_h = self.h
        self.cond_init = cond_init
        self.delta = delta
        # self.tpoints = np.arange(self.a,self.b,self.h)
        # self.tpoints = [0]


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





    def runge_kutta(self, h, initial_conditions):
        k1 = h * self.f(initial_conditions)
        k2 = h * self.f(initial_conditions+0.5*k1)
        k3 = h * self.f(initial_conditions+0.5*k2)
        k4 = h * self.f(initial_conditions+k3)
        return k1, k2, k3, k4
        


    def simulation(self):
        """Méthode de Runge-Kutta."""
        xpoints = []
        vxpoints = []
        ypoints = []
        vypoints = []
        zpoints = []
        vzpoints = []
        h_values = []
        tpoints = [0]

        def rho(i, j):
            eps_x = abs(i[0]-j[0])/30
            eps_y = abs(i[2]-j[2])/30
            eps_z = abs(i[4]-j[4])/30
            return self.h * self.delta / np.sqrt(eps_x**2 + eps_y**2 + eps_z**2)
        
        def h_prime(rho, h):
            return h * rho**0.25

        last_percent_completed = 0
        while tpoints[-1] < self.b:
            if math.floor(tpoints[-1]/self.b * 100) > last_percent_completed:
                last_percent_completed = math.floor(tpoints[-1]/self.b * 100)
                print(f"{last_percent_completed} % complété")
                
            xpoints.append(self.cond_init[0])
            vxpoints.append(self.cond_init[1])

            ypoints.append(self.cond_init[2])
            vypoints.append(self.cond_init[3])
            
            zpoints.append(self.cond_init[4])
            vzpoints.append(self.cond_init[5])
            
            h_values.append(self.h)

            # h + h
            k1_h, k2_h, k3_h, k4_h = self.runge_kutta(self.h, self.cond_init)
            cond_init_h = self.cond_init + (k1_h+2*k2_h+2*k3_h+k4_h)/6

            k1_h_p_h, k2_h_p_h, k3_h_p_h, k4_h_p_h = self.runge_kutta(self.h, cond_init_h)            
            cond_init_h_p_h = cond_init_h + (k1_h_p_h+2*k2_h_p_h+2*k3_h_p_h+k4_h_p_h)/6


            # 2h
            k1_2h, k2_2h, k3_2h, k4_2h = self.runge_kutta(2 * self.h, self.cond_init)
            cond_init_2h = self.cond_init + (k1_2h+2*k2_2h+2*k3_2h+k4_2h)/6

            # Calcul rho
            p = rho(cond_init_h_p_h, cond_init_2h)

            h_pri = h_prime(p, self.h)

            # TODO : MODIFIER LES LIGNES 124 À 140 DANS LES AUTRES FICHIERS
            if p >= 1:
                self.cond_init = cond_init_h_p_h
                new_time = tpoints[-1] + 2* self.h
                
                self.h = min(2* self.h, h_pri)


            else:
                k1, k2, k3, k4 = self.runge_kutta(h_pri, self.cond_init)
                self.cond_init += (k1+2*k2+2*k3+k4)/6
                new_time = tpoints[-1] + h_pri
                self.h = h_pri

            
            
            tpoints.append(new_time)

        # TODO : MODIFIER LA LIGNE SUIVANTE POUR PRENDRE LES BONS TPOINTS
        return tpoints[:-1], xpoints, vxpoints, ypoints, vypoints, zpoints, vzpoints, h_values



if __name__ == "__main__":
    # t, x, vx, y, vy, z, vz, h = runge_kutta(0, 2e9, 1E05, [4e12, 0, 0, 500, 0, 0], 1E-06).simulation()
    t, x, vx, y, vy, z, vz, h = runge_kutta(0, 2e9, 1E05, [1.521E11, 0, 0, 29290, 0, 0], 1E-08).simulation()

    print(len(t))
    print(len(x))

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
    
