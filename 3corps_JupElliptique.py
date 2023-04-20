import numpy as np
from scipy.constants import gravitational_constant
import matplotlib.pyplot as plt
import time
import math
# from tqdm import tqdm
# from tqdm.auto import tqdm

class runge_kutta:
    def __init__(
            self, 
            a:float, 
            b:float,
            N:int,
            cond_init:np,
            masses:np,
            delta:int):
        """TODO

        Args:
            a (float): Borne inférieure.
            b (float): Borne supérieure.
            N : nombre estimé d'itérations (?)
            cond_init (np): Conditions initiales du problème.
                Colonnes : x, vx, y, vy, z, vz
                Rangées : Halley, Jupiter
            delta (int) : précision visée
        """
        self.a = a
        self.b = b
        self.h = (self.b-self.a)/N
        self.etat = cond_init
        self.delta = delta
        self.masses = masses


    def f(self, cond_ini:list)->np:
        """
        TODO
        Returns:
            np: Array contenant les nouvelles conditions initiales.
        """
        masse_soleil = 1.98840987e+30
        masse_Hal = self.masses[0][0]
        masse_Jup = self.masses[0][1]
        r_Hal_Sol = np.sqrt(cond_ini[0][0]**2 + cond_ini[0][2]**2+ cond_ini[0][4]**2)
        r_Jup_Sol = np.sqrt(cond_ini[1][0]**2 + cond_ini[1][2]**2+ cond_ini[1][4]**2)
        r_Jup_Hal = np.sqrt((cond_ini[1][0] - cond_ini[0][0])**2 + (cond_ini[1][2] - cond_ini[0][2])**2+ (cond_ini[1][4] - cond_ini[0][4])**2)

        f2x_Hal =  -(masse_soleil * gravitational_constant) *cond_ini[0][0]/(r_Hal_Sol**3) -(masse_Jup * gravitational_constant) *(cond_ini[0][0] - cond_ini[1][0])/(r_Jup_Hal**3)
        f1x_Hal = cond_ini[0][1]

        f2y_Hal =  -(masse_soleil * gravitational_constant) *cond_ini[0][2]/(r_Hal_Sol**3) -(masse_Jup * gravitational_constant) *(cond_ini[0][2] - cond_ini[1][2])/(r_Jup_Hal**3)
        f1y_Hal = cond_ini[0][3]

        f2z_Hal =  -(masse_soleil * gravitational_constant) *cond_ini[0][4]/(r_Hal_Sol**3) -(masse_Jup * gravitational_constant) *(cond_ini[0][4] - cond_ini[1][4])/(r_Jup_Hal**3)
        f1z_Hal = cond_ini[0][5]
        
        f2x_Jup =  -(masse_soleil * gravitational_constant) *cond_ini[1][0]/(r_Jup_Sol**3)
        f1x_Jup = cond_ini[1][1]

        f2y_Jup =  -(masse_soleil * gravitational_constant) *cond_ini[1][2]/(r_Jup_Sol**3)
        f1y_Jup = cond_ini[1][3]

        f2z_Jup =  -(masse_soleil * gravitational_constant) *cond_ini[1][4]/(r_Jup_Sol**3)
        f1z_Jup = cond_ini[1][5]

        return np.array([
            [f1x_Hal, f2x_Hal, f1y_Hal, f2y_Hal, f1z_Hal, f2z_Hal],
            [f1x_Jup, f2x_Jup, f1y_Jup, f2y_Jup, f1z_Jup, f2z_Jup]
            ])





    def runge_kutta(self, h, initial_conditions):
        k1 = h * self.f(initial_conditions)
        k2 = h * self.f(initial_conditions+0.5*k1)
        k3 = h * self.f(initial_conditions+0.5*k2)
        k4 = h * self.f(initial_conditions+k3)
        return k1, k2, k3, k4
        


    def simulation(self):
        """Méthode de Runge-Kutta."""
        xpoints_Hal = []
        vxpoints_Hal = []
        ypoints_Hal = []
        vypoints_Hal = []
        zpoints_Hal = []
        vzpoints_Hal = []
        
        xpoints_Jup = []
        vxpoints_Jup = []
        ypoints_Jup = []
        vypoints_Jup = []
        zpoints_Jup = []
        vzpoints_Jup = []
        
        h_values = []
        tpoints = [0]

        def rho(etat_h_p_h, etat_2h):
            eps_Hal = [abs(etat_h_p_h[0][0]-etat_2h[0][0])/30, 
                       abs(etat_h_p_h[0][2]-etat_2h[0][2])/30, 
                       abs(etat_h_p_h[0][4]-etat_2h[0][4])/30]
            eps_Jup = [abs(etat_h_p_h[1][0]-etat_2h[1][0])/30, 
                       abs(etat_h_p_h[1][2]-etat_2h[1][2])/30, 
                       abs(etat_h_p_h[1][4]-etat_2h[1][4])/30]
            eps = np.array([eps_Hal + eps_Jup])
            return self.h * self.delta / np.linalg.norm(eps)
        
        def h_prime(rho, h):
            return h * rho**0.25

        last_percent_completed = 0
        while tpoints[-1] < self.b:
            if math.floor(tpoints[-1]/self.b * 100) > last_percent_completed:
                last_percent_completed = math.floor(tpoints[-1]/self.b * 100)
                print(f"{last_percent_completed} % complété")
            
            xpoints_Hal.append(self.etat[0][0])
            vxpoints_Hal.append(self.etat[0][1])

            ypoints_Hal.append(self.etat[0][2])
            vypoints_Hal.append(self.etat[0][3])
            
            zpoints_Hal.append(self.etat[0][4])
            vzpoints_Hal.append(self.etat[0][5])
            
            xpoints_Jup.append(self.etat[1][0])
            vxpoints_Jup.append(self.etat[1][1])

            ypoints_Jup.append(self.etat[1][2])
            vypoints_Jup.append(self.etat[1][3])
            
            zpoints_Jup.append(self.etat[1][4])
            vzpoints_Jup.append(self.etat[1][5])
            
            h_values.append(self.h)

            # On calcule l'état à t + 2h en faisant deux incréments de h (h + h)
            k1_h, k2_h, k3_h, k4_h = self.runge_kutta(self.h, self.etat)
            etat_h = self.etat + (k1_h+2*k2_h+2*k3_h+k4_h)/6

            k1_h_p_h, k2_h_p_h, k3_h_p_h, k4_h_p_h = self.runge_kutta(self.h, etat_h)            
            etat_h_p_h = etat_h + (k1_h_p_h+2*k2_h_p_h+2*k3_h_p_h+k4_h_p_h)/6

            # On calcule l'état à t + 2h en faisant un incrément de 2h (2h)
            k1_2h, k2_2h, k3_2h, k4_2h = self.runge_kutta(2 * self.h, self.etat)
            etat_2h = self.etat + (k1_2h+2*k2_2h+2*k3_2h+k4_2h)/6

            # Calcul rho
            p = rho(etat_h_p_h, etat_2h)
            h_pri = h_prime(p, self.h)

            if p >= 1:
                self.etat = etat_h_p_h
                self.h = min(2* self.h, h_pri)

            else:
                k1, k2, k3, k4 = self.runge_kutta(h_pri, self.etat)
                self.etat += (k1+2*k2+2*k3+k4)/6
                self.h = h_pri

            
            new_time = tpoints[-1] + self.h
            tpoints.append(new_time)

        x_and_v_points = np.array([
            [xpoints_Hal, vxpoints_Hal, ypoints_Hal, vypoints_Hal, zpoints_Hal, vzpoints_Hal],
            [xpoints_Jup, vxpoints_Jup, ypoints_Jup, vypoints_Jup, zpoints_Jup, vzpoints_Jup]
            ])
        
        return tpoints[1:], x_and_v_points, h_values



if __name__ == "__main__":
    cond_init_Hal = [4E12, 0, 0, 500, 0, 0]
    cond_init_Jup = [7.4051E11, 0, 0, 13000, 0, 0]
    cond_init = np.array([cond_init_Hal, cond_init_Jup])
    masses = np.array([[2.2E14, 1.898E27]])
    
    t1 = time.time()
    t, x_and_v_points, h = runge_kutta(0, 2e9, 1E05, cond_init, masses, 1E-07).simulation()
    t2 = time.time()
    
    print(f"La simulation a pris {t2-t1} secondes à réaliser")
    # plt.plot(t, h)
    # plt.show()


    x_Hal = x_and_v_points[0][0]
    y_Hal = x_and_v_points[0][2]
    x_Jup = x_and_v_points[1][0]
    y_Jup = x_and_v_points[1][2]
    
    # plt.scatter(x_Hal,y_Hal)
    plt.plot(x_Hal, y_Hal, label="Halley")
    plt.scatter(x_Jup, y_Jup, label="Jupiter")
    plt.legend()
    plt.show()
    
    
    

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
