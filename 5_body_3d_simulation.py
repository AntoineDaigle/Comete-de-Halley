
import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit
import matplotlib.pyplot as plt
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime, timedelta
import pandas as pd

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
        self.cond_init = cond_init
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
        masse_Sat = self.masses[0][2]
        masse_Ter = self.masses[0][3]
        r_Hal_Sol = np.sqrt(cond_ini[0][0]**2 + cond_ini[0][2]**2+ cond_ini[0][4]**2)
        r_Jup_Sol = np.sqrt(cond_ini[1][0]**2 + cond_ini[1][2]**2+ cond_ini[1][4]**2)
        r_Jup_Hal = np.sqrt((cond_ini[1][0] - cond_ini[0][0])**2 + (cond_ini[1][2] - cond_ini[0][2])**2+ (cond_ini[1][4] - cond_ini[0][4])**2)
        r_Sat_Sol = np.sqrt(cond_ini[2][0]**2 + cond_ini[2][2]**2+ cond_ini[2][4]**2)
        r_Sat_Hal = np.sqrt((cond_ini[2][0] - cond_ini[0][0])**2 + (cond_ini[2][2] - cond_ini[0][2])**2+ (cond_ini[2][4] - cond_ini[0][4])**2)
        r_Ter_Sol = np.sqrt(cond_ini[3][0]**2 + cond_ini[3][2]**2+ cond_ini[3][4]**2)
        r_Ter_Hal = np.sqrt((cond_ini[3][0] - cond_ini[0][0])**2 + (cond_ini[3][2] - cond_ini[0][2])**2+ (cond_ini[3][4] - cond_ini[0][4])**2)
        
        f2x_Hal =  -(masse_soleil * gravitational_constant) *cond_ini[0][0]/(r_Hal_Sol**3) - \
            (masse_Jup * gravitational_constant) *(cond_ini[0][0] - cond_ini[1][0])/(r_Jup_Hal**3) - \
            (masse_Sat * gravitational_constant) *(cond_ini[0][0] - cond_ini[2][0])/(r_Sat_Hal**3)
        f1x_Hal = cond_ini[0][1]

        f2y_Hal =  -(masse_soleil * gravitational_constant) *cond_ini[0][2]/(r_Hal_Sol**3) - \
            (masse_Jup * gravitational_constant) *(cond_ini[0][2] - cond_ini[1][2])/(r_Jup_Hal**3) - \
            (masse_Sat * gravitational_constant) *(cond_ini[0][2] - cond_ini[2][2])/(r_Sat_Hal**3)
        f1y_Hal = cond_ini[0][3]

        f2z_Hal =  -(masse_soleil * gravitational_constant) *cond_ini[0][4]/(r_Hal_Sol**3) - \
            (masse_Jup * gravitational_constant) *(cond_ini[0][4] - cond_ini[1][4])/(r_Jup_Hal**3) - \
            (masse_Sat * gravitational_constant) *(cond_ini[0][4] - cond_ini[2][4])/(r_Sat_Hal**3)
        f1z_Hal = cond_ini[0][5]
        
        f2x_Jup =  -(masse_soleil * gravitational_constant) *cond_ini[1][0]/(r_Jup_Sol**3)
        f1x_Jup = cond_ini[1][1]

        f2y_Jup =  -(masse_soleil * gravitational_constant) *cond_ini[1][2]/(r_Jup_Sol**3)
        f1y_Jup = cond_ini[1][3]

        f2z_Jup =  -(masse_soleil * gravitational_constant) *cond_ini[1][4]/(r_Jup_Sol**3)
        f1z_Jup = cond_ini[1][5]
        
        f2x_Sat =  -(masse_soleil * gravitational_constant) *cond_ini[2][0]/(r_Sat_Sol**3)
        f1x_Sat = cond_ini[2][1]

        f2y_Sat =  -(masse_soleil * gravitational_constant) *cond_ini[2][2]/(r_Sat_Sol**3)
        f1y_Sat = cond_ini[2][3]

        f2z_Sat =  -(masse_soleil * gravitational_constant) *cond_ini[2][4]/(r_Sat_Sol**3)
        f1z_Sat = cond_ini[2][5]
        
        f2x_Ter =  -(masse_soleil * gravitational_constant) *cond_ini[3][0]/(r_Ter_Sol**3)
        f1x_Ter = cond_ini[3][1]

        f2y_Ter =  -(masse_soleil * gravitational_constant) *cond_ini[3][2]/(r_Ter_Sol**3)
        f1y_Ter = cond_ini[3][3]

        f2z_Ter =  -(masse_soleil * gravitational_constant) *cond_ini[3][4]/(r_Ter_Sol**3)
        f1z_Ter = cond_ini[3][5]

        return np.array([
            [f1x_Hal, f2x_Hal, f1y_Hal, f2y_Hal, f1z_Hal, f2z_Hal],
            [f1x_Jup, f2x_Jup, f1y_Jup, f2y_Jup, f1z_Jup, f2z_Jup],
            [f1x_Sat, f2x_Sat, f1y_Sat, f2y_Sat, f1z_Sat, f2z_Sat],
            [f1x_Ter, f2x_Ter, f1y_Ter, f2y_Ter, f1z_Ter, f2z_Ter]
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
        
        xpoints_Sat = []
        vxpoints_Sat = []
        ypoints_Sat = []
        vypoints_Sat = []
        zpoints_Sat = []
        vzpoints_Sat = []

        xpoints_Ter = []
        vxpoints_Ter = []
        ypoints_Ter = []
        vypoints_Ter = []
        zpoints_Ter = []
        vzpoints_Ter = []
        
        h_values = []
        tpoints = [0]

        def rho(etat_h_p_h, etat_2h):
            eps_Hal = [abs(etat_h_p_h[0][0]-etat_2h[0][0])/30, 
                       abs(etat_h_p_h[0][2]-etat_2h[0][2])/30, 
                       abs(etat_h_p_h[0][4]-etat_2h[0][4])/30]
            eps_Jup = [abs(etat_h_p_h[1][0]-etat_2h[1][0])/30, 
                       abs(etat_h_p_h[1][2]-etat_2h[1][2])/30, 
                       abs(etat_h_p_h[1][4]-etat_2h[1][4])/30]
            eps_Sat = [abs(etat_h_p_h[2][0]-etat_2h[2][0])/30, 
                       abs(etat_h_p_h[2][2]-etat_2h[2][2])/30, 
                       abs(etat_h_p_h[2][4]-etat_2h[2][4])/30]
            eps_Ter = [abs(etat_h_p_h[3][0]-etat_2h[3][0])/30, 
                       abs(etat_h_p_h[3][2]-etat_2h[3][2])/30, 
                       abs(etat_h_p_h[3][4]-etat_2h[3][4])/30]
            eps = np.array([eps_Hal + eps_Jup + eps_Sat + eps_Ter])
            return self.h * self.delta / np.linalg.norm(eps)
        
        def h_prime(rho, h):
            return h * rho**0.25

        last_percent_completed = 0
        while tpoints[-1] < self.b:
            if math.floor(tpoints[-1]/self.b * 100) > last_percent_completed:
                last_percent_completed = math.floor(tpoints[-1]/self.b * 100)
                print(f"{last_percent_completed} % complété")


            xpoints_Hal.append(self.cond_init[0][0])
            vxpoints_Hal.append(self.cond_init[0][1])

            ypoints_Hal.append(self.cond_init[0][2])
            vypoints_Hal.append(self.cond_init[0][3])
            
            zpoints_Hal.append(self.cond_init[0][4])
            vzpoints_Hal.append(self.cond_init[0][5])
            
            xpoints_Jup.append(self.cond_init[1][0])
            vxpoints_Jup.append(self.cond_init[1][1])

            ypoints_Jup.append(self.cond_init[1][2])
            vypoints_Jup.append(self.cond_init[1][3])
            
            zpoints_Jup.append(self.cond_init[1][4])
            vzpoints_Jup.append(self.cond_init[1][5])
            
            xpoints_Sat.append(self.cond_init[2][0])
            vxpoints_Sat.append(self.cond_init[2][1])

            ypoints_Sat.append(self.cond_init[2][2])
            vypoints_Sat.append(self.cond_init[2][3])
            
            zpoints_Sat.append(self.cond_init[2][4])
            vzpoints_Sat.append(self.cond_init[2][5])
            
            xpoints_Ter.append(self.cond_init[3][0])
            vxpoints_Ter.append(self.cond_init[3][1])

            ypoints_Ter.append(self.cond_init[3][2])
            vypoints_Ter.append(self.cond_init[3][3])
            
            zpoints_Ter.append(self.cond_init[3][4])
            vzpoints_Ter.append(self.cond_init[3][5])
            
            h_values.append(self.h)

            # On calcule l'état à t + 2h en faisant deux incréments de h (h + h)
            k1_h, k2_h, k3_h, k4_h = self.runge_kutta(self.h, self.cond_init)
            cond_init_h = self.cond_init + (k1_h+2*k2_h+2*k3_h+k4_h)/6

            k1_h_p_h, k2_h_p_h, k3_h_p_h, k4_h_p_h = self.runge_kutta(self.h, cond_init_h)            
            cond_init_h_p_h = cond_init_h + (k1_h_p_h+2*k2_h_p_h+2*k3_h_p_h+k4_h_p_h)/6

            # On calcule l'état à t + 2h en faisant un incrément de 2h (2h)
            k1_2h, k2_2h, k3_2h, k4_2h = self.runge_kutta(2 * self.h, self.cond_init)
            cond_init_2h = self.cond_init + (k1_2h+2*k2_2h+2*k3_2h+k4_2h)/6

            # Calcul rho
            p = rho(cond_init_h_p_h, cond_init_2h)
            h_pri = h_prime(p, self.h)

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


        d = {"time":tpoints[:-1],
             "h_values":h_values,
             "x_halley":xpoints_Hal,
             "vx_halley":vxpoints_Hal,
             "y_halley":ypoints_Hal,
             "vy_halley":vypoints_Hal,
             "z_halley":zpoints_Hal,
             "vz_halley":vzpoints_Hal,
             "x_jupiter":xpoints_Jup,
             "vx_jupiter":vxpoints_Jup,
             "y_jupiter":ypoints_Jup,
             "vy_jupiter":vypoints_Jup,
             "z_jupiter":zpoints_Jup,
             "vz_jupiter":vzpoints_Jup,
             "x_saturne":xpoints_Sat,
             "vx_saturne":vxpoints_Sat,
             "y_saturne":ypoints_Sat,
             "vy_saturne":vypoints_Sat,
             "z_saturne":zpoints_Sat,
             "vz_saturne":vzpoints_Sat,
             "x_terre":xpoints_Ter,
             "vx_terre":vxpoints_Ter,
             "y_terre":ypoints_Ter,
             "vy_terre":vypoints_Ter,
             "z_terre":zpoints_Ter,
             "vz_terre":vzpoints_Ter}
        

        simulation_res = pd.DataFrame(data=d)
        

        return simulation_res
    



if __name__ == "__main__":

    # Condition initiale des corps
    cond_init_Hal = [4E12, 0, 0, 300, 0, 400]
    cond_init_Jup = [7.4051E11, 0, 0, 13000, 0, 0]
    cond_init_Sat = [1.434E12, 0, 0, 9690, 0, 0]
    cond_init_Ter = [1.521E11, 0, 0, 29290, 0, 0]

    cond_init = np.array([cond_init_Hal, 
                          cond_init_Jup, 
                          cond_init_Sat, 
                          cond_init_Ter])
    masses = np.array([[2.2E14, 1.898E27, 5.683E26, 5.972E24]])
    masses_non_perturb = np.array([[0,0,0,0]])

    # Setup simulation
    a = 0
    b = 1e9
    N = 1e5
    delta = 1e-7

    d = {"a": [a],
         "b": [b],
         "N": [N],
         "delta":[delta],}
    condition_initiale_simulation = pd.DataFrame(data=d)
    condition_initiale_simulation.to_csv("Condition_initiale_simulation.csv", sep=",")

    # Simulation orbite perturbé
    t1_p = time.time()
    pert_data = runge_kutta(a, b, N, cond_init, masses, delta).simulation()
    t2_p = time.time()

    # Simulation orbite non perturbé
    t1_np = time.time()
    non_pert_data = runge_kutta(a, b, N, cond_init, masses_non_perturb, delta).simulation()
    t2_np = time.time()

    pert_data.to_csv("pert_data.csv", sep=",")
    non_pert_data.to_csv("non_pert_data.csv", sep=",")

    print(f"\nData saved\nSimulation perturbé: {round(t2_p-t1_p,2)} s\nSimulation non-perturbé: {round(t2_np-t1_np,2)}")
