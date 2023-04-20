import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit
import matplotlib.pyplot as plt
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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
        masse_Sat = self.masses[0][2]
        r_Hal_Sol = np.sqrt(cond_ini[0][0]**2 + cond_ini[0][2]**2+ cond_ini[0][4]**2)
        r_Jup_Sol = np.sqrt(cond_ini[1][0]**2 + cond_ini[1][2]**2+ cond_ini[1][4]**2)
        r_Jup_Hal = np.sqrt((cond_ini[1][0] - cond_ini[0][0])**2 + (cond_ini[1][2] - cond_ini[0][2])**2+ (cond_ini[1][4] - cond_ini[0][4])**2)
        r_Sat_Sol = np.sqrt(cond_ini[2][0]**2 + cond_ini[2][2]**2+ cond_ini[2][4]**2)
        r_Sat_Hal = np.sqrt((cond_ini[2][0] - cond_ini[0][0])**2 + (cond_ini[2][2] - cond_ini[0][2])**2+ (cond_ini[2][4] - cond_ini[0][4])**2)
        
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

        return np.array([
            [f1x_Hal, f2x_Hal, f1y_Hal, f2y_Hal, f1z_Hal, f2z_Hal],
            [f1x_Jup, f2x_Jup, f1y_Jup, f2y_Jup, f1z_Jup, f2z_Jup],
            [f1x_Sat, f2x_Sat, f1y_Sat, f2y_Sat, f1z_Sat, f2z_Sat]
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
            eps = np.array([eps_Hal + eps_Jup + eps_Sat])
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
            
            xpoints_Sat.append(self.etat[2][0])
            vxpoints_Sat.append(self.etat[2][1])

            ypoints_Sat.append(self.etat[2][2])
            vypoints_Sat.append(self.etat[2][3])
            
            zpoints_Sat.append(self.etat[2][4])
            vzpoints_Sat.append(self.etat[2][5])
            
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
            [xpoints_Hal, vxpoints_Hal, ypoints_Hal, vypoints_Hal, zpoints_Hal, 
             vzpoints_Hal],
            [xpoints_Jup, vxpoints_Jup, ypoints_Jup, vypoints_Jup, zpoints_Jup, 
             vzpoints_Jup],
            [xpoints_Sat, vxpoints_Sat, ypoints_Sat, vypoints_Sat, zpoints_Sat, 
             vzpoints_Sat]
            ])
        
        return tpoints[1:], x_and_v_points, h_values



if __name__ == "__main__":
    cond_init_Hal = [4E12, 0, 0, 300, 0, 400]
    cond_init_Jup = [7.4051E11, 0, 0, 13000, 0, 0]
    cond_init_Sat = [1.434E12, 0, 0, 9690, 0, 0]
    cond_init = np.array([cond_init_Hal, cond_init_Jup, cond_init_Sat])
    masses = np.array([[2.2E14, 1.898E27, 5.683E26]])
    masses_non_perturb = np.array([[0,0,0]])
    
    t1 = time.time()
    t, x_and_v_points, h = runge_kutta(0, 1e9, 1E05, cond_init, masses, 
                                       1E-07).simulation()
    t2 = time.time()
    
    print(f"La simulation a pris {t2-t1} secondes à réaliser")
    
    
    print("Simulation des corps non perturbés")
    t1 = time.time()
    t_non_perturb, x_and_v_points_non_perturb, h_non_perturb = runge_kutta(0, 
                1e09, 1E05, cond_init, masses_non_perturb, 1E-07).simulation()
    t2 = time.time()
    
    print(f"La simulation a pris {t2-t1} secondes à réaliser")


    x_Hal = x_and_v_points[0][0] / astronomical_unit 
    y_Hal = x_and_v_points[0][2] / astronomical_unit
    z_Hal = x_and_v_points[0][4] / astronomical_unit
    x_Jup = x_and_v_points[1][0] / astronomical_unit
    y_Jup = x_and_v_points[1][2] / astronomical_unit
    z_Jup = x_and_v_points[1][4] / astronomical_unit
    x_Sat = x_and_v_points[2][0] / astronomical_unit
    y_Sat = x_and_v_points[2][2] / astronomical_unit
    z_Sat = x_and_v_points[2][4] / astronomical_unit
    
    nb_points = 10000
    t_interp = np.linspace(t[0], t[-1], nb_points)
    
    x_Hal_interp = np.interp(t_interp, t, x_Hal)
    y_Hal_interp = np.interp(t_interp, t, y_Hal)
    z_Hal_interp = np.interp(t_interp, t, z_Hal)
    x_Jup_interp = np.interp(t_interp, t, x_Jup)
    y_Jup_interp = np.interp(t_interp, t, y_Jup)
    z_Jup_interp = np.interp(t_interp, t, z_Jup)
    x_Sat_interp = np.interp(t_interp, t, x_Sat)
    y_Sat_interp = np.interp(t_interp, t, y_Sat)
    z_Sat_interp = np.interp(t_interp, t, z_Sat)
    
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
    
    # halley, = ax.plot([], [], [], 'g', markersize=15)
    # jupiter, = ax.plot([], [], [], 'r', markersize=15)
    # saturne, = ax.plot([], [], [], 'b', markersize=15)
    # halley = ax.scatter([], [], [], 'g.')
    # jupiter = ax.scatter([], [], [], 'r.')
    # saturne = ax.scatter([], [], [], 'b.')
    
    ax.scatter(0, 0, 0, color="orange", label="Soleil")
    ax.axis([-30,30,-30,30])
    # ax.set_aspect("equal")
    time_text = ax.text(x=-30,y=-30, z=1, s="Temps : 0 années")
    
    # On run deux fois la simulation, une fois pour tracer les ellipses 
    # non perturbées, et l'autre pour la dépendance temporelle. Ici, on traces 
    # les orbites non perturbées
    ax.plot(x_and_v_points_non_perturb[0][0] / astronomical_unit, 
            x_and_v_points_non_perturb[0][2] / astronomical_unit, 
            x_and_v_points_non_perturb[0][4] / astronomical_unit, 
            'g-', alpha=0.3)   # Halley
    ax.plot(x_and_v_points_non_perturb[1][0] / astronomical_unit, 
            x_and_v_points_non_perturb[1][2] / astronomical_unit, 
            x_and_v_points_non_perturb[1][4] / astronomical_unit, 
            'r-', alpha=0.3)  # Jupiter
    ax.plot(x_and_v_points_non_perturb[2][0] / astronomical_unit, 
            x_and_v_points_non_perturb[2][2] / astronomical_unit, 
            x_and_v_points_non_perturb[2][4] / astronomical_unit, 
            'b-', alpha=0.3)    # Saturne



    def animate(i):
        graph_halley._offsets3d = ([x_Hal_interp[i]], 
                                   [y_Hal_interp[i]], 
                                   [z_Hal_interp[i]])
        graph_jupiter._offsets3d = ([x_Jup_interp[i]], 
                                    [y_Jup_interp[i]], 
                                    [z_Jup_interp[i]])
        graph_saturne._offsets3d = ([x_Sat_interp[i]], 
                                    [y_Sat_interp[i]], 
                                    [z_Sat_interp[i]])
        time_text.set_text(f"Temps : {round(t_interp[i]/(365*24*3600),1)} années")
        return graph_halley, graph_jupiter, graph_saturne
    
    
    graph_halley = ax.scatter(cond_init_Hal[0]/ astronomical_unit, 
                              cond_init_Hal[2]/ astronomical_unit, 
                              cond_init_Hal[4]/ astronomical_unit, 
                              "X", color='green', label="Halley")
    graph_jupiter = ax.scatter(cond_init_Jup[0]/ astronomical_unit, 
                               cond_init_Jup[2]/ astronomical_unit, 
                               cond_init_Jup[4]/ astronomical_unit, 
                               "X", color='red', label="Jupiter")
    graph_saturne = ax.scatter(cond_init_Sat[0]/ astronomical_unit, 
                               cond_init_Sat[2]/ astronomical_unit, 
                               cond_init_Sat[4]/ astronomical_unit, 
                               "X", color='blue', label="Saturne")
    plt.legend()
    anim = animation.FuncAnimation(fig, animate, frames=nb_points, 
                                   interval=40, repeat=False)
    # anim.save("gg_bo_ga.gif", dpi=300, writer=PillowWriter(fps=25))
    plt.show()

    