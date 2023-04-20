import numpy as np
from scipy.constants import gravitational_constant
import matplotlib.pyplot as plt
from tqdm import tqdm


class runge_kutta:
    def __init__(
            self, 
            a:float, 
            b:float, 
            N:int,
            cond_init:list,
            delta:int):
        """Classe permettant d'appliquer la methode de Runge-Kutta d'ordre 4 a la resolution du probleme de la comete de Halley.

        Args:
            a (float): Borne inferieur
            b (float): Borne superieur
            N (int): Nombre d'iteration
            cond_init (list): Liste des conditions initiales [x, vx, y, vy, z, vz]
            delta (int): Erreur souhaite
        """

        self.a = a
        self.b = b 
        self.N = N 
        self.h = (self.b-self.a)/self.N
        self.og_h = self.h
        self.cond_init = cond_init
        self.delta = delta
        # self.tpoints = np.arange(self.a,self.b,self.h)


    def f(self, cond_ini:list)->np:
        """Methode contenant les equations differentielles.

        Args:
            cond_ini (list): Liste contenant les conditions initiales comme [x, vx, y, vy, z, vz]

        Returns:
            np: Matrice contenant les valeurs d'equations differentielles.
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





    def runge_kutta(self, h:float, initial_conditions:list)->tuple:
        """Methode appliquant la routine de Runge-Kutta de 4eme ordre.

        Args:
            h (float): Increment temporel
            initial_conditions (list): Listes des conditions initiales

        Returns:
            tuple: Tuple contenant les coefficients de Runge-Kutta
        """
        k1 = h * self.f(initial_conditions)
        k2 = h * self.f(initial_conditions+0.5*k1)
        k3 = h * self.f(initial_conditions+0.5*k2)
        k4 = h * self.f(initial_conditions+k3)
        return k1, k2, k3, k4
        


    def simulation(self):
        """Methode de la classe contenant la simulation.

        Returns:
            tuple: Resultat de la simulation dans un tuple.
        """
        xpoints = []
        vxpoints = []
        ypoints = []
        vypoints = []
        zpoints = []
        vzpoints = []
        h_values = []
        tpoints = [0]

        def rho(i:np, j:np)->float:
            """Fonction permettant de calculer le ratio rho de l'equation 8.53 du Newman.

            Args:
                i (np): Element x_1 de 8.53.
                j (np): Element x_2 de 8.53.

            Returns:
                float: Valeur de du ratio rho.
            """
            eps_x = abs(i[0]-j[0])/30
            eps_y = abs(i[2]-j[2])/30
            eps_z = abs(i[4]-j[4])/30
            return self.h * self.delta / np.sqrt(eps_x**2 + eps_y**2 + eps_z**2)
        
        def h_prime(rho:float, h:float)->float:
            """Fonction qui calcul la valeur de l'increment ideal. 8.52 du Newman.

            Args:
                rho (float): Ratio rho de l'equation 8.53.
                h (float): Increment regulier.

            Returns:
                float: Valeur de l'increment ideal
            """
            return h * rho**0.25


        while tpoints[-1] < self.b:
            # Ligne permettant d'avoir un indicateur de l'avancement de la boucle while.
            # print(100 * tpoints[-1] / self.b)

            xpoints.append(self.cond_init[0])
            vxpoints.append(self.cond_init[1])

            ypoints.append(self.cond_init[2])
            vypoints.append(self.cond_init[3])
            
            zpoints.append(self.cond_init[4])
            vzpoints.append(self.cond_init[5])
            
            h_values.append(self.h)

            # Calcul de l'element x_1
            k1_h, k2_h, k3_h, k4_h = self.runge_kutta(self.h, self.cond_init)
            cond_init_h = self.cond_init + (k1_h+2*k2_h+2*k3_h+k4_h)/6

            k1_h_p_h, k2_h_p_h, k3_h_p_h, k4_h_p_h = self.runge_kutta(self.h, cond_init_h)            
            cond_init_h_p_h = cond_init_h + (k1_h_p_h+2*k2_h_p_h+2*k3_h_p_h+k4_h_p_h)/6


            # Calcul de l'element x_2
            k1_2h, k2_2h, k3_2h, k4_2h = self.runge_kutta(2 * self.h, self.cond_init)
            cond_init_2h = self.cond_init + (k1_2h+2*k2_2h+2*k3_2h+k4_2h)/6


            # Optimisation de l'increment
            p = rho(cond_init_h_p_h, cond_init_2h)
            h_pri = h_prime(p, self.h)


            if p >= 1:
                self.cond_init = cond_init_h_p_h
                self.h = min(2* self.h, h_pri)

            else:
                k1, k2, k3, k4 = self.runge_kutta(h_pri, self.cond_init)
                self.cond_init += (k1+2*k2+2*k3+k4)/6
                self.h = h_pri


            new_time = tpoints[-1] + self.h
            tpoints.append(new_time)


        return tpoints[1:], xpoints, vxpoints, ypoints, vypoints, zpoints, vzpoints, h_values



if __name__ == "__main__":
    t, x, vx, y, vy, z, vz, h = runge_kutta(0, 2e9, 1E05, [4e12, 0, 0, 500, 0, 0], 1E-06).simulation()

    plt.plot(t, x, label="x")
    plt.plot(t, y, label="y")
    plt.legend()
    plt.show()
    
    
    

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot(x, y, z)
    ax.scatter(0, 0, c="red", label="Soleil")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.legend()
    plt.show()
