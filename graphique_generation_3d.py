import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime, timedelta
import pandas as pd



# Load simulation data
pert_data = pd.read_csv("pert_data.csv")
non_pert_data = pd.read_csv("non_pert_data.csv")
condition_initiale = pd.read_csv("Condition_initiale_simulation.csv")
delta = condition_initiale["delta"][0]


# Date lors de la simulation
# first_date = datetime.today()
first_date = datetime(year=1900, month=1, day=1)




nb_points = 500
t = list(pert_data["time"])
t_interp = np.linspace(t[0], t[-1], nb_points)


x_Hal_interp = np.interp(t_interp, t, pert_data["x_halley"]/astronomical_unit)
y_Hal_interp = np.interp(t_interp, t, pert_data["y_halley"]/astronomical_unit)
z_Hal_interp = np.interp(t_interp, t, pert_data["z_halley"]/astronomical_unit)
x_Jup_interp = np.interp(t_interp, t, pert_data["x_jupiter"]/astronomical_unit)
y_Jup_interp = np.interp(t_interp, t, pert_data["y_jupiter"]/astronomical_unit)
z_Jup_interp = np.interp(t_interp, t, pert_data["z_jupiter"]/astronomical_unit)
x_Sat_interp = np.interp(t_interp, t, pert_data["x_saturne"]/astronomical_unit)
y_Sat_interp = np.interp(t_interp, t, pert_data["y_saturne"]/astronomical_unit)
z_Sat_interp = np.interp(t_interp, t, pert_data["z_saturne"]/astronomical_unit)
x_Ter_interp = np.interp(t_interp, t, pert_data["x_terre"]/astronomical_unit)
y_Ter_interp = np.interp(t_interp, t, pert_data["y_terre"]/astronomical_unit)
z_Ter_interp = np.interp(t_interp, t, pert_data["z_terre"]/astronomical_unit)




fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))



ax.scatter(0, 0, 0, color="orange", label="Soleil")
ax.axis([-30,30,-30,30])
dist_Ter_Hal = np.linalg.norm([x_Hal_interp[0]-x_Ter_interp[0], 
                                y_Hal_interp[0]-y_Ter_interp[0], 
                                z_Hal_interp[0]-z_Ter_interp[0]])


# On run deux fois la simulation, une fois pour tracer les ellipses 
# non perturbées, et l'autre pour la dépendance temporelle. Ici, on traces 
# les orbites non perturbées
ax.plot(non_pert_data["x_halley"] / astronomical_unit, 
        non_pert_data["y_halley"] / astronomical_unit, 
        non_pert_data["z_halley"] / astronomical_unit, 
        'g-', alpha=0.3)   # Halley
ax.plot(non_pert_data["x_jupiter"] / astronomical_unit, 
        non_pert_data["y_jupiter"] / astronomical_unit, 
        non_pert_data["z_jupiter"] / astronomical_unit, 
        'r-', alpha=0.3)  # Jupiter
ax.plot(non_pert_data["x_saturne"] / astronomical_unit, 
        non_pert_data["y_saturne"] / astronomical_unit, 
        non_pert_data["z_saturne"] / astronomical_unit, 
        'b-', alpha=0.3)    # Saturne
ax.plot(non_pert_data["x_terre"] / astronomical_unit, 
        non_pert_data["y_terre"] / astronomical_unit, 
        non_pert_data["z_terre"] / astronomical_unit, 
        'k-', alpha=0.3)



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
    graph_terre._offsets3d = ([x_Ter_interp[i]],
                                [y_Ter_interp[i]],
                                [z_Ter_interp[i]])
    dist_Ter_Hal = np.linalg.norm([x_Hal_interp[i]-x_Ter_interp[i], 
                                y_Hal_interp[i]-y_Ter_interp[i], 
                                z_Hal_interp[i]-z_Ter_interp[i]])
    
    temps_ecoule = t_interp[i]
    iterative_time = first_date + timedelta(seconds=temps_ecoule)
    erreur_temporel = timedelta(seconds=(temps_ecoule * delta))

    ax.set_title(f"Date : {iterative_time.strftime('%Y/%m/%d')} \u00B1 {round(erreur_temporel.total_seconds())}s \nDistance Terre-comète : {round(dist_Ter_Hal,1)} UA")

    return graph_halley, graph_jupiter, graph_saturne, graph_terre


graph_halley = ax.scatter(pert_data["x_halley"][0]/ astronomical_unit, 
                            pert_data["y_halley"][0]/ astronomical_unit, 
                            pert_data["z_halley"][0]/ astronomical_unit, 
                            "X", color='green', label="Halley")
graph_jupiter = ax.scatter(pert_data["x_jupiter"][0]/ astronomical_unit, 
                            pert_data["y_jupiter"][0]/ astronomical_unit, 
                            pert_data["z_jupiter"][0]/ astronomical_unit, 
                            "X", color='red', label="Jupiter")
graph_saturne = ax.scatter(pert_data["x_saturne"][0]/ astronomical_unit, 
                            pert_data["y_saturne"][0]/ astronomical_unit, 
                            pert_data["z_saturne"][0]/ astronomical_unit, 
                            "X", color='blue', label="Saturne")
graph_terre = ax.scatter(pert_data["x_terre"][0]/ astronomical_unit, 
                            pert_data["y_terre"][0]/ astronomical_unit, 
                            pert_data["z_terre"][0]/ astronomical_unit,
                            "X", color='k', label="Terre")


plt.legend()
anim = animation.FuncAnimation(fig, animate, frames=nb_points, 
                                interval=10, repeat=False)
# anim.save("simulation.gif", dpi=300)
plt.show()




