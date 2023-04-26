import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime, timedelta
import pandas as pd



# Load simulation data
data = pd.read_csv("data_simulation_1758.csv")

# Date lors de la simulation
# first_date = datetime(year=2023, month=4, day=28, hour=14, minute=27)
first_date = datetime(year=1758, month=11, day=14)

nb_points = 500
t = list(data["time"])
t_interp = np.linspace(t[0], t[-1], nb_points)


x_Hal_interp = np.interp(t_interp, t, data["x_halley"]/astronomical_unit)
y_Hal_interp = np.interp(t_interp, t, data["y_halley"]/astronomical_unit)
z_Hal_interp = np.interp(t_interp, t, data["z_halley"]/astronomical_unit)
x_Jup_interp = np.interp(t_interp, t, data["x_jupiter"]/astronomical_unit)
y_Jup_interp = np.interp(t_interp, t, data["y_jupiter"]/astronomical_unit)
z_Jup_interp = np.interp(t_interp, t, data["z_jupiter"]/astronomical_unit)
x_Sat_interp = np.interp(t_interp, t, data["x_saturne"]/astronomical_unit)
y_Sat_interp = np.interp(t_interp, t, data["y_saturne"]/astronomical_unit)
z_Sat_interp = np.interp(t_interp, t, data["z_saturne"]/astronomical_unit)
x_Ter_interp = np.interp(t_interp, t, data["x_terre"]/astronomical_unit)
y_Ter_interp = np.interp(t_interp, t, data["y_terre"]/astronomical_unit)
z_Ter_interp = np.interp(t_interp, t, data["z_terre"]/astronomical_unit)




fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))



ax.scatter(0, 0, 0, color="orange", label="Soleil")
ax.axis([-37,37,-37,37])
dist_Ter_Hal = np.linalg.norm([x_Hal_interp[0]-x_Ter_interp[0], 
                                y_Hal_interp[0]-y_Ter_interp[0], 
                                z_Hal_interp[0]-z_Ter_interp[0]])


# On run deux fois la simulation, une fois pour tracer les ellipses 
# non perturbées, et l'autre pour la dépendance temporelle. Ici, on traces 
# les orbites non perturbées
ax.plot(data["x_halley"] / astronomical_unit, 
        data["y_halley"] / astronomical_unit, 
        data["z_halley"] / astronomical_unit, 
        'g-', alpha=0.3)   # Halley
ax.plot(data["x_jupiter"] / astronomical_unit, 
        data["y_jupiter"] / astronomical_unit, 
        data["z_jupiter"] / astronomical_unit, 
        'r-', alpha=0.3)  # Jupiter
ax.plot(data["x_saturne"] / astronomical_unit, 
        data["y_saturne"] / astronomical_unit, 
        data["z_saturne"] / astronomical_unit, 
        'b-', alpha=0.3)    # Saturne
ax.plot(data["x_terre"] / astronomical_unit, 
        data["y_terre"] / astronomical_unit, 
        data["z_terre"] / astronomical_unit, 
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

    ax.set_title(f"Date : {iterative_time.strftime('%Y/%m/%d')} \nDistance Terre-comète : {round(dist_Ter_Hal,1)} UA")

    return graph_halley, graph_jupiter, graph_saturne, graph_terre


graph_halley = ax.scatter(data["x_halley"][0]/ astronomical_unit, 
                            data["y_halley"][0]/ astronomical_unit, 
                            data["z_halley"][0]/ astronomical_unit, 
                            "X", color='green', label="Halley")
graph_jupiter = ax.scatter(data["x_jupiter"][0]/ astronomical_unit, 
                            data["y_jupiter"][0]/ astronomical_unit, 
                            data["z_jupiter"][0]/ astronomical_unit, 
                            "X", color='red', label="Jupiter")
graph_saturne = ax.scatter(data["x_saturne"][0]/ astronomical_unit, 
                            data["y_saturne"][0]/ astronomical_unit, 
                            data["z_saturne"][0]/ astronomical_unit, 
                            "X", color='blue', label="Saturne")
graph_terre = ax.scatter(data["x_terre"][0]/ astronomical_unit, 
                            data["y_terre"][0]/ astronomical_unit, 
                            data["z_terre"][0]/ astronomical_unit,
                            "X", color='k', label="Terre")


plt.legend()
anim = animation.FuncAnimation(fig, animate, frames=nb_points, 
                                interval=10, repeat=False)
# anim.save("simulation.gif", dpi=300)
plt.show()





