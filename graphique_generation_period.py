import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from scipy.signal import find_peaks


# Load simulation data
data = pd.read_csv("data_simulation.csv")
condition_initiale = pd.read_csv("Condition_initiale_simulation.csv")
delta = condition_initiale["delta"][0]


nb_points = 10000
t = list(data["time"])
t_interp = np.linspace(t[0], t[-1], nb_points)
x_ter_interp = np.interp(t_interp, t, data["x_terre"]/astronomical_unit)
x_hal_interp = np.interp(t_interp, t, data["x_halley"]/astronomical_unit)
x_jup_interp = np.interp(t_interp, t, data["x_jupiter"]/astronomical_unit)
x_sat_interp = np.interp(t_interp, t, data["x_saturne"]/astronomical_unit)

try:
    pic_ter = find_peaks(x_ter_interp)
    print("Terre", timedelta(seconds=(t[-1]-t[0]) * np.mean(np.diff(pic_ter[0]))/nb_points))
    pic_hal = find_peaks(x_hal_interp)
    print("Halley", timedelta(seconds=(t[-1]-t[0]) * np.mean(np.diff(pic_hal[0]))/nb_points))
    pic_jup = find_peaks(x_jup_interp)
    print("Jupiter", timedelta(seconds=(t[-1]-t[0]) * np.mean(np.diff(pic_jup[0]))/nb_points))
    pic_sat = find_peaks(x_sat_interp)
    print("Saturne", timedelta(seconds=(t[-1]-t[0]) * np.mean(np.diff(pic_sat[0]))/nb_points))
except ValueError:
    print("Pas de période.")

plt.plot(data["time"], data["x_terre"], c="k", label="Terre")
plt.plot(data["time"], data["x_jupiter"], c="red", label="Jupiter")
plt.plot(data["time"], data["x_saturne"], c="blue", label="Saturne")
plt.plot(data["time"], data["x_halley"], c="green", label="Halley")
plt.legend()

plt.show()