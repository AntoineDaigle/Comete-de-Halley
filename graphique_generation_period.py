import numpy as np
from scipy.constants import gravitational_constant, astronomical_unit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from scipy.signal import find_peaks


# Load simulation data
pert_data = pd.read_csv("pert_data.csv")
non_pert_data = pd.read_csv("non_pert_data.csv")
condition_initiale = pd.read_csv("Condition_initiale_simulation.csv")
delta = condition_initiale["delta"][0]


nb_points = 10000
t = list(pert_data["time"])
t_interp = np.linspace(t[0], t[-1], nb_points)
x_interp = np.interp(t_interp, t, pert_data["x_terre"]/astronomical_unit)

pic = find_peaks(x_interp)
print(np.diff(pic[0]))


moyenne = np.mean(np.diff(pic[0]))
print(timedelta(seconds=(t[-1]-t[0]) * moyenne/nb_points))





# for i in pic_terre[0]:
#     plt.axvline(pert_data["time"][i])



# plt.plot(pert_data["time"], pert_data["x_terre"], c="k", label="Terre")
# plt.plot(pert_data["time"], pert_data["x_jupiter"], c="red", label="Jupiter")
# plt.plot(pert_data["time"], pert_data["x_saturne"], c="blue", label="Saturne")
# plt.plot(pert_data["time"], pert_data["x_halley"], c="green", label="Halley")
# plt.legend()

# plt.show()