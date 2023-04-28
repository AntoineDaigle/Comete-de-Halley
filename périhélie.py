import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.constants import astronomical_unit


# data = pd.read_csv("data_simulation_1757.csv")
# date_initiale = datetime(year=1757, month=6, day=1)

# data = pd.read_csv("data_simulation_vendredi.csv")
data = pd.read_csv("data_simulation_impact_rien.csv")
date_initiale = datetime(year=2023, month=4, day=28, hour=14, minute=27)

# data = pd.read_csv("data_simulation_1758.csv")
# date_initiale = datetime(year=1758, month=11, day=14)

norme_halley = np.sqrt(data["x_halley"]**2 + data["y_halley"]**2 + data["z_halley"]**2 )


dist_min = np.min(norme_halley)

for i in range(len(norme_halley)):
    if norme_halley[i] == dist_min:
        loca = i
        break


date_périhélie = date_initiale + timedelta(seconds=data["time"][loca])
print("La date du périhélie est le:", date_périhélie)

temps = [date_initiale + timedelta(seconds=i) for i in data["time"]]



plt.plot(temps, norme_halley/astronomical_unit)
plt.scatter(date_périhélie, dist_min/astronomical_unit)
plt.xlabel("Date [YYYY-MM]")
plt.ylabel("Distance [au]")
plt.title("Distance entre la comète et le soleil.")
plt.tight_layout()
plt.minorticks_on()
plt.show()






