import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


data = pd.read_csv("data_simulation.csv")

date_initiale = datetime(year=1758, month=11, day=14)

projection = []


for i in range(len(data["time"])):
    d_tc = np.array([data["x_terre"][i]-data["x_halley"][i],
                    data["y_terre"][i]-data["y_halley"][i],
                    data["z_terre"][i]-data["z_halley"][i]])
    d_ts = np.array([data["x_terre"][i],
                    data["y_terre"][i],
                    data["z_terre"][i]])
    
    
    res_dot = np.dot(d_tc, d_ts)
    if res_dot >= 0 :
        projection.append(1)
    else:
        projection.append(0)


temps = [date_initiale + timedelta(seconds=i) for i in data["time"]]
plt.plot(temps, projection)
plt.xlabel("Date [YYYY-MM]")
plt.minorticks_on()
plt.title("Observation possible de la comète de Halley.")
plt.tight_layout()
plt.show()
