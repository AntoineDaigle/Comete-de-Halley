import numpy as np
import pandas as pd
from datetime import datetime


data = pd.read_csv("data_simulation.csv")

diff = (data["x_terre"]-data["x_halley"])**2
print(diff)

distance_terre_comete = np.sqrt(1)
distance_terre_soleil = 1


