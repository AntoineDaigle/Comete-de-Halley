
import numpy as np


# cond_init_Hal = [208752443562, -10384, 826804313528, -12985, -180080504407, 5594]
# Goal: 15.962 km/s


speed_cal = np.sqrt((-10384)**2 + 5594**2 + (-12985)**2)
speed_real = 15.962e3
print(speed_real, speed_cal)
print(100 * speed_cal/speed_real)