import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# https://stackoverflow.com/questions/50342300/animating-3d-scatter-plot-using-python-mplotlib-via-serial-data

def update_lines(num):
    dx, dy, dz = np.random.random((3,)) * 255 * 2 - 255  # replace this line with code to get data from serial line
    text.set_text("{:d}: [{:.0f},{:.0f},{:.0f}]".format(num, dx, dy, dz))  # for debugging
    # x.append(dx)
    # y.append(dy)
    # z.append(dz)
    graph._offsets3d = ([dx], [dy], [dz])
    return graph,


x = [0]
y = [0]
z = [0]

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
graph = ax.scatter(x, y, z, color='orange')
text = fig.text(0, 1, "TEXT", va='top')  # for debugging

ax.set_xlim3d(-255, 255)
ax.set_ylim3d(-255, 255)
ax.set_zlim3d(-255, 255)

# Creating the Animation object
ani = animation.FuncAnimation(fig, update_lines, frames=200, interval=50, blit=False)
plt.show()