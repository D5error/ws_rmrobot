import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs)

def draw_axes(ax, x, y, z, roll, pitch, yaw, length=1):
    # Convert angles from degrees to radians
    roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    # Axes directions
    axes = np.eye(3) * length
    rotated_axes = R @ axes
    
    # Draw rotated axes (RGB colors)
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        arrow = Arrow3D([x, x+rotated_axes[0,i]], [y, y+rotated_axes[1,i]], [z, z+rotated_axes[2,i]], 
                       mutation_scale=15, lw=2, arrowstyle="-|>", color=colors[i])
        ax.add_artist(arrow)
        ax.text(x+rotated_axes[0,i]*1.1, y+rotated_axes[1,i]*1.1, z+rotated_axes[2,i]*1.1, 
                labels[i], color=colors[i], fontsize=12)

# Parameters (position in meters, angles in degrees)
x, y, z = -0.23105600011929198, -0.015000000000000008, 0.6578508848717886
roll, pitch, yaw = -157.82400741109663, -46.041792997382224, 150.47984836514502

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the original axes at (0,0,0) in light gray
draw_axes(ax, 0, 0, 0, 0, 0, 0, length=1.5)

# Draw the rotated axes at specified position
draw_axes(ax, x, y, z, roll, pitch, yaw, length=1)

# Set limits and labels
ax.set_xlim([-5, 0.5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel('X (Red)')
ax.set_ylabel('Y (Green)')
ax.set_zlabel('Z (Blue)')
ax.set_title(f'Position: ({x},{y},{z})\nOrientation: Roll={roll}°, Pitch={pitch}°, Yaw={yaw}°')

plt.tight_layout()
plt.show()