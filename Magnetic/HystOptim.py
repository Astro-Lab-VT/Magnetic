from numpy.linalg import norm
import math
from scipy.optimize import differential_evolution
from scipy.special import erfinv
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import oommfc as mc
import discretisedfield as df
import micromagneticmodel as mm


def save_figure(waveDirections):
    waveDirections=waveDirections.reshape(3,2)
    resolution = 5000
    waveNumber = 6 * np.pi
    numWaves = 3
    relativeDensity = 0.6
    thetas = [15, 45]  # degrees
    R = np.eye(2)
    
    axes1 = R @ np.array([1, 0])
    axes2 = R @ np.array([0, 1])
    
#     waveDirections = np.zeros((numWaves, 2))
    
#     # Generate constrained directions
#     for i in range(numWaves):
#         while True:
#             ni = np.random.randn(2)
#             ni_norm = ni / norm(ni)

#             angle1 = min(
#                 np.degrees(np.arccos(np.clip(np.dot(ni_norm, axes1), -1, 1))),
#                 np.degrees(np.arccos(np.clip(np.dot(ni_norm, -axes1), -1, 1))),
#             )
#             angle2 = min(
#                 np.degrees(np.arccos(np.clip(np.dot(ni_norm, axes2), -1, 1))),
#                 np.degrees(np.arccos(np.clip(np.dot(ni_norm, -axes2), -1, 1))),
#             )

#             if angle1 < thetas[0] or angle2 < thetas[1]:
#                 waveDirections[i, :] = ni_norm
#                 break

    # Generate GRF with fixed phases
    wavePhases = 2 * np.pi * np.ones(numWaves)
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    GRF = np.zeros_like(X)
    print(np.shape(waveDirections))
    for i in range(numWaves):
        dotProd = waveDirections[i, 0] * X + waveDirections[i, 1] * Y
        GRF += np.sqrt(2 / numWaves) * np.cos(dotProd * waveNumber + wavePhases[i])

    # Level set
    phi0 = np.sqrt(2) * erfinv(2 * relativeDensity - 1)
    binaryImage = GRF > phi0

    # Add frame
    frame_thickness = 50
    binaryImage[:frame_thickness, :] = 0
    binaryImage[-frame_thickness:, :] = 0
    binaryImage[:, :frame_thickness] = 0
    binaryImage[:, -frame_thickness:] = 0

    # Save image
    image_filename = "spinodoid_image.png"
    imageio.imwrite(image_filename, (binaryImage * 255).astype(np.uint8))

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(binaryImage, cmap='gray')
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return waveDirections  # Optionally return directions for reuse




def angular_constraint(x):
    c = []
    numWaves = 3
    thetas = [15, 90]  # in degrees
    axes1 = np.array([1, 0])
    axes2 = np.array([0, 1])
    ni_norm = np.reshape(x, (numWaves, 2))

    for i in range(numWaves):
        ni = ni_norm[i]
        ni = ni / np.linalg.norm(ni)

        angle1 = min(
            np.degrees(np.arccos(np.clip(np.dot(ni, axes1), -1, 1))),
            np.degrees(np.arccos(np.clip(np.dot(ni, -axes1), -1, 1)))
        )
        angle2 = min(
            np.degrees(np.arccos(np.clip(np.dot(ni, axes2), -1, 1))),
            np.degrees(np.arccos(np.clip(np.dot(ni, -axes2), -1, 1)))
        )

        angdiff1 = angle1 - thetas[0]
        angdiff2 = angle2 - thetas[1]

        c.append(angdiff1)
        c.append(angdiff2)

    return np.array(c)


def objective_function(x):
    waveDirections=x
    save_figure(waveDirections)
    
    region = df.Region(p1=(-50e-9, -50e-9, -50e-9), p2=(50e-9, 50e-9, 50e-9))
    mesh = df.Mesh(region=region, cell=(2.5e-9, 2.5e-9, 2.5e-9))

    system = mm.System(name="hysteresis")
    system.energy = (
        mm.Exchange(A=1e-12)
        #+ mm.CubicAnisotropy(K=1e4, u1=(0, 0, 1), u2=(0, 1, 0))
        #+ mm.DMI(D=1e-3, crystalclass="O")
        + mm.UniaxialAnisotropy(K=1e3, u=(0, 1, 0))

        + mm.Demag()
    ) 

    image_path = "spinodoid_image.png"
    image = Image.open(image_path).convert("L")  
    image = image.resize((64, 64))  
    image_data = np.array(image) 

    x_min, x_max = -40e-9, 40e-9
    y_min, y_max = -40e-9, 40e-9
    z_min, z_max = -1.25e-9, 1.25e-9  
    frame_thickness = 2.5e-9

    def Ms_fun(point):
        x, y, z = point

        if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
            if (
                x <= x_min + frame_thickness or x >= x_max - frame_thickness or
                y <= y_min + frame_thickness or y >= y_max - frame_thickness
            ):
                return 1e6  

            ix = int(((x - x_min) / (x_max - x_min)) * (image_data.shape[1] - 1))
            iy = int(((y - y_min) / (y_max - y_min)) * (image_data.shape[0] - 1))

            intensity = image_data[iy, ix]

            if intensity < 0.1:  
                return 0

            return 1e6  

        return 0  

    num_points = 200  

    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x_vals, y_vals)

    Ms_values = np.zeros_like(X)

    for i in range(num_points):
        for j in range(num_points):
            Ms_values[i, j] = Ms_fun((X[i, j], Y[i, j], 0))

    Ms_values[Ms_values > 0] = 1 
    plt.figure(figsize=(8, 6))
    plt.imshow(Ms_values, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="gray_r")
    plt.colorbar(label="Ms (normalized)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Visualization of Ms_fun on xy-plane (z=0)")
    plt.show()
    system.m = df.Field(mesh, nvdim=3, value=(0, 0, -1), norm=Ms_fun, valid="norm")
    Hmin = (0.00001/mm.consts.mu0, -1 / mm.consts.mu0, 0)
    Hmax = (0.00001/mm.consts.mu0, 1/ mm.consts.mu0, 0)
    n = 41
    hd = mc.HysteresisDriver()
    hd.drive(system, Hmin=Hmin, Hmax=Hmax, n=n)
    hyst_B=np.array(system.table.data["By_hysteresis"])
    def find_zero_crossings(x, y):
        sign_changes = np.where(np.diff(np.sign(y)) != 0)[0]
        zero_crossings = x[sign_changes] - y[sign_changes] * (x[sign_changes + 1] - x[sign_changes]) / (y[sign_changes + 1] - y[sign_changes])
        return zero_crossings
    hyst_y=np.array(system.table.data["my"])
    zero_points = find_zero_crossings(hyst_B, hyst_y)
    
    print(np.min(zero_points))
    
    
    return np.min(zero_points)


def penalized_objective(x):
    penalty = np.sum(np.maximum(angular_constraint(x), 0.0)) * 1e6
    return objective_function(x) + penalty



def run_optimization(num_iterations=1, numWaves=3):
    dim = 2 * numWaves
    bounds = [(0, 1)] * dim

    for iteration in range(num_iterations):
        result = differential_evolution(
            penalized_objective,
            bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            polish=True,
            seed=None,
            disp=True
        )

        print(f"\nIteration {iteration+1}")
        print("Optimal x =", result.x)
        print("Objective value =", result.fun)

        generate_spinodoid_image(result.x, filename=f'spinodoid_image_iter_{iteration+1}.png')


if __name__ == '__main__':
    run_optimization(num_iterations=1)
