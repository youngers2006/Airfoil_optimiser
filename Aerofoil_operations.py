import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

class Airfoil:
    """Airfoil, holds it genes and fitness value"""
    def __init__(self, genes):
        self.genes = genes  # list containing the Hicks-Henne amplitudes that were added to optimise the airfoil
        self.fitness = 0.0  # fitness is defines as the L/D max of the airfoil as this is what were optimising

def plot_airfoil(coords, filename, title):
    """Plots the airfoil shape and saves it to a file"""
    plt.figure(figsize=(12, 6))
    plt.plot(coords[:, 0], coords[:, 1], 'b-')
    plt.title(title)
    plt.xlabel("Chord (x/c)")
    plt.ylabel("Thickness (y/c)")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(filename)
    plt.close() 

def load_baseline_airfoil(coords_string):
    """Loads starting airfoil coordinates and returns """
    lines = coords_string.strip().split('\n') # splits chord file up into individual chords
    lines = [line for line in lines if line.strip()] # removes whitespace
    coords = np.array([list(map(float, line.split())) for line in lines]) # gets chords
    
    # Find the split point (leading edge, approximately x=0)
    le_index = np.argmin(coords[:, 0])
    
    # Split into upper and lower surfaces based on standard Selig format
    upper_surface = np.flipud(coords[:le_index+1, :]) # Flip to go from LE to TE
    lower_surface = coords[le_index:, :]             # Already goes from LE to TE
    
    # Create interpolation functions for each surface to create plot
    interp_upper = interp1d(upper_surface[:, 0], upper_surface[:, 1], kind='cubic', fill_value="extrapolate")
    interp_lower = interp1d(lower_surface[:, 0], lower_surface[:, 1], kind='cubic', fill_value="extrapolate")

    return interp_upper, interp_lower