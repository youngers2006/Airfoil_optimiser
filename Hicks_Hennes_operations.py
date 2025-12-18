import numpy as np

def hicks_henne_bump(x, x_loc, t=0.1):
    """Hicks-Henne bump function"""
    return np.sin(np.pi * np.power(x, np.log(0.5) / np.log(x_loc)))**4

def apply_hicks_henne(genes, base_upper, base_lower):
    """Applies Hicks-Henne bumps to the baseline airfoil."""
    num_bumps_per_surface = len(genes) // 2 # divide by two as half on upper surface and half on lower
    x = np.linspace(0.0, 1.0, 100)
    
    bump_locations = np.linspace(0.1, 0.8, num_bumps_per_surface)
    
    y_upper_mod = np.zeros_like(x)
    y_lower_mod = np.zeros_like(x)
    
    for i in range(num_bumps_per_surface):
        y_upper_mod += genes[i] * hicks_henne_bump(x, bump_locations[i]) # calculates the effect of the bump on all points on upper surface

    for i in range(num_bumps_per_surface):
        y_lower_mod += genes[i + num_bumps_per_surface] * hicks_henne_bump(x, bump_locations[i]) # calculates the effect of the bump on all points on lower surface

    new_y_upper = base_upper(x) + y_upper_mod
    new_y_lower = base_lower(x) + y_lower_mod
    
    if np.any(new_y_lower[1:-1] > new_y_upper[1:-1]): # checks that the airfoil upper and lower sufaces do not intersect
        return None # if they intersect the airfoil is flagged by None to be repaired

    upper = np.vstack((np.flipud(x), np.flipud(new_y_upper))).T # Vertically stacks the upper surface but reverses the order to comply with standard ordering
    lower = np.vstack((x, new_y_lower)).T[1:] # Vertically stacks lower surface
    
    final_coords = np.vstack((upper, lower)) # Verticall stacks both together to create new coords
    return final_coords