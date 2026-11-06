"""Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap (bands 4-5)."""
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Approach: Single central square dielectric rod in air.
    This is a distinct approach from the multi-rod periodic structure,
    relying on a single high-contrast feature to create strong Bragg scattering.
    Square rods have slower-decaying Fourier coefficients than circular rods,
    which often leads to larger band gaps in square lattices.
    """
    size = problem["pixels"]
    cell = [[0] * size for _ in range(size)]
    
    # Define a central square region.
    # Symmetry requires the square to be centered.
    # For size 32, center is 15.5.
    # Let side length be L.
    # Start index S such that the square spans [S, S+L-1].
    # Symmetry condition: S + (31 - (S+L-1)) = 31 => 2S + L - 1 = 31 => 2S + L = 32.
    # We choose L=20, which gives 2S = 12 => S=6.
    # The square covers indices 6 to 25 inclusive.
    # Fill factor: 20*20 / 32*32 = 400/1024 ~ 39%.
    L = 20
    S = (size - L) // 2
    
    for i in range(size):
        for j in range(size):
            if S <= i < S + L and S <= j < S + L:
                cell[i][j] = 1
                
    return cell
