"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    # Convert inputs to numpy arrays for vectorized operations
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Vectorized calculation
    # d_ij is distance_matrix
    # d_j0 is the column 0 of distance_matrix, reshaped to (1, n) to broadcast against rows
    d_j0 = distance_matrix[:, 0].reshape(1, -1)
    
    # The heuristic is prize_j / (d_ij + d_j0)
    # We add a small epsilon to the denominator to avoid division by zero 
    # (though d_j0 > 0 for j != 0, and j != 0 is always true in ACO moves from 0)
    # However, d_ij can be 0 if i==j, but i!=j in moves.
    # So denominator is strictly positive for valid moves.
    # We still add 1e-9 for safety against any 0s in the matrix if inputs are weird.
    denom = distance_matrix + d_j0 + 1e-9
    
    # Calculate heuristic
    # prizes is shape (n,), need to broadcast to (n, n) by rows?
    # Wait. eta[i, j] depends on prize of j.
    # So we need prize[j] for each column j.
    # prizes.reshape(1, -1) is shape (1, n).
    # denom is shape (n, n).
    # Division broadcasts correctly.
    eta = prizes.reshape(1, -1) / denom
    
    # Handle non-positive or non-finite values as per contract
    # "Heuristic entries that are not positive become 1e-9"
    # This implies if eta <= 0 or not finite, set to 1e-9.
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
