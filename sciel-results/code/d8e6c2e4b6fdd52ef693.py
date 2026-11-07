"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Calculate pair sums and check if they fit
    pair_sums = demands[:, None] + demands[None, :]
    fits = pair_sums <= capacity
    
    # Calculate residual space for each pair
    residuals = capacity - pair_sums
    
    # We need to count how many items from the original demand set fit into the residual space.
    # Let k(R) be the number of items d_k such that d_k <= R.
    # We use binary search on the sorted demands for efficiency.
    sorted_demands = np.sort(demands)
    
    # Initialize the heuristic matrix
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if np.any(fits):
        # Extract valid residuals and sums
        valid_residuals = residuals[fits]
        valid_sums = pair_sums[fits]
        
        # Count items fitting in each residual
        # searchsorted with side='right' gives the count of elements <= value
        counts = np.searchsorted(sorted_demands, valid_residuals, side='right')
        
        # Heuristic: Inversely proportional to the count of items that fit in the residual.
        # We also scale by the pair sum to prefer larger pairs (tight packing).
        # eta = pair_sum / (1 + count_fitting)
        # The +1 ensures that if count is 0 (perfect dead gap), we get pair_sum / 1 = pair_sum.
        # If count is large, the value drops.
        
        valid_eta = valid_sums / (1.0 + counts)
        
        # Place back into matrix
        eta[fits] = valid_eta
        
    # Ensure no negative or zero values remain (though logic above handles fits, 
    # and non-fits are 1e-9). 
    # The task says "Heuristic entries that are not positive become 1e-9".
    eta[eta <= 0] = 1e-9
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    return eta
# EVOLVE-BLOCK-END
