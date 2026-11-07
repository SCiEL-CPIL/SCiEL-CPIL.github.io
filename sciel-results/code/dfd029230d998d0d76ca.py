import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0 # 300s limit, keep 20s margin

def heuristics(demands, capacity):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones((len(demands), len(demands))) * 1e-9

    n = len(demands)
    # Convert to float array to be safe
    d = np.array(demands, dtype=np.float64)
    
    # Calculate sums
    sums = d[:, None] + d[None, :]
    
    # Create mask for valid pairs (fit in capacity)
    valid = sums <= capacity
    
    # Initialize eta with small positive values for invalid entries
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(valid):
        # Calculate gap for valid pairs
        gaps = capacity - sums[valid]
        
        # Structural Change: 
        # 1. Use a softer singularity (alpha=1.5) to balance sharpness and exploration.
        # 2. Use a moderate fill ratio power (beta=3) to encourage full bins.
        # 3. ROW-NORMALIZE the heuristic. 
        #    This ensures that for each item i, the total "attractiveness" 
        #    is scaled, preventing large items (which have fewer valid partners) 
        #    from having disproportionately large heuristic sums compared to small items.
        #    This allows pheromone trails to have a more consistent influence 
        #    across all items, potentially improving global search.
        
        alpha = 1.5
        beta = 3.0
        epsilon = 0.5
        
        # Calculate raw scores
        # inv_gap_term = 1 / (gap + epsilon)^alpha
        # fill_term = (sum / capacity)^beta
        ratio = sums[valid] / capacity
        
        # Avoid division by zero or negative issues (gaps are >= 0)
        inv_gap_term = 1.0 / (gaps + epsilon) ** alpha
        fill_term = ratio ** beta
        
        raw_scores = inv_gap_term * fill_term
        
        # Assign to eta
        eta[valid] = raw_scores
        
        # Row Normalization:
        # For each row i, divide by the sum of valid entries in that row.
        # This makes the heuristic a probability distribution over valid next items.
        # If a row has no valid items, it stays at 1e-9 (handled by valid mask).
        row_sums = np.sum(eta, axis=1, keepdims=True)
        # Avoid division by zero for rows with no valid items (sum=0)
        row_sums[row_sums == 0] = 1.0
        
        eta = eta / row_sums
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
        
        # Ensure invalid entries remain 1e-9 (normalization shouldn't change them if sum was 0, 
        # but if sum > 0, they are 0/sum = 0. We need to set them back to 1e-9 if they became 0)
        # Actually, if valid[i,j] is False, eta[i,j] was 1e-9.
        # If row sum > 0, eta[i,j] becomes 1e-9 / row_sum. This is still ~1e-9.
        # If row sum == 0, we didn't divide.
        # So invalid entries are effectively 0 or ~1e-9.
        # The task says "Heuristic entries that are not positive become 1e-9".
        # 1e-9 / 100 is 1e-11, which is positive but very small.
        # To be safe and consistent with "not positive becomes 1e-9", 
        # we should probably keep the absolute scale or re-apply the floor.
        
        # Let's re-apply the floor to ensure compliance and numerical stability
        eta[eta < 1e-9] = 1e-9
        
    return eta
