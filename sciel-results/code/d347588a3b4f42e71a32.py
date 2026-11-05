import random
import numpy as np

N = 64
MAX_DIST = 2 * (N - 1) ** 2

# EVOLVE-BLOCK-START
def construct():
    # Generate all candidate points
    points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    # Use multiple random restarts to find a larger set
    # Vectorized bitset approach allows for more restarts
    num_restarts = 200
    
    for _ in range(num_restarts):
        random.shuffle(points)
        S = []
        # used_bool tracks which squared distances have been seen between any pair in S
        used_bool = np.zeros(MAX_DIST + 1, dtype=bool)
        
        for p in points:
            px, py = p
            
            if not S:
                S.append(p)
                continue
            
            # Convert current set to array for vectorized distance calculation
            S_arr = np.array(S)
            p_arr = np.array([px, py])
            
            # Calculate squared distances from p to all points in S
            dists = np.sum((S_arr - p_arr) ** 2, axis=1)
            
            # Check for duplicates in new distances (apex at p)
            # and intersection with used distances (apex at existing points)
            # Using bincount for efficiency
            counts = np.bincount(dists, minlength=MAX_DIST + 1)
            
            # If any count > 1, there is a duplicate distance in the new set
            if np.any(counts > 1):
                continue
            
            # If any distance in dists is already in used_bool, reject
            # We can check this by seeing if any of the indices in dists have True in used_bool
            if np.any(used_bool[dists]):
                continue
            
            # Valid point
            S.append(p)
            # Update used_bool
            used_bool[dists] = True
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # Early stopping if we reach a high value
        if best_size > 130:
            break

    return best_set
# EVOLVE-BLOCK-END
