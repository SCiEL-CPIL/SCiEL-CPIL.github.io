import time
import random
import numpy as np

N = 64

def construct():
        # Precompute coordinates
        coords = np.array([(x, y) for y in range(N) for x in range(N)])
        # indices 0 to N*N-1
        # x = idx % N, y = idx // N
        
        # Precompute squared distances matrix
        # Memory: 4096x4096 int32 ~ 64MB. OK.
        # Time: ~0.1s. OK.
        x = coords[:, 0].astype(np.int32)
        y = coords[:, 1].astype(np.int32)
        
        # dist2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
        # Using broadcasting
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist2 = dx**2 + dy**2
        
        num_points = N * N
        best_size = 0
        best_indices = []
        
        deadline = time.time() + 280 # 20s safety margin
        
        seed = 0
        while time.time() < deadline:
            random.seed(seed)
            candidates = list(range(num_points))
            random.shuffle(candidates)
            
            S = []
            # dist_sets[i] is a set of squared distances from i to other points in S
            dist_sets = {}
            
            for idx in candidates:
                # If we already have more than best, we can just keep going to see if we beat it
                # But checking validity is O(|S|)
                
                # 1. Check distances from idx to all p in S
                # We need to check if any distance is repeated (apex at idx)
                # And if any distance matches an existing distance from p (apex at p)
                
                # Optimization: if |S| is large, this loop is slow.
                # But |S| is expected to be small (< 50).
                
                if not S:
                    S.append(idx)
                    dist_sets[idx] = set()
                    continue
                
                # Get distances to existing points
                # dist2[idx, S] gives array of distances
                # We need to check uniqueness among these
                # And check against dist_sets[p]
                
                # Efficient way:
                # Create a set of distances from idx to S
                # If len(set) < len(S), conflict at idx.
                
                # Iterate to check conflicts at existing points
                valid = True
                dists_from_idx = []
                
                # Manual loop for speed in Python? 
                # NumPy indexing dist2[idx, S] is fast.
                dists_arr = dist2[idx, S]
                dists_set = set(dists_arr.tolist())
                
                if len(dists_set) < len(S):
                    valid = False
                else:
                    # Check if any dist is in dist_sets[p]
                    # We need to map dist to p? No, just check each p.
                    for i, p in enumerate(S):
                        d = dists_arr[i]
                        if d in dist_sets[p]:
                            valid = False
                            break
                
                if valid:
                    S.append(idx)
                    dist_sets[idx] = dists_set
                    # Update dist_sets for existing points
                    for i, p in enumerate(S[:-1]): # S[:-1] is old S
                        dist_sets[p].add(dists_arr[i])
                    
                    if len(S) > best_size:
                        best_size = len(S)
                        best_indices = S.copy()
            
            seed += 1
            
        # Convert indices back to coordinates
        return [tuple(coords[i]) for i in best_indices]
