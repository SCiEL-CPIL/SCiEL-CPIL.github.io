import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

MAX_DIST_SQ = int(np.max(dist_sq))

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_size = NUM_POINTS + 1
    best_set = None
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # 1. Initial Greedy Construction
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S_indices = []
        used_dist_local = np.zeros((NUM_POINTS, MAX_DIST_SQ + 1), dtype=bool)
        
        for p in order:
            if not S_indices:
                S_indices.append(p)
                continue
            
            dists = dist_sq[p, S_indices]
            
            # Check uniqueness
            if len(dists) != len(np.unique(dists)):
                continue
            
            # Check conflicts
            if np.any(used_dist_local[S_indices, dists]):
                continue
            
            S_indices.append(p)
            used_dist_local[p, dists] = True
            used_dist_local[S_indices[:-1], dists] = True
        
        current_set = S_indices
        current_size = len(current_set)
        
        if current_size < best_size:
            best_size = current_size
            best_set = current_set.copy()
            if best_size <= 10: # Very good, might break early
                break

        # 2. Local Search / Minimization
        # Try to remove points and refill to see if we get a smaller maximal set
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle the set to try removing in random order
            removal_order = current_set.copy()
            random.shuffle(removal_order)
            
            for p in removal_order:
                if time.time() > deadline:
                    break
                
                # Create reduced set
                S_reduced = [x for x in current_set if x != p]
                
                # Try to maximize S_reduced, but stop if size >= current_size
                # We want a set strictly smaller than current_size
                max_allowed = current_size - 1
                
                # Greedy Completion
                S_new = list(S_reduced)
                used_dist_new = np.zeros((NUM_POINTS, MAX_DIST_SQ + 1), dtype=bool)
                
                # Initialize used_dist for S_reduced
                if S_reduced:
                    # For each pair in S_reduced, mark distances
                    # This is O(K^2) where K is len(S_reduced)
                    for i in range(len(S_reduced)):
                        for j in range(i + 1, len(S_reduced)):
                            d = dist_sq[S_reduced[i], S_reduced[j]]
                            used_dist_new[S_reduced[i], d] = True
                            used_dist_new[S_reduced[j], d] = True
                
                # Candidates: all points not in S_new
                candidates = [x for x in range(NUM_POINTS) if x not in S_new]
                
                # Heuristic: Sort candidates by number of conflicts with S_new
                # A point with more conflicts is "harder" to place, so place it first?
                # Or "easier" to place?
                # If we place "hard" points first, they might block many others.
                # Let's calculate conflict count for each candidate.
                # Conflict count = number of points b in S_new such that d(c, b) is already used by b?
                # Or simply distance to S_new?
                # Let's use: number of b in S_new such that d(c,b) is in used_dist_new[b]
                # This is expensive.
                # Simpler heuristic: Random order is often robust.
                # Let's stick to random for now, but maybe sort by distance to centroid?
                # Actually, let's just use random order for the completion.
                random.shuffle(candidates)
                
                for c in candidates:
                    if len(S_new) >= max_allowed:
                        break # No improvement possible
                    
                    dists = dist_sq[c, S_new]
                    
                    # Check uniqueness
                    if len(dists) != len(np.unique(dists)):
                        continue
                    
                    # Check conflicts
                    if np.any(used_dist_new[S_new, dists]):
                        continue
                    
                    # Add c
                    S_new.append(c)
                    used_dist_new[c, dists] = True
                    used_dist_new[S_new[:-1], dists] = True
                
                # After completion, check if S_new is smaller
                if len(S_new) < current_size:
                    current_set = S_new
                    current_size = len(S_new)
                    improved = True
                    break # Restart inner loop with new set
            
            # Update global best
            if current_size < best_size:
                best_size = current_size
                best_set = current_set.copy()
                if best_size <= 10:
                    break

    if best_set is None:
        return []
    
    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
