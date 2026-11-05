import time
import random
import numpy as np

N = 64
TOTAL_POINTS = N * N
# Deadline guard
START_TIME = time.time()
DEADLINE = 325.0  # 5s safety margin

# Precompute squared distances
# Using int32 to save memory, max dist is 63^2 + 63^2 = 7938, fits in int16, but int32 is safe.
# Actually, max dist is (63)^2 + (63)^2 = 7938.
dist_matrix = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)

def _precompute_distances():
    global dist_matrix
    coords = np.array([(x, y) for y in range(N) for x in range(N)])
    # Calculate pairwise squared distances
    # Broadcasting: (4096, 1, 2) - (1, 4096, 2) -> (4096, 4096, 2)
    # This might be memory heavy (4096*4096*2*4 bytes = 128MB).
    # Let's do it in chunks or just loop. 4096*4096 = 16M ops. Python loop is slow.
    # Vectorized:
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    # dx = x_i - x_j
    # d2 = (x_i - x_j)^2 + (y_i - y_j)^2
    # We can compute this using outer products
    x_outer = x_coords[:, None] - x_coords[None, :]
    y_outer = y_coords[:, None] - y_coords[None, :]
    dist_matrix = (x_outer ** 2 + y_outer ** 2).astype(np.int32)

_precompute_distances()

def _greedy_init():
    """Generate a valid set using greedy addition."""
    S = set()
    dist_sets = {i: set() for i in range(TOTAL_POINTS)} # Only needed for points in S, but let's keep it simple
    # Actually, we only need dist_sets for points in S.
    
    candidates = list(range(TOTAL_POINTS))
    random.shuffle(candidates)
    
    for P in candidates:
        if time.time() - START_TIME > DEADLINE:
            break
        # Check if P can be added
        valid = True
        seen_dists = set()
        for Q in S:
            d = dist_matrix[P, Q]
            if d in seen_dists:
                valid = False
                break
            seen_dists.add(d)
            if d in dist_sets[Q]:
                valid = False
                break
        
        if valid:
            S.add(P)
            # Update dist_sets for existing points
            for Q in S:
                if Q != P:
                    dist_sets[Q].add(dist_matrix[P, Q])
            dist_sets[P] = seen_dists.copy()
            
    return S, dist_sets

def _try_add(P, S, dist_sets):
    """Try to add point P to set S. Returns True if added."""
    if P in S:
        return False
    
    seen_dists = set()
    for Q in S:
        d = dist_matrix[P, Q]
        if d in seen_dists:
            return False
        seen_dists.add(d)
        if d in dist_sets[Q]:
            return False
            
    S.add(P)
    for Q in S:
        if Q != P:
            dist_sets[Q].add(dist_matrix[P, Q])
    dist_sets[P] = seen_dists.copy()
    return True

def _remove(P, S, dist_sets):
    """Remove point P from set S."""
    if P not in S:
        return
    for Q in S:
        if Q != P:
            dist_sets[Q].discard(dist_matrix[P, Q])
    S.remove(P)
    del dist_sets[P]

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_S, best_dist_sets = _greedy_init()
    best_size = len(best_S)
    
    # Local Search with Batch Swaps
    # We maintain current S and dist_sets
    current_S, current_dist_sets = best_S.copy(), {k: v.copy() for k, v in best_dist_sets.items()}
    current_size = len(current_S)
    
    # List of all points
    all_points = list(range(TOTAL_POINTS))
    
    # We will try to improve current_S
    # Strategy:
    # 1. Try to add any point.
    # 2. If stuck, try batch swaps (remove k, add k+1).
    
    stall_count = 0
    max_stall = 1000
    
    while time.time() - START_TIME < DEADLINE:
        improved = False
        
        # Try to add a random point
        random.shuffle(all_points)
        for P in all_points:
            if time.time() - START_TIME > DEADLINE:
                break
            if P in current_S:
                continue
            if _try_add(P, current_S, current_dist_sets):
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S.copy()
                    best_dist_sets = {k: v.copy() for k, v in current_dist_sets.items()}
                break # Restart add loop after successful add
        
        if improved:
            stall_count = 0
            continue
            
        # If no single add worked, try batch swaps
        # Start with small batch sizes
        batch_size = 1
        max_batch = 3
        
        swapped = False
        while batch_size <= max_batch:
            if time.time() - START_TIME > DEADLINE:
                break
            
            # Pick batch_size points to remove
            # Prefer points that are "expensive" (have many conflicts) or just random
            # Random is simpler
            candidates_to_remove = list(current_S)
            if len(candidates_to_remove) < batch_size:
                break
            random.shuffle(candidates_to_remove)
            to_remove = candidates_to_remove[:batch_size]
            
            # Save state
            saved_S = current_S.copy()
            saved_dist_sets = {k: v.copy() for k, v in current_dist_sets.items()}
            
            # Remove
            for P in to_remove:
                _remove(P, current_S, current_dist_sets)
                
            # Try to add batch_size + 1 points
            added_count = 0
            # Shuffle all points not in current_S
            candidates_to_add = [P for P in all_points if P not in current_S]
            random.shuffle(candidates_to_add)
            
            for P in candidates_to_add:
                if time.time() - START_TIME > DEADLINE:
                    break
                if _try_add(P, current_S, current_dist_sets):
                    added_count += 1
                    if added_count >= batch_size + 1:
                        break
                        
            # Evaluate
            if added_count > len(to_remove):
                # Accept the swap
                current_size += (added_count - len(to_remove))
                swapped = True
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S.copy()
                    best_dist_sets = {k: v.copy() for k, v in current_dist_sets.items()}
                break # Break out of batch_size loop, restart outer loop
            else:
                # Reject, restore state
                current_S = saved_S
                current_dist_sets = saved_dist_sets
                # Try next batch size or different removal set
                # To avoid infinite loops on same bad removals, we can just increment batch_size
                batch_size += 1
                # Or restart the removal selection with a new random seed
                # Let's just increment batch size for now.
                # If batch_size > max_batch, we will exit the while loop.
        
        if swapped:
            stall_count = 0
            continue
            
        # If no swap worked
        stall_count += 1
        if stall_count > max_stall:
            # Restart with new greedy
            current_S, current_dist_sets = _greedy_init()
            current_size = len(current_S)
            if current_size > best_size:
                best_size = current_size
                best_S = current_S.copy()
                best_dist_sets = {k: v.copy() for k, v in current_dist_sets.items()}
            stall_count = 0
            max_stall = 1000 # Reset stall limit

    return [(p % N, p // N) for p in best_S]

