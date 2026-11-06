import numpy as np
import time
import random

N = 64
NUM_POINTS = N * N
MAX_DIST2 = 2 * (N - 1) ** 2

# Pre-compute squared distances
# Using uint16 to save memory and speed up access
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
xs = points[:, 0]
ys = points[:, 1]

# Calculate distance matrix
# dist2[i, j] = (xs[i] - xs[j])^2 + (ys[i] - ys[j])^2
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.uint16)

def construct():
    """
    Find a large set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    Uses a bitset-accelerated greedy + local search (kicks) approach.
    """
    deadline = time.time() + 280  # Leave margin
    
    best_size = 0
    best_active = []
    
    # Helper functions using closures for speed
    def try_add(p, selected, masks, active):
        if selected[p]:
            return False
        # Check if p conflicts with any active point
        # Conflict if for any a in active, dist2[a, p] is already in masks[a]
        for a in active:
            d2 = dist2[a, p]
            if (masks[a] >> d2) & 1:
                return False
        return True

    def do_add(p, selected, masks, active):
        selected[p] = True
        mp = 0
        for a in active:
            d2 = dist2[a, p]
            masks[a] |= (1 << d2)
            mp |= (1 << d2)
        masks[p] = mp
        active.append(p)

    def do_remove(p, selected, masks, active):
        selected[p] = False
        # Find and remove from active
        # Since active is small, linear search is okay, but we can optimize
        # We assume p is in active
        for i, a in enumerate(active):
            if a == p:
                active.pop(i)
                break
        
        # Update masks
        for a in active:
            d2 = dist2[a, p]
            masks[a] &= ~(1 << d2)
        
        masks[p] = 0

    random.seed(42)
    
    # Pre-generate a few random permutations to reuse? 
    # Or just generate on the fly.
    
    while time.time() < deadline - 10:
        # 1. Greedy Initialization
        selected = np.zeros(NUM_POINTS, dtype=bool)
        masks = [0] * NUM_POINTS
        active = []
        
        order = np.random.permutation(NUM_POINTS)
        
        for p in order:
            if time.time() > deadline - 10:
                break
            if not selected[p]:
                if try_add(p, selected, masks, active):
                    do_add(p, selected, masks, active)
        
        current_size = len(active)
        if current_size > best_size:
            best_size = current_size
            best_active = active.copy()
        
        # 2. Local Search with Kicks
        # We try to improve the current solution by removing some points and re-filling greedily.
        
        # To speed up re-filling, we can use a static random order or generate one.
        # Generating a permutation of 4096 elements takes ~1ms.
        
        improved = True
        # Limit number of kick attempts per restart to avoid spending too much time on one bad restart
        max_kicks = 50 
        
        kick_count = 0
        while improved and kick_count < max_kicks and time.time() < deadline - 10:
            improved = False
            kick_count += 1
            
            # Decide kick size: 1 or 2
            kick_size = 1 if random.random() < 0.7 else 2
            
            if len(active) < kick_size:
                continue
            
            # Backup state
            old_selected = selected.copy()
            old_masks = masks.copy()
            old_active = active.copy()
            
            # Remove random points
            remove_indices = random.sample(range(len(active)), kick_size)
            remove_indices.sort(reverse=True) # Remove from end to preserve indices
            
            for idx in remove_indices:
                p_remove = old_active[idx]
                do_remove(p_remove, selected, masks, active)
            
            # Greedily add points
            # Use a new random permutation
            order2 = np.random.permutation(NUM_POINTS)
            
            for p in order2:
                if time.time() > deadline - 10:
                    break
                if not selected[p]:
                    if try_add(p, selected, masks, active):
                        do_add(p, selected, masks, active)
            
            new_size = len(active)
            
            if new_size > current_size:
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_active = active.copy()
            else:
                # Revert
                selected[:] = old_selected
                masks = old_masks
                active = old_active
        
        # Optional: Polish the best solution found so far
        # (Already done in the greedy phase of the last successful kick or init)
        
    result_idx = np.array(best_active)
    result = [(int(points[i, 0]), int(points[i, 1])) for i in result_idx]
    return result

# EVOLVE-BLOCK-END
