# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in 11 dimensions satisfying:
    1. No point is zero.
    2. min_{i!=j} ||p_i - p_j||^2 >= max_i ||p_i||^2
    
    Strategy: Search for a large subset of weight-3 vectors (norm^2 = 3).
    Constraint becomes: p_i . p_j <= 1 for all i != j.
    We use a greedy construction followed by simulated annealing to maximize k.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    d = 11
    
    # Generate all weight-3 vectors
    # C(11,3) = 165 combinations of indices. 2^3 = 8 sign combinations.
    # Total 1320 vectors.
    all_weight3 = []
    for comb in combinations(range(d), 3):
        for mask in range(8):
            p = np.zeros(d, dtype=np.int64)
            for idx, c in enumerate(comb):
                p[c] = 1 if (mask >> idx) & 1 else -1
            all_weight3.append(p)
    
    all_weight3 = np.array(all_weight3, dtype=np.int64)
    n_candidates = len(all_weight3)
    
    # Precompute dot products? 
    # Dot product matrix D where D[i,j] = all_weight3[i] . all_weight3[j]
    # This is 1320 x 1320. 1.7M entries. Manageable.
    # But we only care about D[i,j] <= 1.
    # Actually, we can just compute on the fly or precompute a compatibility matrix.
    # Let's precompute a boolean matrix 'compat' where compat[i,j] is True if 
    # all_weight3[i] . all_weight3[j] <= 1.
    
    # Optimization: Use matrix multiplication for dot products
    # D = all_weight3 @ all_weight3.T
    # This might be slow if done repeatedly, but once is fine.
    # 1320x1320 matrix mult is fast in NumPy.
    
    D = all_weight3 @ all_weight3.T
    
    # Condition for validity of a set S:
    # For all i, j in S, i != j, D[i,j] <= 1.
    # Note: D[i,i] = 3.
    
    # We want to find the largest subset S of indices such that 
    # max_{i,j in S, i!=j} D[i,j] <= 1.
    
    # Initialize with a greedy approach
    # Start with an empty set, add vectors that are compatible with all existing ones.
    # To break ties, we might randomize the order.
    
    def greedy_construct(order):
        selected = []
        selected_set = set()
        for idx in order:
            # Check compatibility with all currently selected
            if not selected:
                selected.append(idx)
                selected_set.add(idx)
                continue
            
            compatible = True
            for s_idx in selected:
                if D[idx, s_idx] > 1:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
        
        return selected

    # Try multiple random orders to find a good starting set
    best_selected = []
    best_size = 0
    
    # Run several greedy passes
    for _ in range(50):
        if time.time() > deadline - 100:
            break
        order = list(range(n_candidates))
        random.shuffle(order)
        selected = greedy_construct(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
    # Now, use Simulated Annealing to try to improve the set size
    # State: list of selected indices
    # Move: Remove one, add one (swap), or just add one (if possible), or remove one (to allow future adds)
    
    current_selected = best_selected[:]
    current_size = best_size
    
    # Helper to check if a vector can be added to a set
    # We maintain a set of selected indices for O(1) lookup? 
    # For SA, we need to evaluate moves quickly.
    # Let's keep a list and a set.
    
    def can_add(idx, sel_set):
        for s in sel_set:
            if D[idx, s] > 1:
                return False
        return True

    sel_set = set(current_selected)
    
    # SA Parameters
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    # We want to maximize size.
    # Move types:
    # 1. Add a random unselected vector if compatible. (Increases size by 1)
    # 2. Remove a random selected vector. (Decreases size by 1)
    # 3. Swap: Remove one, Add one. (Size unchanged)
    
    # To allow growth, we prioritize "Add" moves.
    # If we can add, we always do it (greedy step within SA).
    
    # Precompute list of all indices for random selection
    all_indices = list(range(n_candidates))
    
    while time.time() < deadline - 20:
        if T < min_T:
            T = min_T
            
        # 1. Try to add a random vector
        # Pick a random unselected vector
        # Efficient way: pick random index, if in sel_set, skip or pick another.
        # Since n_candidates is 1320 and sel_size is ~300-500, rejection is low.
        
        move_made = False
        
        # Try a few random additions
        for _ in range(10):
            if time.time() > deadline - 20: break
            idx = random.choice(all_indices)
            if idx in sel_set:
                continue
            
            if can_add(idx, sel_set):
                # Accept addition
                current_selected.append(idx)
                sel_set.add(idx)
                current_size += 1
                move_made = True
                break
        
        if move_made:
            # If we added, we might want to restart T or just continue
            # Adding is always an improvement, so we don't need to "cool" as fast?
            # But we also need to explore.
            # Let's just continue.
            continue
            
        # 2. Try a swap (Remove one, Add one) to enable future additions or just explore
        # Or try to remove one to see if a better configuration emerges?
        # Usually, we want to keep size high.
        # Let's try: Remove a random selected vector, then try to add a random unselected one.
        # If the new set is valid, accept.
        
        if len(current_selected) > 0:
            # Pick a vector to remove
            remove_idx = random.choice(current_selected)
            # Pick a vector to add
            add_idx = random.choice(all_indices)
            
            if add_idx in sel_set:
                # If we pick a selected one, it's not a swap, it's a no-op or error.
                # Let's ensure add_idx != remove_idx and add_idx not in sel_set
                # Actually, if add_idx is in sel_set, we can't add it.
                # So we need add_idx not in sel_set.
                pass
            
            # Re-pick add_idx if it's in sel_set
            while add_idx in sel_set:
                add_idx = random.choice(all_indices)
            
            # Check if removing remove_idx and adding add_idx is valid
            # New set = (current - {remove_idx}) U {add_idx}
            # We need to check add_idx against (current - {remove_idx})
            
            valid = True
            for s in current_selected:
                if s == remove_idx:
                    continue
                if D[add_idx, s] > 1:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                # Replace remove_idx with add_idx in list
                for i, v in enumerate(current_selected):
                    if v == remove_idx:
                        current_selected[i] = add_idx
                        break
                sel_set.remove(remove_idx)
                sel_set.add(add_idx)
                # Size unchanged
                # Accept? 
                # In SA, we might accept even if it doesn't improve size, 
                # to escape local minima where no single add is possible.
                # But here, if valid, it's a neutral move.
                # We should accept with probability 1 if it opens up possibilities?
                # Hard to know. Let's accept all valid swaps to explore the space.
                move_made = True
                
        if not move_made:
            # No move made, cool down
            T *= cooling
            
    # Final verification and formatting
    if not current_selected:
        # Fallback to a simple valid set
        # e.g. 2 points
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result_points = all_weight3[current_selected]
    
    # Verify validity one last time to be safe
    k = len(result_points)
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    norms_sq = np.sum(result_points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    valid = True
    for i in range(k):
        for j in range(i+1, k):
            diff = result_points[i] - result_points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If our SA produced an invalid set (shouldn't happen), fallback
        # Find the largest valid subset? Too slow.
        # Just return a small valid set.
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    return np.around(result_points).astype(np.int64)

# EVOLVE-BLOCK-END
