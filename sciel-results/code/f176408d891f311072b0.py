import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 110  # Safety margin
    
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute conflict matrix: conflict if dist_sq < 4 (since max norm will be 2 if we include any norm-2 vector)
    # But we need to be careful: if we only select norm-sqrt(2) vectors, the constraint is looser.
    # For safety, use dist_sq < 4 as conflict (strictest case)
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Convert to adjacency list for faster operations
    # But for now, use the matrix
    
    best_selected = None
    best_size = 0
    np.random.seed(42)
    
    def greedy_mis(conflict, n, seed):
        """Greedy maximum independent set with randomization"""
        rng = np.random.RandomState(seed)
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = rng.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            neigh_mask = conflict[chosen_global]
            to_remove = neigh_mask.copy()
            to_remove[chosen_global] = True
            
            actual_remove = to_remove & remaining
            if np.any(actual_remove):
                # Update degrees
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
            
            remaining[actual_remove] = False
        
        return selected
    
    def local_search(conflict, n, selected, seed, max_iters=1000):
        """Improve solution via swap-based local search"""
        rng = np.random.RandomState(seed)
        selected_set = set(selected)
        current = list(selected)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try swapping one selected with one unselected
            for _ in range(50):
                if time.time() > deadline:
                    break
                idx = rng.randint(len(current))
                to_remove = current[idx]
                
                # Find candidates to add
                unselected = [i for i in range(n) if i not in selected_set]
                if not unselected:
                    break
                    
                # Try random unselected
                to_add = rng.choice(unselected)
                
                # Check if to_add conflicts with any in (current minus to_remove)
                others = [x for x in current if x != to_remove]
                if any(conflict[to_add, x] for x in others):
                    continue
                
                # Check if to_remove was not needed (i.e., removing it doesn't violate anything)
                # Actually we're removing it, so we just need to_add to not conflict with others
                # And to_remove was in the set, so we need to make sure removing it is fine
                # (it was valid before, so removing it is always fine)
                
                # Do the swap
                current[idx] = to_add
                selected_set.discard(to_remove)
                selected_set.add(to_add)
                improved = True
                break
            
            if not improved:
                # Try adding without removing (if possible)
                unselected = [i for i in range(n) if i not in selected_set]
                if not unselected:
                    break
                added = False
                for _ in range(20):
                    to_add = rng.choice(unselected)
                    if not any(conflict[to_add, x] for x in current):
                        current.append(to_add)
                        selected_set.add(to_add)
                        added = True
                        break
                if not added:
                    break
        
        return current
    
    # Run multiple greedy restarts
    num_restarts = 50
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        selected = greedy_mis(conflict, n, seed=42 + attempt)
        size = len(selected)
        
        if size > best_size:
            best_size = size
            best_selected = selected
        
        # Local search to improve
        if size >= best_size - 5:  # Only improve promising solutions
            improved = local_search(conflict, n, selected, seed=1000 + attempt, max_iters=500)
            if len(improved) > best_size:
                best_size = len(improved)
                best_selected = improved
    
    result = candidates[best_selected]
    return result
