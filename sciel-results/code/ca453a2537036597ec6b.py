import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 140  # Safety margin
    
    candidates = []
    
    # Type 1: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 2: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Conflict if dot > 2 (since norm^2 is 4 for all, dist^2 = 4+4-2dot = 8-2dot >= 4 => dot <= 2)
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)
    
    # Convert to list of conflicts for faster lookup
    # conflict_adj[i] is a set of j such that conflict[i,j] is True
    # But for speed, we can use the matrix directly with numpy operations if we batch them.
    # However, for single node checks, a list of lists or a set is faster.
    # Given n=5302, a set of ints is fine.
    conflict_sets = [set(np.where(conflict[i])[0]) for i in range(n)]
    
    best_size = 0
    best_selected = []
    rng = np.random.RandomState(42)
    
    # Multiple random restarts
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        order = rng.permutation(n)
        selected = []
        selected_set = set()
        
        # Greedy Phase
        for idx in order:
            if time.time() > deadline:
                break
            # Check if idx conflicts with any in selected
            # If selected is empty, add
            # Otherwise, check intersection
            if not selected_set:
                selected.append(idx)
                selected_set.add(idx)
            else:
                # Check if any neighbor of idx is in selected_set
                # conflict_sets[idx] contains neighbors
                # We can check if len(conflict_sets[idx].intersection(selected_set)) == 0
                # This might be slow if sets are large.
                # Alternative: maintain a count of conflicts for each unselected node?
                # For now, intersection is O(size of smaller set).
                if not conflict_sets[idx].intersection(selected_set):
                    selected.append(idx)
                    selected_set.add(idx)
                    
        size = len(selected)
        if size > best_size:
            best_size = size
            best_selected = selected.copy()
            
        # Local Search Phase
        # Add phase: add any compatible node
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            unselected = [i for i in range(n) if i not in selected_set]
            # To speed up, we can just iterate through all and check
            # But checking all 5300 every iteration is slow.
            # Better: maintain a list of "free" nodes?
            # For simplicity, just check a random subset or all if time permits.
            # Let's try adding one by one.
            for idx in unselected:
                if not conflict_sets[idx].intersection(selected_set):
                    selected.append(idx)
                    selected_set.add(idx)
                    improved = True
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = selected.copy()
                    # After adding, some other nodes might become incompatible, but we only add, so no removals.
                    # So "improved" loop can just continue scanning?
                    # Actually, if we add idx, it doesn't make other nodes *more* compatible.
                    # So we can break the inner loop and restart the scan?
                    # Or just continue scanning the remaining unselected.
                    # Since we modified selected_set, subsequent checks are valid.
        
        # Swap phase: try to replace one selected with one unselected
        # This is complex. Let's try a simple "kick-out" heuristic.
        # Pick a random selected node, remove it, then try to add as many as possible?
        # This is essentially restarting the greedy from a perturbed state.
        
        if size >= best_size - 5:
             # Try a few swap moves
             for _ in range(10):
                 if time.time() > deadline:
                     break
                 if not selected:
                     break
                 idx_to_remove = rng.choice(selected)
                 selected.remove(idx_to_remove)
                 selected_set.remove(idx_to_remove)
                 
                 # Try to add compatible nodes
                 added_any = False
                 unselected = [i for i in range(n) if i not in selected_set]
                 for idx in unselected:
                     if not conflict_sets[idx].intersection(selected_set):
                         selected.append(idx)
                         selected_set.add(idx)
                         added_any = True
                         
                 if len(selected) > best_size:
                     best_size = len(selected)
                     best_selected = selected.copy()
                     
                 # If we didn't add anything, or added less than removed, revert?
                 # For now, keep the state.
    
    return candidates[best_selected]
