import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit
    
    # Moduli in range [20, 200]
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    
    # Precompute max possible sum from index i to end
    # This helps with pruning
    max_suffix_sum = [0.0] * (n_mods + 1)
    for i in range(n_mods - 1, -1, -1):
        max_suffix_sum[i] = max_suffix_sum[i + 1] + 1.0 / moduli[i]
        
    best_score = 0.0
    best_classes = []
    
    # To speed up compatibility check, we can store current classes as list of (m, r)
    # and their gcds with future m? No, just compute gcd on the fly.
    
    def is_compatible(new_m, new_r, current_classes):
        for m, r in current_classes:
            g = gcd(new_m, m)
            if new_r % g == r % g:
                return False
        return True
        
    def backtrack(idx, current_classes, current_score):
        nonlocal best_score, best_classes
        
        # Check time limit
        if time.time() > deadline:
            return
            
        # Update best if current is better
        if current_score > best_score:
            best_score = current_score
            best_classes = list(current_classes)
            
        # If we've considered all moduli, we are done with this branch
        if idx == n_mods:
            return
            
        # Pruning: if even taking all remaining moduli doesn't beat best_score, stop
        if current_score + max_suffix_sum[idx] <= best_score:
            return
            
        m = moduli[idx]
        
        # Option 1: Include this modulus m
        # Try all residues
        # Optimization: If current_classes is not empty, we might filter residues faster?
        # For now, just iterate.
        for r in range(m):
            if is_compatible(m, r, current_classes):
                current_classes.append((m, r))
                backtrack(idx + 1, current_classes, current_score + 1.0 / m)
                current_classes.pop()
                
                # Optional: If we found a valid residue, do we need to try others?
                # Yes, because different residues might allow different future moduli.
                # However, if we just want *any* solution, we could stop. 
                # But we want to *maximize* the sum, so we must explore.
                
        # Option 2: Skip this modulus m
        backtrack(idx + 1, current_classes, current_score)

    backtrack(0, [], 0.0)
    
    # Return in required format [[m, r], ...]
    return [[m, r] for m, r in best_classes]
