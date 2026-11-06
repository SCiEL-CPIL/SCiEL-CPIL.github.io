import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcd
    gcds = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            g = math.gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g

    # State:
    # selected: list of (m, r)
    # available: list of lists of available residues for each modulus?
    # Better: maintain `blocked` status dynamically.
    
    # We will use a recursive backtracking solver.
    # To speed it up, we will precompute for each pair (m, m') the list of residues r' in m' 
    # that are incompatible with a given residue r in m?
    # No, that's too big.
    
    # Instead, we maintain an array `residue_valid[m][r]` which is True if r is valid for m given current selections.
    # When we add (m, r), we update residue_valid for all other moduli.
    
    def solve(restart_seed=0):
        random.seed(restart_seed)
        
        # Initialize validity
        # valid[m] is a set of valid residues for m
        # Initially all residues 0..m-1 are valid
        valid = [set(range(m)) for m in range(max_mod + 1)]
        
        # Order of moduli to try. 
        # We can shuffle the moduli to break symmetry.
        # But MRV will reorder dynamically.
        
        best_solution = []
        
        def backtrack(selected, used_moduli):
            nonlocal best_solution
            if time.time() > deadline:
                return
            
            if len(selected) > len(best_solution):
                best_solution = selected[:]
                
            if len(used_moduli) == len(moduli):
                return
                
            # MRV: Find modulus with fewest valid residues
            min_count = float('inf')
            best_m = None
            candidates = None
            
            # Optimization: Prune if we can't beat best
            if len(selected) + (len(moduli) - len(used_moduli)) <= len(best_solution):
                return

            for m in moduli:
                if m in used_moduli:
                    continue
                count = len(valid[m])
                if count == 0:
                    continue
                if count < min_count:
                    min_count = count
                    best_m = m
                    candidates = list(valid[m])
                    if min_count == 1:
                        break
            
            if best_m is None:
                return
            
            # Randomize candidates to explore different branches
            random.shuffle(candidates)
            
            for r in candidates:
                # Try (best_m, r)
                # Check if r is still valid (it should be, but double check)
                if r not in valid[best_m]:
                    continue
                
                # Save state
                # We need to save the valid sets for all moduli?
                # That's expensive.
                # Alternative: Use a stack of changes.
                # Or, since we are doing restarts, and depth is <= 127,
                # copying the `valid` list of sets might be okay?
                # 127 sets, each size up to 128.
                # Copying is O(127 * 128) ~ 16k operations.
                # With 127 depth, total copy cost is high.
                
                # Better: Undo log.
                # When we add (m, r), we modify valid[m'] for m' != m.
                # We record which (m', r') were removed.
                
                changes = []
                for m2 in moduli:
                    if m2 == best_m or m2 in used_moduli:
                        continue
                    g = gcds[best_m][m2]
                    # Residues r2 in valid[m2] such that r2 % g == r % g are INVALID
                    # We need to remove them.
                    target_residue = r % g
                    to_remove = []
                    for r2 in valid[m2]:
                        if r2 % g == target_residue:
                            to_remove.append(r2)
                    if to_remove:
                        for r2 in to_remove:
                            valid[m2].remove(r2)
                            changes.append((m2, r2))
                            
                # Recurse
                new_selected = selected + [(best_m, r)]
                new_used = used_moduli | {best_m}
                backtrack(new_selected, new_used)
                
                # Undo
                for m2, r2 in changes:
                    valid[m2].add(r2)
                    
                if time.time() > deadline:
                    return

        backtrack([], set())
        return best_solution

    # Run multiple restarts
    best_overall = []
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        sol = solve(i)
        if len(sol) > len(best_overall):
            best_overall = sol
            
    # Convert to required format
    result = [[r, m] for m, r in best_overall]
    
    if not result:
        # Fallback
        result = [[0, 2]]
        
    return result
