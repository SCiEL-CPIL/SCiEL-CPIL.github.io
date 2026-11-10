import time
import math
import random

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we return within 60s wall-clock
    # The evaluator kills at 90s, but previous run failed at 60s limit? 
    # "candidate exceeded 60 s" suggests the limit might be tighter or overhead was high.
    # Let's use 40s for the algorithm to be safe.
    deadline = start_time + 40.0
    MAX_MOD = 128
    
    # Candidate moduli: All even numbers from 2 to 128
    # This set has the highest potential for disjoint classes because gcds are large.
    moduli = list(range(2, MAX_MOD + 1, 2))
    n_mods = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    def is_valid(residues, indices):
        """Check if the subset of residues at given indices is valid."""
        n = len(indices)
        for i in range(n):
            idx_i = indices[i]
            r_i = residues[idx_i]
            for j in range(i + 1, n):
                idx_j = indices[j]
                r_j = residues[idx_j]
                g = gcds[idx_i][idx_j]
                if r_i % g == r_j % g:
                    return False
        return True

    def count_conflicts(residues, indices):
        """Count number of conflicting pairs in the subset."""
        conflicts = 0
        n = len(indices)
        for i in range(n):
            idx_i = indices[i]
            r_i = residues[idx_i]
            for j in range(i + 1, n):
                idx_j = indices[j]
                r_j = residues[idx_j]
                g = gcds[idx_i][idx_j]
                if r_i % g == r_j % g:
                    conflicts += 1
        return conflicts

    # Strategy 1: Greedy construction with random restarts
    # We try to build a large valid set by adding moduli one by one.
    best_size = 0
    best_solution = []
    
    # We will run several random restarts of the greedy process
    # Ordering of moduli matters for greedy.
    
    for restart in range(100):
        if time.time() > deadline - 5.0:
            break
            
        # Randomly shuffle the order in which we consider moduli
        order = list(range(n_mods))
        random.shuffle(order)
        
        chosen_indices = []
        chosen_residues = [0] * n_mods # We only care about residues for chosen ones
        
        for idx in order:
            m = moduli[idx]
            found_r = None
            
            # Try to find a residue r for m that is disjoint from all chosen
            # To make it faster, we can just try random residues or 0..m-1
            # Since m <= 128, trying all is cheap.
            for r in range(m):
                # Check against all chosen
                ok = True
                for c_idx in chosen_indices:
                    g = gcds[idx][c_idx]
                    if r % g == chosen_residues[c_idx] % g:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            
            if found_r is not None:
                chosen_indices.append(idx)
                chosen_residues[idx] = found_r
        
        # Evaluate this solution
        size = len(chosen_indices)
        if size > best_size:
            best_size = size
            # Construct the result list
            sol = []
            for idx in chosen_indices:
                sol.append([chosen_residues[idx], moduli[idx]])
            best_solution = sol
            
        # If we found the maximum possible (all 64), we can stop early
        if best_size == n_mods:
            break

    # Strategy 2: If we have a good solution, try to improve it with local search?
    # Or if greedy didn't find 64, maybe local search on the full set can find 64?
    # The previous local search was slow. Let's try a very fast local search on the full set
    # only if we have time left.
    
    if time.time() < deadline - 10.0 and best_size < n_mods:
        # Try to solve for all 64 moduli using a fast local search
        # Initialize with the best greedy solution extended with random residues for missing ones?
        # Or just random start.
        
        # Let's try to improve the best greedy solution by adding missing moduli?
        # Actually, if greedy failed to add a modulus, it's because no residue worked.
        # Changing residues of existing ones might free up a slot.
        
        # Let's do a local search on the full set of 64 moduli.
        # Start with random residues.
        residues = [random.randint(0, m - 1) for m in moduli]
        indices = list(range(n_mods))
        
        conflicts = count_conflicts(residues, indices)
        
        # Simple hill climbing with restarts
        for ls_restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            current_residues = residues[:]
            current_conflicts = conflicts
            
            # If we start from a valid partial solution, we should initialize those residues.
            # But for simplicity, let's just do random restarts.
            
            no_improvement = 0
            while time.time() < deadline - 5.0 and current_conflicts > 0:
                if no_improvement > 200:
                    break
                    
                # Pick random index
                i = random.randint(0, n_mods - 1)
                old_r = current_residues[i]
                new_r = random.randint(0, moduli[i] - 1)
                if new_r == old_r:
                    continue
                    
                current_residues[i] = new_r
                new_conflicts = count_conflicts(current_residues, indices)
                
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    no_improvement = 0
                else:
                    current_residues[i] = old_r
                    no_improvement += 1
            
            if current_conflicts == 0:
                # Found a valid solution for all 64
                sol = [[current_residues[i], moduli[i]] for i in range(n_mods)]
                return sol
            else:
                # Restart with random residues
                residues = [random.randint(0, m - 1) for m in moduli]
                conflicts = count_conflicts(residues, indices)

    # Fallback: Return the best greedy solution found
    if best_solution:
        return best_solution
    else:
        # Absolute fallback: return a single class
        return [[0, 2]]
