import time
import math
import random

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we return within 60s wall-clock
    # The evaluator kills at 90s, but let's use 50s for the algorithm to be safe.
    deadline = start_time + 50.0
    MAX_MOD = 128

    # Candidate moduli: All numbers from 2 to 128
    # Using all moduli gives the maximum potential set size (127).
    moduli = list(range(2, MAX_MOD + 1))
    n_mods = len(moduli)

    # Precompute gcds
    gcds = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    def count_conflicts_fast(residues, indices):
        """Count number of conflicting pairs in the subset using optimized loops."""
        conflicts = 0
        n = len(indices)
        for i in range(n):
            idx_i = indices[i]
            r_i = residues[idx_i]
            # Use local vars for speed
            gcds_i = gcds[idx_i]
            for j in range(i + 1, n):
                idx_j = indices[j]
                r_j = residues[idx_j]
                g = gcds_i[idx_j]
                if (r_i - r_j) % g == 0:
                    conflicts += 1
        return conflicts

    def is_valid(residues, indices):
        """Check if the subset of residues at given indices is valid."""
        n = len(indices)
        for i in range(n):
            idx_i = indices[i]
            r_i = residues[idx_i]
            gcds_i = gcds[idx_i]
            for j in range(i + 1, n):
                idx_j = indices[j]
                r_j = residues[idx_j]
                g = gcds_i[idx_j]
                if (r_i - r_j) % g == 0:
                    return False
        return True

    def try_add_residue(m_idx, chosen_indices, chosen_residues):
        """Find a valid residue for modulus at m_idx given current chosen set."""
        m = moduli[m_idx]
        # Try residues 0..m-1. Randomizing order might help escape local optima in greedy,
        # but sequential is faster. We'll stick to sequential for speed in the main loop,
        # but we can try random offsets if we want variety.
        # Actually, let's try all residues. Since m <= 128, this is cheap.
        for r in range(m):
            ok = True
            for c_idx in chosen_indices:
                g = gcds[m_idx][c_idx]
                if (r - chosen_residues[c_idx]) % g == 0:
                    ok = False
                    break
            if ok:
                return r
        return None

    best_size = 0
    best_solution = []

    # Strategy 1: Greedy construction with multiple orderings
    # We try different orderings: Random, Descending, Ascending, by number of divisors/factors.
    
    # Precompute a heuristic key: number of prime factors (with multiplicity)
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    factor_counts = [count_prime_factors(m) for m in moduli]

    # Define orderings
    orderings = []
    # 1. Random
    orderings.append(None) # Placeholder for random
    # 2. Descending Modulus
    orderings.append(sorted(range(n_mods), key=lambda i: -moduli[i]))
    # 3. Ascending Modulus
    orderings.append(sorted(range(n_mods), key=lambda i: moduli[i]))
    # 4. High Factor Count (more constrained first? or less? Usually more constrained first is better for packing)
    orderings.append(sorted(range(n_mods), key=lambda i: -factor_counts[i]))
    # 5. Low Factor Count
    orderings.append(sorted(range(n_mods), key=lambda i: factor_counts[i]))

    # Run greedy for each ordering type multiple times (with randomization for the random one)
    # We will do a loop of restarts.
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Select ordering
        order_type = restart % 5
        if order_type == 0:
            order = list(range(n_mods))
            random.shuffle(order)
        else:
            order = orderings[order_type][:]
            # Add some randomness to tie-breaking or slight perturbation?
            # For now, use the fixed ordering for deterministic types to ensure coverage.
            
        chosen_indices = []
        chosen_residues = [0] * n_mods
        
        for idx in order:
            m = moduli[idx]
            found_r = try_add_residue(idx, chosen_indices, chosen_residues)
            
            if found_r is not None:
                chosen_indices.append(idx)
                chosen_residues[idx] = found_r

        # Evaluate this solution
        size = len(chosen_indices)
        if size > best_size:
            best_size = size
            sol = []
            for idx in chosen_indices:
                sol.append([chosen_residues[idx], moduli[idx]])
            best_solution = sol
            
        # Early exit if we found the maximum possible (all 127)
        if best_size == n_mods:
            break

    # Strategy 2: Local Search / Hill Climbing on the full set
    # If greedy didn't find all 127, we try to optimize the full set of residues.
    # We start from the best greedy solution's residues for the chosen ones, and random for others?
    # Or just random start.
    # A better start: Take the best greedy solution, assign random residues to the missing moduli,
    # then minimize conflicts.
    
    if time.time() < deadline - 5.0 and best_size < n_mods:
        
        # Initialize residues
        # Start with best greedy residues where available
        current_residues = [0] * n_mods
        # Copy best solution residues
        best_res_map = {m: r for r, m in best_solution}
        for i in range(n_mods):
            m = moduli[i]
            if m in best_res_map:
                current_residues[i] = best_res_map[m]
            else:
                current_residues[i] = random.randint(0, m - 1)
                
        indices = list(range(n_mods))
        current_conflicts = count_conflicts_fast(current_residues, indices)
        
        # Hill Climbing
        # We try to reduce conflicts by changing individual residues.
        # To speed up, we can precompute which pairs are in conflict?
        # Or just recompute conflicts for the changed index.
        
        def get_conflicts_for_index(i, residues):
            """Return list of indices j > i or j < i that conflict with i."""
            # This is O(N). We do this for each move candidate.
            # Total O(N^2) per step. With N=127, this is ~16k ops.
            # 1000 steps * 16k = 16M ops. Should be fine in Python? Maybe tight.
            # Let's optimize by only checking against others.
            conflicts = []
            r_i = residues[i]
            for j in range(n_mods):
                if i == j: continue
                r_j = residues[j]
                g = gcds[i][j]
                if (r_i - r_j) % g == 0:
                    conflicts.append(j)
            return conflicts

        # Simple Local Search Loop
        ls_restarts = 10
        for ls_restart in range(ls_restarts):
            if time.time() > deadline - 2.0:
                break
            
            no_improvement_count = 0
            max_no_improvement = 1000 # Adjust based on speed
            
            while time.time() < deadline - 2.0 and current_conflicts > 0:
                if no_improvement_count > max_no_improvement:
                    break
                
                # Pick a random index to change
                i = random.randint(0, n_mods - 1)
                m = moduli[i]
                old_r = current_residues[i]
                
                # Try all possible residues for this index?
                # m is up to 128. Trying all 128 residues and checking conflicts is 128 * 127 ~ 16k ops.
                # This is a "best move" search for index i.
                best_new_r = old_r
                best_new_conflicts = current_conflicts
                
                # To speed up, we can estimate the change in conflicts.
                # But full recount is safer.
                # Let's just try a few random residues first, then all if needed?
                # Let's try all residues for the selected index.
                
                for new_r in range(m):
                    if new_r == old_r:
                        continue
                    current_residues[i] = new_r
                    # Count conflicts involving i
                    c = 0
                    r_i = new_r
                    for j in range(n_mods):
                        if i == j: continue
                        r_j = current_residues[j]
                        g = gcds[i][j]
                        if (r_i - r_j) % g == 0:
                            c += 1
                    # Total conflicts = (current_conflicts - conflicts_old_involving_i) + c
                    # We don't have conflicts_old_involving_i easily without storing it.
                    # So we just compare c against the number of conflicts i had before?
                    # Actually, simpler: just compute total conflicts for the new state?
                    # That's O(N^2) per residue trial. 128 * 16k = 2M ops per index move.
                    # Too slow if we do many moves.
                    
                    # Let's stick to random moves with delta evaluation.
                    # We need to know how many conflicts i had before.
                    # Let's precompute conflict count for each index?
                    # That's O(N^2) per step.
                    # Then changing i only affects conflicts involving i.
                    # Delta = new_conflicts_i - old_conflicts_i.
                    # Total conflicts += Delta.
                    
                    pass # Aborting this inner loop approach for speed, will use random moves
                    
                # Revert
                current_residues[i] = old_r
                
                # Random Move Strategy
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                # Calculate delta
                # 1. Count conflicts of i with old_r
                old_conf_i = 0
                for j in range(n_mods):
                    if i == j: continue
                    if (old_r - current_residues[j]) % gcds[i][j] == 0:
                        old_conf_i += 1
                
                # 2. Count conflicts of i with new_r
                new_conf_i = 0
                for j in range(n_mods):
                    if i == j: continue
                    if (new_r - current_residues[j]) % gcds[i][j] == 0:
                        new_conf_i += 1
                
                delta = new_conf_i - old_conf_i
                
                if delta < 0:
                    current_residues[i] = new_r
                    current_conflicts += delta
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
            if current_conflicts == 0:
                sol = [[current_residues[i], moduli[i]] for i in range(n_mods)]
                return sol
            else:
                # Restart local search with random residues if stuck
                if ls_restart < ls_restarts - 1:
                    current_residues = [random.randint(0, moduli[i] - 1) for i in range(n_mods)]
                    current_conflicts = count_conflicts_fast(current_residues, indices)

    # Fallback: Return the best greedy solution found
    if best_solution:
        return best_solution
    else:
        # Absolute fallback: return a single class
        return [[0, 2]]
