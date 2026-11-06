import time
import math
import random

MAX_MOD = 128
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    moduli = list(range(2, MAX_MOD + 1))
    
    def num_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
            i += 1
        count += 1
        return count

    # Sort by descending divisors
    moduli.sort(key=num_divisors, reverse=True)
    n_mods = len(moduli)
    
    gcd_table = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_compatible(r1, m1_idx, r2, m2_idx):
        g = gcd_table[m1_idx][m2_idx]
        return (r1 - r2) % g != 0

    # Greedy initialization
    best_count = 0
    best_solution = []
    
    def greedy_solve(order, rng):
        chosen = []
        for idx in order:
            m = moduli[idx]
            found = False
            # Shuffle residues for diversity
            residues = list(range(m))
            rng.shuffle(residues)
            for r in residues:
                ok = True
                for (r2, m2_idx) in chosen:
                    if not is_compatible(r, idx, r2, m2_idx):
                        ok = False
                        break
                if ok:
                    chosen.append((r, idx))
                    found = True
                    break
        return chosen

    rng = random.Random(42)
    # Try a few random orders for greedy to find a good initial solution
    for _ in range(100):
        order = list(range(n_mods))
        rng.shuffle(order)
        sol = greedy_solve(order, rng)
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = [[r, moduli[i]] for r, i in sol]
            
    # Backtracking
    initial_domains = [set(range(m)) for m in moduli]
    assignment = [None] * n_mods
    
    def backtrack_copy(index, current_count, domains, assignment):
        nonlocal best_count, best_solution
        
        if time.time() - start_time > DEADLINE:
            return
            
        if current_count + (n_mods - index) <= best_count:
            return
            
        if index == n_mods:
            if current_count > best_count:
                best_count = current_count
                sol = []
                for i in range(n_mods):
                    if assignment[i] is not None:
                        sol.append([assignment[i], moduli[i]])
                best_solution = sol
            return
            
        # Option 1: Skip
        backtrack_copy(index + 1, current_count, domains, assignment)
        
        # Option 2: Assign
        m = moduli[index]
        
        valid_residues = []
        for r in domains[index]:
            compatible = True
            for i in range(index):
                if assignment[i] is not None:
                    if not is_compatible(r, index, assignment[i], i):
                        compatible = False
                        break
            if compatible:
                valid_residues.append(r)
                
        if not valid_residues:
            return
            
        # Sort valid residues: try those that are "more central" or just random?
        # Let's try random to escape local minima, but seeded?
        # No, deterministic is better for B&B correctness, but random can help find good solutions faster.
        # Let's just iterate.
        
        for r in valid_residues:
            if time.time() - start_time > DEADLINE:
                return
                
            assignment[index] = r
            
            new_domains = [None] * n_mods
            # Copy domains for indices <= index (not strictly needed as we don't touch them)
            for j in range(index + 1, n_mods):
                if assignment[j] is not None:
                    new_domains[j] = domains[j]
                else:
                    g = gcd_table[index][j]
                    if g == 1:
                        new_domains[j] = domains[j]
                    else:
                        r_mod_g = r % g
                        new_domains[j] = {r_j for r_j in domains[j] if r_j % g != r_mod_g}
                        
            valid = True
            for j in range(index + 1, n_mods):
                if assignment[j] is None and not new_domains[j]:
                    valid = False
                    break
                    
            if valid:
                backtrack_copy(index + 1, current_count + 1, new_domains, assignment)
                
            assignment[index] = None

    backtrack_copy(0, 0, initial_domains, assignment)
    
    if not best_solution:
        best_solution = [[0, 2]]
        
    return best_solution
