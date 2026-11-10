import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_MOD = 128
    
    # Focus on the set of even moduli, which is the largest set of pairwise non-coprime moduli
    moduli = list(range(2, MAX_MOD + 1, 2))
    n_mods = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    def initial_random():
        return [random.randint(0, m - 1) for m in moduli]

    def calc_conflicts(residues):
        c = 0
        for i in range(n_mods):
            ri = residues[i]
            for j in range(i + 1, n_mods):
                if ri % gcds[i][j] == residues[j] % gcds[i][j]:
                    c += 1
        return c

    def local_search():
        residues = initial_random()
        current_conflicts = calc_conflicts(residues)
        best_residues = residues[:]
        best_conflicts = current_conflicts
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 2.0:
            if current_conflicts == 0:
                return residues
            
            i = random.randint(0, n_mods - 1)
            old_r = residues[i]
            new_r = random.randint(0, moduli[i] - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta in conflicts
            # Remove old conflicts involving i
            delta = 0
            for j in range(n_mods):
                if j == i: continue
                g = gcds[i][j]
                if old_r % g == residues[j] % g:
                    delta -= 1
                if new_r % g == residues[j] % g:
                    delta += 1
            
            new_conflicts = current_conflicts + delta
            
            if new_conflicts < current_conflicts:
                residues[i] = new_r
                current_conflicts = new_conflicts
                no_improve = 0
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_residues = residues[:]
            else:
                # Revert
                no_improve += 1
                if no_improve > max_no_improve:
                    # Restart
                    residues = initial_random()
                    current_conflicts = calc_conflicts(residues)
                    no_improve = 0
                    max_no_improve = random.randint(200, 500)
                    
        return best_residues if best_conflicts < current_conflicts else residues

    best_solution = None
    
    # Try multiple times
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        res = local_search()
        if calc_conflicts(res) == 0:
            best_solution = [[res[i], moduli[i]] for i in range(n_mods)]
            break

    if best_solution is None:
        # Fallback: Greedy on a subset
        chosen = []
        for m in moduli:
            if time.time() > deadline - 1.0:
                break
            # Find a residue that works with all chosen
            found = False
            for r in range(m):
                ok = True
                for r2, m2 in chosen:
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    chosen.append([r, m])
                    found = True
                    break
            if not found:
                continue
            if len(chosen) > 50:
                break
        best_solution = chosen

    return best_solution
