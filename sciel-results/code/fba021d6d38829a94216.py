"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mods = len(moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[1] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Conflict degree: number of other moduli with gcd > 1
    conflict_deg = [0] * n_mods
    for i in range(n_mods):
        for j in range(n_mods):
            if i != j and gcd_mat[i][j] > 1:
                conflict_deg[i] += 1
    order = sorted(range(n_mods), key=lambda i: -conflict_deg[i])
    
    def get_incompatible_residues(mod_idx, chosen_state):
        """Return set of residues incompatible with mod_idx given chosen_state."""
        m = moduli[mod_idx]
        incompatible = set()
        for other_idx, r_other in chosen_state.items():
            g = gcd_mat[mod_idx][other_idx]
            if g > 1:
                # r is incompatible if (r - r_other) % g == 0
                # i.e., r % g == r_other % g
                target = r_other % g
                for r in range(target, m, g):
                    incompatible.add(r)
        return incompatible
    
    def greedy():
        chosen = {}
        for mod_idx in order:
            m = moduli[mod_idx]
            incompat = get_incompatible_residues(mod_idx, chosen)
            for r in range(m):
                if r not in incompat:
                    chosen[mod_idx] = r
                    break
        return chosen
    
    chosen = greedy()
    best_size = len(chosen)
    best_chosen = dict(chosen)
    
    # Local search
    def local_search(chosen_state, max_iters=200000):
        nonlocal best_size, best_chosen
        chosen_ls = dict(chosen_state)
        current_count = len(chosen_ls)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding
            for mod_idx in range(n_mods):
                if mod_idx in chosen_ls:
                    continue
                m = moduli[mod_idx]
                incompat = get_incompatible_residues(mod_idx, chosen_ls)
                for r in range(m):
                    if r not in incompat:
                        chosen_ls[mod_idx] = r
                        current_count += 1
                        improved = True
                        if current_count > best_size:
                            best_size = current_count
                            best_chosen = dict(chosen_ls)
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try swap
            for remove_idx in list(chosen_ls.keys()):
                old_r = chosen_ls[remove_idx]
                del chosen_ls[remove_idx]
                current_count -= 1
                
                for add_idx in range(n_mods):
                    if add_idx in chosen_ls:
                        continue
                    m = moduli[add_idx]
                    incompat = get_incompatible_residues(add_idx, chosen_ls)
                    for r in range(m):
                        if r not in incompat:
                            chosen_ls[add_idx] = r
                            current_count += 1
                            improved = True
                            if current_count > best_size:
                                best_size = current_count
                                best_chosen = dict(chosen_ls)
                            break
                    if improved:
                        break
                
                if improved:
                    break
                else:
                    chosen_ls[remove_idx] = old_r
                    current_count += 1
            
            if not improved:
                break
        
        return chosen_ls
    
    chosen_ls = local_search(chosen)
    
    result = []
    for mod_idx, r in chosen_ls.items():
        result.append([r, moduli[mod_idx]])
    
    if not result:
        result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return result
# EVOLVE-BLOCK-END
