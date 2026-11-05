import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute number of non-coprime neighbors for each modulus
    non_coprime_count = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and gcd_tab[i][j] > 1:
                non_coprime_count[i] += 1

    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected."""
        for (idx2, r2, m2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_greedy(order, residue_order=None):
        selected = []
        for idx in order:
            m = moduli[idx]
            if residue_order is None:
                residues = [1, 0] + [r for r in range(2, m)]
            else:
                residues = residue_order(m)
            
            found = False
            for r in residues:
                if check_residue(idx, r, selected):
                    selected.append((idx, r, m))
                    found = True
                    break
        return selected

    def residue_order_random(m):
        res = list(range(m))
        random.shuffle(res)
        return res

    best = []
    best_len = 0

    # Deterministic Orderings
    orderings = [
        sorted(range(n), key=lambda i: -non_coprime_count[i]),
        sorted(range(n), key=lambda i: non_coprime_count[i]),
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        sorted(range(n), key=lambda i: (0 if non_coprime_count[i] == 0 else 1, moduli[i])),
    ]

    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random Restarts
    random.seed(12345)
    restarts = 5000
    for it in range(restarts):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order, residue_order=residue_order_random)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Efficient local search using conflict-based approach
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        # Build residue lookup: idx -> residue
        residue_map = {idx: r for (idx, r, m) in best}
        m_map = {idx: m for (idx, r, m) in best}
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # For each unselected modulus, try to add it
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                
                # Find which selected moduli conflict with each possible residue of new_idx
                # A residue r for new_idx conflicts with selected idx2 if (r - r2) % gcd(new_idx, idx2) == 0
                # We want a residue that doesn't conflict with ANY selected modulus
                
                # For each selected modulus, compute the set of forbidden residues for new_idx
                # Forbidden residues: r such that (r - r2) % g == 0, i.e., r ≡ r2 (mod g)
                # Within [0, m_new), these are: r2 % g, (r2 % g) + g, (r2 % g) + 2g, ... < m_new
                
                forbidden = [False] * m_new
                for sel_idx in selected_idx:
                    g = gcd_tab[new_idx][sel_idx]
                    if g == 1:
                        continue  # coprime, no restriction
                    r2 = residue_map[sel_idx]
                    base = r2 % g
                    for r in range(base, m_new, g):
                        forbidden[r] = True
                
                # Find first non-forbidden residue
                found_r = -1
                for r in range(m_new):
                    if not forbidden[r]:
                        found_r = r
                        break
                
                if found_r >= 0:
                    best.append((new_idx, found_r, m_new))
                    selected_idx.add(new_idx)
                    residue_map[new_idx] = found_r
                    m_map[new_idx] = m_new
                    best_len += 1
                    improved = True
            
            if not improved:
                # Try swap-based local search
                swap_improved = True
                while swap_improved and time.time() < deadline - 5:
                    swap_improved = False
                    
                    for new_idx in range(n):
                        if new_idx in selected_idx:
                            continue
                        m_new = moduli[new_idx]
                        
                        # Try to find a swap: change residue of one selected modulus to free up space for new_idx
                        for pos, (old_idx, old_r, old_m) in enumerate(best):
                            # Find valid residues for old_idx that are compatible with all OTHERS
                            valid_old_residues = []
                            for new_r_old in range(old_m):
                                if new_r_old == old_r:
                                    continue
                                ok = True
                                for qpos in range(len(best)):
                                    if qpos == pos:
                                        continue
                                    other_idx, other_r, other_m = best[qpos]
                                    g = gcd_tab[old_idx][other_idx]
                                    if (new_r_old - other_r) % g == 0:
                                        ok = False
                                        break
                                if ok:
                                    valid_old_residues.append(new_r_old)
                            
                            if not valid_old_residues:
                                continue

                            # For each valid new residue for old_idx, see if we can add new_idx
                            for new_r_old in valid_old_residues:
                                # Check if new_idx can be added with new_r_old in place of old_r
                                found_new = False
                                for r_new in range(m_new):
                                    ok_new = True
                                    for qpos in range(len(best)):
                                        if qpos == pos:
                                            # Compare with new_r_old
                                            if (r_new - new_r_old) % gcd_tab[new_idx][old_idx] == 0:
                                                ok_new = False
                                                break
                                        else:
                                            other_idx, other_r, other_m = best[qpos]
                                            if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                                ok_new = False
                                                break
                                    if ok_new:
                                        found_new = True
                                        break
                                
                                if found_new:
                                    # Perform the swap
                                    best[pos] = (old_idx, new_r_old, old_m)
                                    best.append((new_idx, r_new, m_new))
                                    selected_idx.add(new_idx)
                                    residue_map[new_idx] = r_new
                                    m_map[new_idx] = m_new
                                    best_len += 1
                                    swap_improved = True
                                    break
                            if swap_improved:
                                break
                        if swap_improved:
                            break

    return [[r, m] for (_, r, m) in best]
