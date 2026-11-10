import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    candidates = list(range(20, 201))
    n = len(candidates)
    
    # Precompute gcd matrix
    gcd_matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(i, n):
            g = gcd(candidates[i], candidates[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    best_score = -float('inf')
    best_result = []
    
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        random.seed(restarts)
        
        selected = [] # List of [m, r]
        selected_indices = [] # Indices in candidates
        skipped = [] # List of (index, m)
        
        for i, m in enumerate(candidates):
            if time.time() > deadline - 10.0:
                break
                
            # Check clique condition
            can_add_clique = True
            for k, (m2, r2) in enumerate(selected):
                g = gcd_matrix[i][selected_indices[k]]
                if g == 1:
                    can_add_clique = False
                    break
            
            if not can_add_clique:
                skipped.append((i, m))
                continue
                
            # Find valid residues
            valid_rs = []
            for r in range(m):
                valid = True
                for k, (m2, r2) in enumerate(selected):
                    g = gcd_matrix[i][selected_indices[k]]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
                    
            if not valid_rs:
                skipped.append((i, m))
                continue
                
            r = random.choice(valid_rs)
            selected.append([m, r])
            selected_indices.append(i)
            
        # Local Search: Try to add skipped moduli by changing residues
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Sort skipped by 1/m descending (smallest m first)
            skipped.sort(key=lambda x: 1.0 / x[1])
            
            for idx_s, m_s in skipped:
                if time.time() > deadline - 5.0:
                    break
                
                # Check if m_s is still in skipped (might have been added)
                # We need to check if m_s is compatible with current selected
                # 1. Clique check
                clique_ok = True
                for k, (m2, r2) in enumerate(selected):
                    g = gcd_matrix[idx_s][selected_indices[k]]
                    if g == 1:
                        clique_ok = False
                        break
                if not clique_ok:
                    continue
                
                # 2. Try to find a residue for m_s
                # First, try with current residues
                found_r_s = False
                for r_s in range(m_s):
                    valid = True
                    for k, (m2, r2) in enumerate(selected):
                        g = gcd_matrix[idx_s][selected_indices[k]]
                        if r_s % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        found_r_s = True
                        break
                        
                if found_r_s:
                    selected.append([m_s, r_s])
                    selected_indices.append(idx_s)
                    skipped.remove((idx_s, m_s))
                    improved = True
                    break # Restart loop to re-sort skipped
                
                # If not found, try to change one residue in selected
                # To speed up, only try changing residues of moduli that share gcd > 1 with m_s
                for k, (m2, r2) in enumerate(selected):
                    g = gcd_matrix[idx_s][selected_indices[k]]
                    if g == 1:
                        continue
                    
                    # Try changing r2 to r2_new
                    for r2_new in range(m2):
                        # Check if r2_new is valid with other selected
                        valid_others = True
                        for j, (m3, r3) in enumerate(selected):
                            if j == k:
                                continue
                            g3 = gcd_matrix[selected_indices[k]][selected_indices[j]]
                            if r2_new % g3 == r3 % g3:
                                valid_others = False
                                break
                        if not valid_others:
                            continue
                        
                        # Check if m_s has a valid residue with r2_new
                        found_r_s_new = False
                        for r_s in range(m_s):
                            # Check against all selected, with k updated
                            valid_s = True
                            for j, (m3, r3) in enumerate(selected):
                                g_s = gcd_matrix[idx_s][selected_indices[j]]
                                r3_eff = r2_new if j == k else r3
                                if r_s % g_s == r3_eff % g_s:
                                    valid_s = False
                                    break
                            if valid_s:
                                found_r_s_new = True
                                break
                            
                        if found_r_s_new:
                            # Apply change
                            selected[k][1] = r2_new
                            selected.append([m_s, r_s])
                            selected_indices.append(idx_s)
                            skipped.remove((idx_s, m_s))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        score = sum(1.0 / m for m, r in selected)
        
        if score > best_score:
            best_score = score
            best_result = [[m, r] for m, r in selected]
            
    if not best_result:
        return [[20, 0]]
        
    return best_result
