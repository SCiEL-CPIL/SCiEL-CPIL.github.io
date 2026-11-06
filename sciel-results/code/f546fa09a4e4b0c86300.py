import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mods = len(moduli)
    
    gcds = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            g = math.gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Greedy to find initial solution
    def greedy():
        current_mods = []
        current_res = []
        for m in moduli:
            for r in range(m):
                valid = True
                for j in range(len(current_mods)):
                    m_j = current_mods[j]
                    r_j = current_res[j]
                    if (r - r_j) % gcds[m][m_j] == 0:
                        valid = False
                        break
                if valid:
                    current_mods.append(m)
                    current_res.append(r)
                    break
        return current_mods, current_res
        
    g_mods, g_res = greedy()
    best_count = len(g_mods)
    best_solution = [[r, m] for m, r in zip(g_mods, g_res)]
    
    # Backtracking to improve
    # We can try to skip some moduli in the greedy solution to allow others?
    # Actually, the backtracking will explore all possibilities.
    
    def backtrack(idx, current_mods, current_res):
        nonlocal best_solution, best_count
        
        if time.time() > deadline:
            return
            
        if len(current_mods) + (n_mods - idx) <= best_count:
            return
            
        if idx == n_mods:
            if len(current_mods) > best_count:
                best_count = len(current_mods)
                best_solution = [[r, m] for m, r in zip(current_mods, current_res)]
            return
            
        m = moduli[idx]
        
        # Option 1: Skip
        backtrack(idx + 1, current_mods, current_res)
        
        # Option 2: Try residues
        # Heuristic: try residues that are "different" from existing ones?
        # Just iterate 0..m-1
        for r in range(m):
            valid = True
            for j in range(len(current_mods)):
                m_j = current_mods[j]
                r_j = current_res[j]
                if (r - r_j) % gcds[m][m_j] == 0:
                    valid = False
                    break
            
            if valid:
                current_mods.append(m)
                current_res.append(r)
                backtrack(idx + 1, current_mods, current_res)
                current_mods.pop()
                current_res.pop()
                
                if best_count == n_mods:
                    return

    backtrack(0, [], [])
    
    return best_solution
