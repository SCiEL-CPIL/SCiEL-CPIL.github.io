import math
import time
import random

def construct():
    deadline = time.time() + 140
    max_m = 200
    min_m = 20
    
    # Precompute gcd
    gcd_table = [[0]*(max_m+1) for _ in range(max_m+1)]
    for i in range(min_m, max_m+1):
        for j in range(min_m, max_m+1):
            gcd_table[i][j] = math.gcd(i, j)

    def is_valid(m, r, selected, skip_idx=None):
        for idx, (m2, r2) in enumerate(selected):
            if skip_idx is not None and idx == skip_idx:
                continue
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0/m for m, _ in selected)

    def find_valid_residue(m, selected, skip_idx=None):
        valids = []
        for r in range(m):
            if is_valid(m, r, selected, skip_idx):
                valids.append(r)
        return valids

    def try_push(m_new, current_sol):
        # 1. Try direct addition
        valids = find_valid_residue(m_new, current_sol)
        if valids:
            current_sol.append([m_new, valids[0]])
            return True
        
        # 2. Try 1-push
        # For each r_new, find conflicts. If exactly 1 conflict, try to move it.
        for r_new in range(m_new):
            conflicts = []
            for idx, (m_old, r_old) in enumerate(current_sol):
                g = gcd_table[m_new][m_old]
                if r_new % g == r_old % g:
                    conflicts.append(idx)
            
            if len(conflicts) != 1:
                continue
                
            idx_conf = conflicts[0]
            m_old, r_old = current_sol[idx_conf]
            g_conf = gcd_table[m_new][m_old]
            
            # Find valid residues for m_old that don't conflict with r_new
            valids_old = find_valid_residue(m_old, current_sol, skip_idx=idx_conf)
            compatible = []
            for r_cand in valids_old:
                if r_new % g_conf != r_cand % g_conf:
                    compatible.append(r_cand)
            
            if compatible:
                # Move m_old to a compatible residue
                current_sol[idx_conf][1] = compatible[0]
                current_sol.append([m_new, r_new])
                return True
                
        return False

    def local_search(selected, deadline):
        current_sol = [list(x) for x in selected]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add missing moduli
            missing = [m for m in range(min_m, max_m+1) if not any(m == m_sel for m_sel, _ in current_sol)]
            missing.sort()
            
            for m_add in missing:
                if try_push(m_add, current_sol):
                    current_score += 1.0/m_add
                    improved = True
                    break
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]
                continue
            
            # 2. Try Swap
            if current_sol:
                max_m_idx = max(range(len(current_sol)), key=lambda i: current_sol[i][0])
                m_rem = current_sol[max_m_idx][0]
                
                missing = [m for m in range(min_m, max_m+1) if not any(m == m_sel for m_sel, _ in current_sol)]
                if missing:
                    missing.sort()
                    m_add = missing[0]
                    
                    if 1.0/m_add > 1.0/m_rem:
                        old_sol = [list(x) for x in current_sol]
                        old_score = current_score
                        
                        current_sol.pop(max_m_idx)
                        current_score -= 1.0/m_rem
                        
                        if try_push(m_add, current_sol):
                            current_score += 1.0/m_add
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_sol = [list(x) for x in current_sol]
                        else:
                            current_sol = old_sol
                            current_score = old_score
            
            if not improved:
                break
                
        return best_sol, best_score

    def initial_solution(deadline):
        selected = []
        for m in range(min_m, max_m+1):
            if time.time() > deadline:
                break
            valids = find_valid_residue(m, selected)
            if valids:
                selected.append([m, valids[0]])
        return selected

    best_score = -1
    best_sol = []
    
    while time.time() < deadline - 10:
        sol = initial_solution(deadline - 20)
        sol, score = local_search(sol, deadline - 5)
        if score > best_score:
            best_score = score
            best_sol = sol
            
    return best_sol

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0/m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
