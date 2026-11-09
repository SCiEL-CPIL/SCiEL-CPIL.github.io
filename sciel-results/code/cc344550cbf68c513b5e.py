import time
from math import gcd

def construct():
    deadline = time.time() + 50
    
    all_moduli = list(range(2, 129))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])
    
    best_result = []
    best_size = 0
    node_count = 0
    MAX_NODES = 500000
    
    def backtrack(selected, assignment, remaining):
        nonlocal best_result, best_size, node_count
        
        node_count += 1
        if node_count > MAX_NODES or time.time() > deadline:
            return
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = [[assignment[i], all_moduli[i]] for i in selected]
        
        if not remaining:
            return
        
        # MRV heuristic: find the modulus with fewest valid residues
        best_idx = -1
        best_valid_rs = None
        best_count = float('inf')
        
        for idx in remaining:
            m = all_moduli[idx]
            valid_rs = []
            for r in range(m):
                ok = True
                for j in selected:
                    g = gcd_mat[idx][j]
                    if (r - assignment[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_rs.append(r)
            
            if not valid_rs:
                return  # Dead end - no valid residue for this modulus
            
            if len(valid_rs) < best_count:
                best_count = len(valid_rs)
                best_idx = idx
                best_valid_rs = valid_rs
                if best_count == 1:
                    break
        
        if best_idx == -1:
            return
        
        remaining.remove(best_idx)
        
        # Try residues in a deterministic order for reproducibility
        for r in best_valid_rs:
            if node_count > MAX_NODES or time.time() > deadline:
                return
            
            selected.append(best_idx)
            assignment[best_idx] = r
            
            backtrack(selected, assignment, remaining)
            
            assignment.pop(best_idx)
            selected.pop()
        
        remaining.add(best_idx)
    
    all_indices = set(range(n))
    backtrack([], {}, all_indices)
    
    if best_result:
        return best_result
    
    return [[0, 2]]
