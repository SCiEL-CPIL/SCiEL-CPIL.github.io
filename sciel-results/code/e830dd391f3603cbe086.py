import sys
from itertools import combinations
import time
import numpy as np

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    Primary: exact MILP via scipy.optimize.milp.
    Fallback: backtracking with MRV heuristic.
    """
    start_time = time.time()
    deadline = start_time + 100.0

    # Map 5-subsets to IDs (0..791)
    all_5subsets = list(combinations(range(12), 5))
    five_to_id = {sub: i for i, sub in enumerate(all_5subsets)}
    
    # Map 6-subsets to IDs (0..923)
    all_6subsets = list(combinations(range(12), 6))
    n6 = len(all_6subsets)  # 924
    n5 = len(all_5subsets)  # 792
    
    # Precompute: for each 6-subset, the list of 5-subset IDs it contains
    six_to_fives = []
    for block in all_6subsets:
        fives = []
        for i in range(6):
            sub = block[:i] + block[i+1:]
            fives.append(five_to_id[sub])
        six_to_fives.append(fives)
    
    # Precompute: for each 5-subset ID, the list of 6-subset indices containing it
    five_to_six = [[] for _ in range(n5)]
    for idx, fives in enumerate(six_to_fives):
        for fid in fives:
            five_to_six[fid].append(idx)
    
    # --- Try MILP first ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Binary variables x_j for each 6-subset j
        # Maximize sum(x_j)
        # Subject to: for each 5-subset i, sum_{j: i in six_to_fives[j]} x_j <= 1
        #             x_j in {0, 1}
        
        # Objective: maximize sum(x) => minimize -sum(x)
        c = -np.ones(n6)
        
        # Constraint matrix A: n5 x n6, A[i,j] = 1 if 5-subset i is in 6-subset j
        # Build sparse
        rows = []
        cols = []
        data = []
        for j, fives in enumerate(six_to_fives):
            for fid in fives:
                rows.append(fid)
                cols.append(j)
                data.append(1)
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(n5, n6))
        
        # Constraints: A @ x <= 1
        constraint = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        
        # Integrate all variables
        integrality = np.ones(n6)
        
        result = milp(
            c=c,
            constraints=[constraint],
            integrality=integrality,
            bounds=bounds
        )
        
        if result.success and result.x is not None:
            blocks = []
            for j in range(n6):
                if result.x[j] > 0.5:
                    blocks.append(list(all_6subsets[j]))
            
            if len(blocks) >= 132:
                # Verify: no 5-subset repeated
                seen_fives = set()
                valid = True
                for b in blocks:
                    for i in range(6):
                        sub = tuple(b[:i] + b[i+1:])
                        if sub in seen_fives:
                            valid = False
                            break
                        seen_fives.add(sub)
                    if not valid:
                        break
                if valid:
                    return blocks[:132]
                else:
                    # Try to fix: keep only first 132 valid
                    seen_fives = set()
                    valid_blocks = []
                    for b in blocks:
                        ok = True
                        for i in range(6):
                            sub = tuple(b[:i] + b[i+1:])
                            if sub in seen_fives:
                                ok = False
                                break
                            seen_fives.add(sub)
                        if ok:
                            valid_blocks.append(b)
                    if len(valid_blocks) >= 132:
                        return valid_blocks[:132]
        
        # MILP didn't give clean 132, fall through to backtracking
        if time.time() > deadline - 5:
            return []
            
    except Exception:
        pass
    
    # --- Fallback: backtracking with MRV ---
    if time.time() > deadline - 5:
        return []
    
    used_fives = [False] * n5
    remaining_6 = [True] * n6
    blocks_found = []
    
    # Symmetry Breaking 1: Fix first block to (0,1,2,3,4,5)
    first_block_idx = all_6subsets.index((0, 1, 2, 3, 4, 5))
    blocks_found.append(list(all_6subsets[first_block_idx]))
    remaining_6[first_block_idx] = False
    for fid in six_to_fives[first_block_idx]:
        used_fives[fid] = True
        
    # Symmetry Breaking 2: Fix second block to the lexicographically smallest valid block
    second_block_idx = -1
    for idx in range(n6):
        if remaining_6[idx]:
            valid = True
            for fid in six_to_fives[idx]:
                if used_fives[fid]:
                    valid = False
                    break
            if valid:
                second_block_idx = idx
                break
                
    if second_block_idx != -1:
        blocks_found.append(list(all_6subsets[second_block_idx]))
        remaining_6[second_block_idx] = False
        for fid in six_to_fives[second_block_idx]:
            used_fives[fid] = True
    
    def backtrack():
        if time.time() > deadline:
            return False
            
        if len(blocks_found) == 132:
            return True
            
        # Find MRV: 5-subset with min available 6-subsets
        min_avail = float('inf')
        best_fid = -1
        
        for fid in range(n5):
            if used_fives[fid]:
                continue
            count = 0
            for idx in five_to_six[fid]:
                if remaining_6[idx]:
                    count += 1
                    if count >= min_avail:
                        break
            if count == 0:
                return False
            if count < min_avail:
                min_avail = count
                best_fid = fid
                if count == 1:
                    break
                    
        if best_fid == -1:
            return True
            
        for idx in five_to_six[best_fid]:
            if not remaining_6[idx]:
                continue
                
            valid = True
            for fid in six_to_fives[idx]:
                if used_fives[fid]:
                    valid = False
                    break
            if not valid:
                continue
                
            remaining_6[idx] = False
            blocks_found.append(list(all_6subsets[idx]))
            for fid in six_to_fives[idx]:
                used_fives[fid] = True
                
            if backtrack():
                return True
                
            remaining_6[idx] = True
            blocks_found.pop()
            for fid in six_to_fives[idx]:
                used_fives[fid] = False
                
        return False

    if backtrack():
        return blocks_found
        
    return blocks_found

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
