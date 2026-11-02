import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate Weight 3 vectors (1320)
    vecs3 = []
    indices = range(D)
    for support in itertools.combinations(indices, 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[support[0]] = s1
                    v[support[1]] = s2
                    v[support[2]] = s3
                    vecs3.append(v)
    V3 = np.array(vecs3, dtype=np.int64)
    n3 = len(V3)
    
    # 2. Generate Weight 2 vectors (220)
    vecs2 = []
    for i, j in itertools.combinations(indices, 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vecs2.append(v)
    V2 = np.array(vecs2, dtype=np.int64)
    n2 = len(V2)
    
    # 3. Compute Conflict Graph for V3
    # Conflict if dot >= 2
    dots3 = V3 @ V3.T
    adj3 = (dots3 >= 2).astype(np.int8)
    np.fill_diagonal(adj3, 0)
    degrees3 = adj3.sum(axis=1)
    
    # 4. Greedy MIS on V3
    order3 = np.argsort(degrees3)
    selected3_mask = np.zeros(n3, dtype=bool)
    
    for i in order3:
        if selected3_mask[i]:
            continue
        # Check conflict with selected
        if np.dot(adj3[i], selected3_mask) == 0:
            selected3_mask[i] = True
            
    # 5. Select V2 vectors compatible with selected V3
    # Condition for v2 (norm sqrt(2)) and u3 (norm sqrt(3)):
    # ||v2 - u3||^2 >= 3 (Max Norm is sqrt(3))
    # 2 + 3 - 2 v2.u3 >= 3
    # 2 - 2 v2.u3 >= 0
    # v2.u3 <= 1
    
    selected2_mask = np.zeros(n2, dtype=bool)
    
    # We need to check each v2 against all selected v3
    # To speed up, we can precompute the selected V3 matrix
    V3_sel = V3[selected3_mask]
    n3_sel = len(V3_sel)
    
    if n3_sel > 0:
        # Dots between V2 and V3_sel
        # Shape: (n2, n3_sel)
        dots23 = V2 @ V3_sel.T
        
        # v2 is compatible if max dot with any selected v3 is <= 1
        # i.e., all dots <= 1
        # np.all(dots23 <= 1, axis=1)
        compatible2 = np.all(dots23 <= 1, axis=1)
        
        # We also need to check V2 vs V2 compatibility?
        # If we add multiple V2, they must be compatible with each other.
        # Max norm is still sqrt(3) (since V3 are present).
        # Condition for v2a, v2b:
        # ||v2a - v2b||^2 >= 3
        # 2 + 2 - 2 v2a.v2b >= 3
        # 2 - 2 v2a.v2b >= 0
        # v2a.v2b <= 1
        # This is the same condition as for the pure V2 set with max norm sqrt(2)?
        # No, pure V2 set had max norm sqrt(2), so dist^2 >= 2, dot <= 1.
        # Here max norm is sqrt(3), so dist^2 >= 3, dot <= 0.5 -> dot <= 0?
        # Wait.
        # If we have V3 vectors, Max Norm is $\sqrt{3}$.
        # For two V2 vectors $u, v$:
        # $\|u-v\|^2 \ge 3$.
        # $2+2-2 u \cdot v \ge 3 \implies 1 \ge 2 u \cdot v \implies u \cdot v \le 0.5$.
        # Since integer, $u \cdot v \le 0$.
        # So V2 vectors must be orthogonal or have negative dot.
        # This is stricter than the pure V2 case (where dot <= 1 was allowed).
        
        # So we need to find a Max Independent Set in the V2 graph where edges are dot >= 1?
        # No, edges are dot > 0?
        # Condition: dot <= 0.
        # So we need V2 vectors with pairwise dot <= 0.
        
        # Let's filter compatible2 first.
        candidates2 = np.where(compatible2)[0]
        
        if len(candidates2) > 0:
            V2_cand = V2[candidates2]
            dots22 = V2_cand @ V2_cand.T
            # Conflict if dot > 0
            adj2 = (dots22 > 0).astype(np.int8)
            np.fill_diagonal(adj2, 0)
            
            # Greedy MIS on V2 candidates
            deg2 = adj2.sum(axis=1)
            order2 = np.argsort(deg2)
            selected2_cand_mask = np.zeros(len(candidates2), dtype=bool)
            
            for i in order2:
                if selected2_cand_mask[i]:
                    continue
                if np.dot(adj2[i], selected2_cand_mask) == 0:
                    selected2_cand_mask[i] = True
            
            # Map back to original V2 indices
            selected2_indices = candidates2[selected2_cand_mask]
            selected2_mask[selected2_indices] = True
        else:
            selected2_indices = []
    else:
        selected2_indices = []

    # Combine
    final_vecs = []
    if np.any(selected3_mask):
        final_vecs.append(V3[selected3_mask])
    if np.any(selected2_mask):
        final_vecs.append(V2[selected2_mask])
        
    if not final_vecs:
        # Fallback to original
        return V2
        
    result = np.vstack(final_vecs)
    return result
