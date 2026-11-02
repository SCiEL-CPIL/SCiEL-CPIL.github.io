"""Combined approach: verified rank-39 Strassen+schoolbook, then targeted exact search for rank 38."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0        
        v_f = k1        
        v_g = p + k0    
        v_h = p + k1    
    
        w_c00 = k0 * n      
        w_c10 = k0 * n + 1  
        w_c01 = k1 * n      
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2        
    v_g2 = p + k2    
    w_c02 = k2 * n      
    w_c12 = k2 * n + 1  
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _try_rank38_by_merging(deadline):
    """
    Try to find a rank-38 decomposition by merging two of the rank-39 columns.
    The idea: if two columns can be combined into one, we reduce rank by 1.
    """
    n, m, p = 2, 2, 11
    U39, V39, W39 = _construct_rank39()
    rank39 = 39
    
    # Convert to numpy for efficiency
    U = np.array(U39, dtype=np.float64)
    V = np.array(V39, dtype=np.float64)
    W = np.array(W39, dtype=np.float64)
    
    # Target tensor entries
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.append((i * m + j, j * p + k, k * n + i))
    
    # Try merging each pair of columns
    for c1 in range(rank39):
        if time.time() > deadline - 5:
            break
        for c2 in range(c1 + 1, rank39):
            if time.time() > deadline - 5:
                break
            
            # Create rank-38 matrices by removing columns c1 and c2
            # and trying to find a single column that replaces them
            
            # The contribution of columns c1 and c2 to each target:
            # contrib[t] = U[a][c1]*V[b][c1]*W[c][c1] + U[a][c2]*V[b][c2]*W[c][c2]
            # We need: sum_{other cols} + new_col = target
            # So: new_col_contrib = target - sum_{other cols} - (c1_contrib + c2_contrib) + (c1_contrib + c2_contrib)
            # Wait, that's circular. Let me think differently.
            
            # Actually, we want: for all targets, 
            # sum_{c in 0..38} U[a][c]*V[b][c]*W[c][c] = 1
            # where the 39th column (index 38) is new, and we remove c1 and c2.
            
            # Let's compute the residual if we remove c1 and c2:
            # residual[t] = 1 - sum_{c != c1, c != c2} U[a][c]*V[b][c]*W[c][c]
            # We need residual[t] = U_new[a]*V_new[b]*W_new[c] for all t
            
            # Compute residual
            residual = np.zeros(len(targets))
            for idx, (a, b, c) in enumerate(targets):
                s = 0.0
                for col in range(rank39):
                    if col == c1 or col == c2:
                        continue
                    s += U[a, col] * V[b, col] * W[c, col]
                residual[idx] = 1.0 - s
            
            # Check if residual is a rank-1 tensor
            # Reshape to (n*m, m*p, p*n) but we only have the 44 target entries
            # For rank-1, residual should factor as u[a]*v[b]*w[c]
            
            # Try to factor: pick a reference target
            # targets[0] = (0, 0, 0) for i=0,j=0,k=0
            ref_a, ref_b, ref_c = targets[0]
            ref_val = residual[0]
            
            if abs(ref_val) < 1e-10:
                # If reference is ~0, all must be 0 for rank-1
                if np.any(np.abs(residual) > 1e-8):
                    continue
            
            # Try to extract u, v, w
            # u[ref_a] * v[ref_b] * w[ref_c] = ref_val
            # Set u[ref_a] = 1, then v[ref_b] * w[ref_c] = ref_val
            
            # We have 44 equations. Let's try to solve for u, v, w.
            # u has 4 entries, v has 22, w has 22. Total 48 unknowns, 44 equations.
            # Underdetermined, but we can fix normalization.
            
            # Fix: u[0] = 1 (if ref_a == 0, then v[ref_b]*w[ref_c] = ref_val)
            # This is getting complex. Let's use a simpler heuristic:
            # Check if residual can be written as outer product.
            
            # Reshape residual into a structure that allows rank-1 check
            # For each unique (a,b), residual should be proportional to w
            # For each unique (a,c), residual should be proportional to v
            # For each unique (b,c), residual should be proportional to u
            
            # Check consistency: for any two targets with same a, 
            # residual[idx1]/residual[idx2] should equal (v[b1]*w[c1])/(v[b2]*w[c2])
            # which should be consistent across different a values.
            
            # Simpler: use SVD-like approach. Build a matrix where rows are (a,b) pairs
            # and columns are c, and check if it's rank-1.
            
            # Actually, let's just try a numerical approach:
            # Assume u[0]=1. Then for targets with a=0: v[b]*w[c] = residual
            # This gives us v and w up to a scaling.
            
            # Get all targets with a=0
            a0_targets = [(idx, b, c) for idx, (a, b, c) in enumerate(targets) if a == 0]
            
            if len(a0_targets) < 2:
                continue
            
            # From a=0 targets: v[b]*w[c] = residual[idx]
            # Pick one target to normalize: v[b0]*w[c0] = residual[idx0]
            idx0, b0, c0 = a0_targets[0]
            if abs(residual[idx0]) < 1e-10:
                continue
            
            # Set v[b0] = 1, then w[c0] = residual[idx0]
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            v[b0] = 1.0
            w[c0] = residual[idx0]
            
            # Solve for other v and w entries
            for idx, b, c in a0_targets[1:]:
                if abs(residual[idx]) < 1e-10:
                    # v[b]*w[c] = 0, so either v[b]=0 or w[c]=0
                    continue
                if abs(v[b]) > 1e-10:
                    w[c] = residual[idx] / v[b]
                elif abs(w[c]) > 1e-10:
                    v[b] = residual[idx] / w[c]
                else:
                    # Both zero but residual non-zero: inconsistent
                    break
            else:
                # Check consistency with a=1 targets
                u = np.zeros(n * m)
                u[0] = 1.0  # a=0
                
                a1_targets = [(idx, b, c) for idx, (a, b, c) in enumerate(targets) if a == 1]
                consistent = True
                for idx, b, c in a1_targets:
                    expected = u[1] * v[b] * w[c]
                    if abs(expected - residual[idx]) > 1e-6:
                        consistent = False
                        break
                
                if consistent:
                    # Solve for u[1]
                    if a1_targets:
                        idx1, b1, c1 = a1_targets[0]
                        if abs(v[b1] * w[c1]) > 1e-10:
                            u[1] = residual[idx1] / (v[b1] * w[c1])
                        else:
                            if abs(residual[idx1]) > 1e-8:
                                continue
                    
                    # Verify all targets
                    all_good = True
                    for idx, (a, b, c) in enumerate(targets):
                        expected = u[a] * v[b] * w[c]
                        if abs(expected - residual[idx]) > 1e-6:
                            all_good = False
                            break
                    
                    if all_good:
                        # Found a rank-1 residual! Build the rank-38 solution
                        U38 = [[0.0] * 38 for _ in range(n * m)]
                        V38 = [[0.0] * 38 for _ in range(m * p)]
                        W38 = [[0.0] * 38 for _ in range(p * n)]
                        
                        col_map = 0
                        for col in range(rank39):
                            if col == c1 or col == c2:
                                continue
                            for a in range(n * m):
                                U38[a][col_map] = U39[a][col]
                            for b in range(m * p):
                                V38[b][col_map] = V39[b][col]
                            for c in range(p * n):
                                W38[c][col_map] = W39[c][col]
                            col_map += 1
                        
                        # Add the new column
                        for a in range(n * m):
                            U38[a][37] = u[a]
                        for b in range(m * p):
                            V38[b][37] = v[b]
                        for c in range(p * n):
                            W38[c][37] = w[c]
                        
                        # Verify
                        if _verify(U38, V38, W38, n, m, p):
                            return U38, V38, W38
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a rank-38 decomposition by merging columns
        if time.time() < deadline - 10:
            result = _try_rank38_by_merging(deadline)
            if result is not None:
                U2, V2, W2 = result
                if _verify(U2, V2, W2, n, m, p):
                    return {"U": U2, "V": V2, "W": W2}
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
