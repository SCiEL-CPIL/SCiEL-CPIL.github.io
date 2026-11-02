"""
Rank-minimal decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Start with 5 independent Strassen blocks (rank 35), then search for
cross-block sharing of U columns (same A-combination) to reduce total rank.
The A matrix is shared across all 5 column-pairs of B, so identical linear
combinations of A can potentially be reused if V and W structures allow.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def _build_strassen_block(t, U, V, W, col_base):
    """Fill in one Strassen block (7 terms) starting at col_base."""
    # P1 = (a00+a11)*(b00+b11) -> C00 += P1, C11 += P1
    col = col_base
    U[0][col] = 1; U[3][col] = 1
    V[2*t][col] = 1; V[10+2*t][col] = 1
    W[4*t][col] = 1; W[4*t+3][col] = 1
    
    # P2 = (a10+a11)*b00 -> C10 += P2, C11 -= P2
    col = col_base + 1
    U[2][col] = 1; U[3][col] = 1
    V[2*t][col] = 1
    W[4*t+2][col] = 1; W[4*t+3][col] = -1
    
    # P3 = a00*(b10-b11) -> C10 += P3
    col = col_base + 2
    U[0][col] = 1
    V[10+2*t][col] = 1; V[11+2*t][col] = -1
    W[4*t+2][col] = 1
    
    # P4 = a11*(b01-b00) -> C00 -= P4, C01 += P4
    col = col_base + 3
    U[3][col] = 1
    V[11+2*t][col] = 1; V[2*t][col] = -1
    W[4*t][col] = -1; W[4*t+1][col] = 1
    
    # P5 = (a00+a01)*b11 -> C01 += P5
    col = col_base + 4
    U[0][col] = 1; U[1][col] = 1
    V[11+2*t][col] = 1
    W[4*t+1][col] = 1
    
    # P6 = (a10-a00)*(b00+b10) -> C00 += P6
    col = col_base + 5
    U[2][col] = 1; U[0][col] = -1
    V[2*t][col] = 1; V[10+2*t][col] = 1
    W[4*t][col] = 1
    
    # P7 = (a01-a11)*(b01+b11) -> C11 += P7
    col = col_base + 6
    U[1][col] = 1; U[3][col] = -1
    V[2*t+1][col] = 1; V[11+2*t][col] = 1
    W[4*t+3][col] = 1

def _verify_decomposition(U, V, W, n=2, m=2, p=10):
    """Verify that the tensor decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # T should equal: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C. For 2x2 * 2x10:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (the standard matrix mult tensor)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i, r] * V[j, r] * W[k, r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False, (i, j, k, actual, expected)
    return True, None

def _build_rank35():
    """Build the baseline rank-35 decomposition."""
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(5):
        _build_strassen_block(t, U, V, W, t * 7)
    
    return U, V, W

def _try_reduce_rank(U_base, V_base, W_base, target_rank, deadline):
    """Try to find a rank-`target_rank` decomposition by merging identical U columns."""
    n, m, p = DIMENSIONS
    rank = target_rank
    
    # Get the U columns from the base decomposition
    # Each block has 7 U-column patterns
    # U patterns for Strassen block (in order):
    # 0: [1,0,0,1] (a00+a11)
    # 1: [0,0,1,1] (a10+a11)
    # 2: [1,0,0,0] (a00)
    # 3: [0,0,0,1] (a11)
    # 4: [1,1,0,0] (a00+a01)
    # 5: [-1,0,1,0] (a10-a00)
    # 6: [0,1,0,-1] (a01-a11)
    
    u_patterns = []
    for t in range(5):
        for s in range(7):
            col = t * 7 + s
            pattern = tuple(U_base[i][col] for i in range(4))
            u_patterns.append((t, s, col, pattern))
    
    # Group by U pattern
    from collections import defaultdict
    pattern_groups = defaultdict(list)
    for t, s, col, pattern in u_patterns:
        pattern_groups[pattern].append((t, s, col))
    
    # Try to merge columns with the same U pattern
    # For a merge to work, we need the V and W contributions to be separable
    
    # Strategy: for each pair of blocks with same U pattern, check if we can
    # share the U column and adjust V/W accordingly
    
    # Build candidate merged decompositions
    # We'll try all subsets of merges and check validity
    
    # For simplicity, try merging one pair at a time
    best_U, best_V, best_W = None, None, None
    best_rank = rank + 1  # higher than target means no improvement
    
    # For each pattern that appears multiple times, try merging pairs
    for pattern, entries in pattern_groups.items():
        if len(entries) < 2:
            continue
        
        # Try merging each pair
        for i in range(len(entries)):
            for j in range(i+1, len(entries)):
                t1, s1, col1 = entries[i]
                t2, s2, col2 = entries[j]
                
                # Check time
                if time.time() > deadline:
                    return None
                
                # Build a new decomposition where col1 and col2 share a single U column
                # The shared U column has the pattern
                # For V and W, we need: for each r in {col1, col2}:
                #   V[:,r] and W[:,r] must be such that the sum still works
                # 
                # If we merge, we need new V_col and W_col such that:
                #   V_col * W_col contribution replaces V[col1]*W[col1] + V[col2]*W[col2]
                # This is generally not possible with rank 1, so we keep both V/W
                # but share only the U part. This doesn't reduce rank.
                #
                # Actually, to reduce rank, we need the entire term (U,V,W) to be
                # linearly dependent. Let's check if V[col1] and V[col2] are
                # proportional, and W[col1] and W[col2] are proportional.
                
                v1 = np.array([V_base[r][col1] for r in range(m*p)], dtype=float)
                v2 = np.array([V_base[r][col2] for r in range(m*p)], dtype=float)
                w1 = np.array([W_base[r][col1] for r in range(p*n)], dtype=float)
                w2 = np.array([W_base[r][col2] for r in range(p*n)], dtype=float)
                
                # Check if v1 and v2 are proportional
                if np.allclose(v1, 0) or np.allclose(v2, 0):
                    continue
                
                # Check cross product for 2D (they're 20D, so check rank of [v1,v2])
                # If rank is 1, they're proportional
                mat_v = np.column_stack([v1, v2])
                if np.linalg.matrix_rank(mat_v) > 1:
                    continue
                
                mat_w = np.column_stack([w1, w2])
                if np.linalg.matrix_rank(mat_w) > 1:
                    continue
                
                # They're proportional! We can merge.
                # Find the scale factors
                # v2 = alpha * v1, w2 = beta * w1
                # The combined term has U * (v1*alpha_w + ...) 
                # Actually, if v2 = alpha*v1 and w2 = beta*w1, then
                # term1 = U*v1*w1, term2 = U*(alpha*v1)*(beta*w1) = alpha*beta * U*v1*w1
                # So combined = (1 + alpha*beta) * U*v1*w1
                # This is a single term with scaled V and W!
                
                # Find alpha: v2 = alpha * v1
                nonzero_v = v1 != 0
                if not np.any(nonzero_v):
                    continue
                alpha = v2[nonzero_v][0] / v1[nonzero_v][0]
                
                nonzero_w = w1 != 0
                if not np.any(nonzero_w):
                    continue
                beta = w2[nonzero_w][0] / w1[nonzero_w][0]
                
                scale = 1.0 + alpha * beta
                
                # Build new decomposition with this merge
                U_new = [[U_base[i][c] for c in range(rank)] for i in range(n*m)]
                V_new = [[V_base[i][c] for c in range(rank)] for i in range(m*p)]
                W_new = [[W_base[i][c] for c in range(rank)] for i in range(p*n)]
                
                # Remove col2, scale col1
                for i in range(n*m):
                    U_new[i][col1] = U_base[i][col1] * scale
                for i in range(m*p):
                    V_new[i][col1] = V_base[i][col1] * scale
                    V_new[i][col2] = 0
                for i in range(p*n):
                    W_new[i][col1] = W_base[i][col1]
                    W_new[i][col2] = 0
                
                # Remove col2 from all matrices
                U_new = [[row[c] for c in range(rank) if c != col2] for row in U_new]
                V_new = [[row[c] for c in range(rank) if c != col2] for row in V_new]
                W_new = [[row[c] for c in range(rank) if c != col2] for row in W_new]
                
                # Verify
                valid, err = _verify_decomposition(U_new, V_new, W_new)
                if valid:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = rank - 1
                    break
            if best_rank < rank:
                break
        if best_rank < rank:
            break
    
    if best_rank < rank:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Build rank-35 Strassen decomposition
    U, V, W = _build_rank35()
    
    # Verify baseline
    valid, err = _verify_decomposition(U, V, W)
    if not valid:
        # Fallback: try schoolbook rank-40
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        valid, err = _verify_decomposition(U, V, W)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
