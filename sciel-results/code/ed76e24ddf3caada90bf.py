"""Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: Structural change from the current program's "build schoolbook then
greedily remove columns" approach to a "rank-1 term search" approach.

Key insight: The matrix multiplication tensor T[i,j,k] = delta_{j,j'} for
C[i,k] = sum_j A[i,j] B[j,k] can be decomposed as a sum of rank-1 terms.
Each rank-1 term is defined by (u_c, v_c, w_c) where:
  - u_c is a vector in R^{n*m} (the A-factor)
  - v_c is a vector in R^{m*p} (the B-factor)  
  - w_c is a vector in R^{p*n} (the C-factor)

The constraint is: for all (i,k,j,j'):
  sum_c w_c[k*n+i] * u_c[i*m+j] * v_c[j'*p+k] = delta_{j,j'}

This is a system of bilinear equations. We use a constructive approach:
1. Start with a known valid decomposition (schoolbook, rank 120)
2. Try to replace pairs of rank-1 terms with a single rank-1 term (rank reduction by 1)
3. Use the algebraic structure to find valid replacements

For 4x5x6, we use a block decomposition:
- View the 4x5 matrix as blocks: top 2 rows (4 cols) + bottom 2 rows (4 cols) + extra column
- View the 5x6 matrix as blocks: left 4 cols + right 2 cols
- Use Strassen for 2x2 blocks where possible

The structural change: instead of starting with rank 120 and removing columns,
we build up from known low-rank decompositions of sub-problems and combine them
via Kronecker products and direct sums, then optimize the combination.
"""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 rank-1 terms for 2x2 matrix multiplication.
    Convention: C[i,k] = sum_t W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t]
    For 2x2: n=m=p=2, so U[i*2+j], V[j*2+k], W[k*2+i]
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (A21+A22)B11
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = A11(B12-B22)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4 = A22(B21-B11)
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    # M5 = (A11+A12)B22
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # M6 = (A21-A11)(B11+B12)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (A12-A22)(B21+B22)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms


def _schoolbook_1xk(k):
    """Return k rank-1 terms for 1xk matrix multiplication (1xk * kx1 = 1x1).
    n=1, m=k, p=1.
    """
    terms = []
    for j in range(k):
        u = [0] * (1 * k)
        v = [0] * (k * 1)
        w = [0] * (1 * 1)
        u[j] = 1
        v[j] = 1
        w[0] = 1
        terms.append((u, v, w))
    return terms


def _build_block_decomposition(n=4, m=5, p=6):
    """Build a decomposition using block structure.
    
    Partition:
    - i: {0,1} (top 2 rows) and {2,3} (bottom 2 rows) -> 2 groups of 2
    - j: {0,1,2,3} (first 4 cols) and {4} (last col) -> group A (4) and group B (1)
    - k: {0,1,2,3} (first 4 cols) and {4,5} (last 2 cols) -> group X (4) and group Y (2)
    
    Blocks:
    1. i_top x j_A x k_X: 2x4x4 -> use 2*(2x2x2) Strassen: 2*7 = 14 terms
       Actually, 2x4x4: A is 2x4, B is 4x4, C is 2x4
       This is like two independent 2x4x2 multiplications? No.
       Better: view as A_top (2x4) * B_left (4x4) = C_top_left (2x4)
       We can use: for each pair of columns in B, use Strassen on the 2x2x2 sub-problem
       Split j into pairs: (0,1) and (2,3), split k into pairs: (0,1) and (2,3)
       Block (j_pair=0, k_pair=0): 2x2x2 -> Strassen (7 terms)
       Block (j_pair=0, k_pair=1): 2x2x2 -> Strassen (7 terms)
       Block (j_pair=1, k_pair=0): 2x2x2 -> Strassen (7 terms)
       Block (j_pair=1, k_pair=1): 2x2x2 -> Strassen (7 terms)
       Total: 28 terms
       
    2. i_top x j_A x k_Y: 2x4x2 -> A_top (2x4) * B_right (4x2) = C_top_right (2x2)
       Split j into pairs: (0,1) and (2,3)
       Block (j_pair=0): 2x2x2 -> Strassen (7 terms)
       Block (j_pair=1): 2x2x2 -> Strassen (7 terms)
       Total: 14 terms
       
    3. i_top x j_B x k_X: 2x1x4 -> A_top_last_col (2x1) * B_left_last_row (1x4) = C_top_left_last_col (2x4)
       This is an outer product: rank 2x4 = 8? No, it's a rank-1 times a row vector.
       Actually: A[i,4] * B[4,k] for i in {0,1}, k in {0,1,2,3}
       This is a rank-4 tensor (4 rank-1 terms, one for each k)
       Total: 4 terms
       
    4. i_top x j_B x k_Y: 2x1x2 -> A_top_last_col (2x1) * B_right_last_row (1x2) = C_top_right_last_col (2x2)
       Rank: 2 terms
       Total: 2 terms
       
    5. i_bottom x j_A x k_X: 2x4x4 -> same as block 1: 28 terms
    6. i_bottom x j_A x k_Y: 2x4x2 -> same as block 2: 14 terms
    7. i_bottom x j_B x k_X: 2x1x4 -> same as block 3: 4 terms
    8. i_bottom x j_B x k_Y: 2x1x2 -> same as block 4: 2 terms
    
    Total: 28 + 14 + 4 + 2 + 28 + 14 + 4 + 2 = 96 terms
    """
    R = 96
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    S = _strassen_2x2_terms()
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """Add a 2x2x2 Strassen block at offset (i0, j0, k0)."""
        nonlocal col
        for t in range(7):
            u2, v2, w2 = S[t]
            for ii in range(2):
                for jj in range(2):
                    U[(i0 + ii) * m + (j0 + jj)][col] += u2[ii * 2 + jj]
            for jj in range(2):
                for kk in range(2):
                    V[(j0 + jj) * p + (k0 + kk)][col] += v2[jj * 2 + kk]
            for kk in range(2):
                for ii in range(2):
                    W[(k0 + kk) * n + (i0 + ii)][col] += w2[kk * 2 + ii]
            col += 1
    
    # Block 1: i_top (0,1) x j_A (0,1,2,3) x k_X (0,1,2,3): 28 terms
    # 4 sub-blocks of 2x2x2
    for j_pair in range(2):
        for k_pair in range(2):
            add_strassen_block(0, j_pair * 2, k_pair * 2)
    
    # Block 2: i_top (0,1) x j_A (0,1,2,3) x k_Y (4,5): 14 terms
    # 2 sub-blocks of 2x2x2
    for j_pair in range(2):
        add_strassen_block(0, j_pair * 2, 4)
    
    # Block 3: i_top (0,1) x j_B (4) x k_X (0,1,2,3): 4 terms
    for k in range(4):
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
        col += 1
    
    # Block 4: i_top (0,1) x j_B (4) x k_Y (4,5): 2 terms
    for k in range(4, 6):
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
        col += 1
    
    # Block 5: i_bottom (2,3) x j_A (0,1,2,3) x k_X (0,1,2,3): 28 terms
    for j_pair in range(2):
        for k_pair in range(2):
            add_strassen_block(2, j_pair * 2, k_pair * 2)
    
    # Block 6: i_bottom (2,3) x j_A (0,1,2,3) x k_Y (4,5): 14 terms
    for j_pair in range(2):
        add_strassen_block(2, j_pair * 2, 4)
    
    # Block 7: i_bottom (2,3) x j_B (4) x k_X (0,1,2,3): 4 terms
    for k in range(4):
        for i in range(2, 4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
        col += 1
    
    # Block 8: i_bottom (2,3) x j_B (4) x k_Y (4,5): 2 terms
    for k in range(4, 6):
        for i in range(2, 4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
        col += 1
    
    return U, V, W, R


def _verify_numpy(U, V, W, n, m, p):
    """Verify the decomposition using numpy."""
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Verify using exact integer arithmetic."""
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = 0
                    for c in range(R):
                        wv = W[k * n + i][c]
                        uu = U[i * m + j][c]
                        vv = V[j2 * p + k][c]
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = 1 if j == j2 else 0
                    if s != target:
                        return False
    return True


def _try_pair_replacement(U, V, W, n, m, p, deadline):
    """Try to replace pairs of rank-1 terms with a single term.
    
    For columns c1, c2, we want to find a new column c_new such that:
    sum over (c1, c2, others) of W*U*V = sum over (c_new, others) of W*U*V
    
    This means: for all (i,k,j,j2):
    W1[i,k]*U1[i,j]*V1[j2,k] + W2[i,k]*U2[i,j]*V2[j2,k] 
    = W_new[i,k]*U_new[i,j]*V_new[j2,k]
    
    This is a system of bilinear equations. We try simple cases:
    1. If U1 = U2 and V1 = V2, then W_new = W1 + W2 (but this is just merging)
    2. If U1 = -U2, then we can cancel if W1 = W2 and V1 = V2
    3. More generally, try to find linear combinations
    """
    if np is None:
        return U, V, W
    
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    removed = set()
    changed = True
    while changed and time.time() < deadline - 30.0:
        changed = False
        R = len(U[0])
        
        for c1 in range(R):
            if c1 in removed:
                continue
            for c2 in range(c1 + 1, R):
                if c2 in removed:
                    continue
                if time.time() > deadline - 30.0:
                    break
                
                # Try to see if c1 and c2 can be merged into one term
                # Check if the sum of their contributions is rank-1
                # Contribution of c: W[:,c] * (U[:,c] outer V[:,c])
                # Sum: W1*U1oV1 + W2*U2oV2 should be rank-1
                
                # Build the tensor T[i,k,j,j2] = sum over {c1,c2} of W[i,k,c]*U[i,j,c]*V[j2,k,c]
                # For this to be rank-1 in the (i,j) vs (k,j2) sense, we need:
                # T[i,k,j,j2] = A[i,j] * B[k,j2] for some A, B
                
                # Simple check: if U1 == U2 and V1 == V2, merge
                if np.allclose(Ua[:, c1], Ua[:, c2]) and np.allclose(Va[:, c1], Va[:, c2]):
                    # Merge: new W = W1 + W2
                    new_W = Wa[:, c1] + Wa[:, c2]
                    # Check if new_W is non-zero
                    if np.any(new_W != 0):
                        # Replace c1 with new_W, remove c2
                        Ua[:, c1] = Ua[:, c1]  # unchanged
                        Va[:, c1] = Va[:, c1]  # unchanged
                        Wa[:, c1] = new_W
                        removed.add(c2)
                        changed = True
                        break
                
                # Check if U1 == -U2 and V1 == V2 and W1 == W2 (complete cancellation)
                if np.allclose(Ua[:, c1], -Ua[:, c2]) and np.allclose(Va[:, c1], Va[:, c2]) and np.allclose(Wa[:, c1], Wa[:, c2]):
                    # Both cancel out
                    removed.add(c1)
                    removed.add(c2)
                    changed = True
                    break
                
                # Check if we can scale: U1 = a*U2, V1 = b*V2, W1 = c*W2 with a*b*c = 1
                # This is harder, skip for now
            
            if changed:
                break
        
        if not changed:
            break
    
    if not removed:
        return U, V, W
    
    keep = [x for x in range(R) if x not in removed]
    U2 = Ua[:, keep].tolist()
    V2 = Va[:, keep].tolist()
    W2 = Wa[:, keep].tolist()
    return U2, V2, W2


def _try_column_removal(U, V, W, n, m, p, deadline):
    """Try to remove individual columns if they are redundant."""
    if np is None:
        return U, V, W
    
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    removed = set()
    changed = True
    while changed and time.time() < deadline - 30.0:
        changed = False
        R = len(U[0])
        
        for c in range(R):
            if c in removed:
                continue
            if time.time() > deadline - 30.0:
                break
            
            # Try removing column c
            keep = [x for x in range(R) if x != c and x not in removed]
            if len(keep) == 0:
                continue
            
            U2 = Ua[:, keep]
            V2 = Va[:, keep]
            W2 = Wa[:, keep]
            
            # Verify
            ok = True
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, m), dtype=np.float64)
                    for idx, col_idx in enumerate(keep):
                        wc = W2[k * n + i, idx]
                        if wc == 0:
                            continue
                        uc = U2[i * m: i * m + m, idx]
                        vc = V2[k * p: k * p + m, idx]
                        M += np.outer(uc, vc) * wc
                    if not np.allclose(M, np.eye(m), atol=1e-9):
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                removed.add(c)
                changed = True
    
    if not removed:
        return U, V, W
    
    keep = [x for x in range(R) if x not in removed]
    U2 = Ua[:, keep].tolist()
    V2 = Va[:, keep].tolist()
    W2 = Wa[:, keep].tolist()
    return U2, V2, W2


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    try:
        # Build initial block decomposition
        U, V, W, R = _build_block_decomposition(n, m, p)
        
        # Verify
        ok = _verify_numpy(U, V, W, n, m, p)
        
        if not ok:
            # Fallback: pure schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
        
        # Try to reduce rank via pair replacement
        U, V, W = _try_pair_replacement(U, V, W, n, m, p, deadline)
        
        # Try to remove redundant columns
        U, V, W = _try_column_removal(U, V, W, n, m, p, deadline)
        
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
