"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i*m+j]
                vj = V[j*p+k]
                wk = W[k*n+i]
                for c in range(rank):
                    s += ui[c] * vj[c] * wk[c]
                if s != 1.0:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using NumPy if available."""
    import numpy as np
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _finite_field_sparsity(U, V, W, n, m, p):
    """
    Use F2 analysis to identify which columns are most redundant.
    A column is more redundant if its pattern appears in other columns' span.
    Returns a list of column indices sorted by redundancy (most redundant first).
    """
    rank = len(U[0])
    # For each column, compute a hash of its support pattern
    # Columns with similar patterns are more likely to be redundant
    col_hashes = []
    for c in range(rank):
        # Get the pattern of which U, V, W entries are nonzero
        u_pattern = tuple(1 if U[i][c] != 0 else 0 for i in range(n*m))
        v_pattern = tuple(1 if V[i][c] != 0 else 0 for i in range(m*p))
        w_pattern = tuple(1 if W[i][c] != 0 else 0 for i in range(p*n))
        col_hashes.append((u_pattern, v_pattern, w_pattern))
    
    # Count how many other columns share similar patterns
    redundancy_scores = [0] * rank
    for c1 in range(rank):
        for c2 in range(c1+1, rank):
            # Check similarity
            h1 = col_hashes[c1]
            h2 = col_hashes[c2]
            # Count matching nonzero positions
            sim = 0
            for a, b in zip(h1[0], h2[0]):
                if a == b: sim += 1
            for a, b in zip(h1[1], h2[1]):
                if a == b: sim += 1
            for a, b in zip(h1[2], h2[2]):
                if a == b: sim += 1
            total = n*m + m*p + p*n
            if sim > total * 0.8:  # High similarity
                redundancy_scores[c1] += 1
                redundancy_scores[c2] += 1
    
    # Sort by redundancy (most redundant first)
    order = sorted(range(rank), key=lambda c: -redundancy_scores[c])
    return order

def _build_strassen_init(n, m, p):
    """
    Build a Strassen-based initialization for 3x4 x 4x4.
    Split the 4x4 matrix into 2x2 blocks and use Strassen (rank 7 per block).
    For 3x4: we have 3 rows of A, each 4 wide. Split into two 2-wide halves.
    B is 4x4, split into four 2x2 blocks.
    
    A is 3x4: rows 0-2, cols 0-3
    Split A into A1 (3x2, cols 0-1) and A2 (3x2, cols 2-3)
    Split B into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2)
    
    C = A*B is 3x4
    C[:,0:2] = A1*B11 + A2*B21
    C[:,2:3] = A1*B12 + A2*B22
    
    For each 3x2 * 2x2 multiplication, we can use a rank-6 or so decomposition.
    Actually, let's just use schoolbook for the sub-blocks but with Strassen for the 2x2 parts.
    
    Simpler approach: Use Strassen on the 2x2 blocks of B, treating A as having 3 rows.
    For a 3x2 * 2x2 product, we need 3*2*2=12 schoolbook mults, but we can do better.
    
    Let me use a different approach: block decomposition with Strassen on 2x2 blocks.
    """
    # For simplicity and correctness, let's build a rank-36 decomposition:
    # Split 4x4 into four 2x2 blocks.
    # C (3x4) = A (3x4) * B (4x4)
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    #
    # Each term is (3x2)*(2x2) -> 3x2, requiring 12 schoolbook mults
    # Total: 4 * 12 = 48 (no improvement)
    #
    # Let's use Strassen on the 2x2 B blocks:
    # For B, compute Strassen products: M1..M7 (each 2x2)
    # Then express B blocks in terms of these.
    #
    # Actually, the cleanest approach: use the known rank-23 Laderman for 3x3x3
    # and add the remaining terms.
    #
    # Let me just return schoolbook rank 48 but with a better column ordering.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build schoolbook initialization
    U, V, W = _build_strassen_init(n, m, p)
    rank = len(U[0])
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Use finite-field sparsity to order columns
    if time.time() < deadline:
        order = _finite_field_sparsity(U, V, W, n, m, p)
    else:
        order = list(range(rank))
    
    # Greedy column removal, trying columns in order of redundancy
    while time.time() < deadline:
        found = False
        current_rank = len(U[0])
        for c in order:
            if time.time() > deadline:
                break
            if c >= current_rank:
                continue
            # Try removing column c
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                found = True
                if len(U[0]) < best_rank:
                    best_rank = len(U[0])
                    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                # Update order: remove c from order and re-sort
                order = [x for x in order if x != c]
                # Re-index: columns after c shift left by 1
                order = [x-1 if x > c else x for x in order]
                # Re-compute sparsity order for remaining columns
                if time.time() < deadline:
                    order = _finite_field_sparsity(U, V, W, n, m, p)
                break
        if not found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
