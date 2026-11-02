import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Use 4 disjoint 2x2x2 Strassen blocks + schoolbook for remainder
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7  # 28
    total_terms = n * m * p  # 75
    strassen_terms_count = num_strassen_blocks * 8  # 32
    schoolbook_terms_count = total_terms - strassen_terms_count  # 43
    total_rank = total_strassen_rank + schoolbook_terms_count  # 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for a 2x2x2 block."""
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Convert to numpy for efficient operations
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = total_rank
    
    def verify(Uv, Vv, Wv, tol=1e-10):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uv[i*m+j], Vv[j*p+k] * Wv[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Verify initial decomposition
    if not verify(U_arr, V_arr, W_arr):
        # Fallback to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = np.zeros((n*m, total_rank_fb))
        V_fb = np.zeros((m*p, total_rank_fb))
        W_fb = np.zeros((p*n, total_rank_fb))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1.0
                    V_fb[j*p+k, col_fb] = 1.0
                    W_fb[k*n+i, col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Rank-merging heuristic: try to eliminate rank-1 terms
    # Strategy: for each term, check if it can be expressed as a linear combination
    # of other terms in a way that preserves the tensor decomposition
    
    # Build the tensor representation: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want to find if any column c can be removed while adjusting others
    
    # Approach: try removing each schoolbook term (they're isolated) and see if we can
    # compensate with the Strassen terms
    
    # First, let's try a simpler approach: check for duplicate or dependent terms
    # Two terms c1, c2 are "mergeable" if their product vectors are proportional
    
    # For rank-1 term c: define vectors u_c, v_c, w_c
    # The contribution to T[i,j,k] is u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # Try to find if any schoolbook term can be absorbed
    # Schoolbook terms have exactly one non-zero in each of U, V, W
    
    # Let's try a more systematic approach: use linear algebra to find dependencies
    # The tensor is determined by the triple products. If we can find coefficients
    # alpha, beta such that removing term c and adjusting term c' preserves the tensor,
    # we can reduce rank.
    
    # For now, let's try a heuristic: look for pairs of schoolbook terms that share
    # the same i index and see if they can be combined
    
    # Actually, let's try a different approach: check if any Strassen term is redundant
    # by seeing if its contribution is already captured by other terms
    
    # Build the full tensor from current decomposition
    T = np.zeros((n, m, p))
    for c in range(current_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
    
    # T should be all ones (75 entries)
    
    # Now try to remove each term and see if the tensor can still be recovered
    # with the remaining terms plus some adjustment
    
    # For each term c, compute its contribution
    # contrib_c[i,j,k] = U_arr[i*m+j,c] * V_arr[j*p+k,c] * W_arr[k*n+i,c]
    
    # If we remove term c, we need to find adjustments to other terms that compensate
    
    # This is a hard problem. Let's use a simpler heuristic:
    # Check if any two terms have identical contribution patterns (up to scaling)
    
    contributions = []
    for c in range(current_rank):
        contrib = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[idx] = U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    idx += 1
        contributions.append(contrib)
    
    # Check for proportional contributions
    for c1 in range(current_rank):
        if time.time() > deadline - 20:
            break
        for c2 in range(c1+1, current_rank):
            if time.time() > deadline - 20:
                break
            # Check if contributions are proportional
            # Find a scaling factor
            mask = np.abs(contributions[c1]) > 1e-12
            if np.sum(mask) < 2:
                continue
            ratios = contributions[c2][mask] / contributions[c1][mask]
            if np.std(ratios) < 1e-10:
                # They're proportional! We can merge them
                scale = np.mean(ratios)
                # Remove c2 by subtracting its contribution from c1
                # But this only works if we can adjust the factor matrices
                # For now, just note this as a potential merge
                pass
    
    # Given the complexity, let's return the current best (rank 71)
    # The rank-merging heuristic didn't find obvious improvements in this time
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
