import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120

    n, m, p = DIMENSIONS

    # Rank 50 decomposition
    # Structure:
    # Two independent 2x2x7 blocks (one for j in {0,1}, one for j in {2,3}).
    # Each 2x2x7 block is decomposed into three 2x2x2 Strassen blocks (rank 7 each)
    # and one 2x2x1 Schoolbook block (rank 4).
    # Total rank per 2x2x7 block: 3*7 + 4 = 25.
    # Total rank: 2 * 25 = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0

    def emit_strassen(i0, j0, k0):
        """
        Emits a 2x2x2 Strassen block.
        A rows i0, i0+1; cols j0, j0+1.
        B rows j0, j0+1; cols k0, k0+1.
        C rows i0, i0+1; cols k0, k0+1.
        """
        nonlocal col
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + (i0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    def emit_schoolbook_2x2x1(j0, k0):
        """
        Emits a 2x2x1 schoolbook block.
        A rows 0, 1; cols j0, j0+1.
        B rows j0, j0+1; col k0.
        C rows 0, 1; col k0.
        """
        nonlocal col
        i0 = 0
        
        # T1: A11 * B11 -> C11
        U[i0*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # T2: A12 * B21 -> C11
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # T3: A21 * B11 -> C21
        U[(i0+1)*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        
        # T4: A22 * B21 -> C21
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1

    # Part 1: j in {0, 1} (A cols 0,1)
    # B rows 0,1 (B top half)
    # Output k in {0, 1, 2, 3, 4, 5, 6}
    
    # Block 1.1: 2x2x2 for k=0,1
    emit_strassen(0, 0, 0)
    # Block 1.2: 2x2x2 for k=2,3
    emit_strassen(0, 0, 2)
    # Block 1.3: 2x2x2 for k=4,5
    emit_strassen(0, 0, 4)
    # Block 1.4: 2x2x1 for k=6
    emit_schoolbook_2x2x1(0, 6)
    
    # Part 2: j in {2, 3} (A cols 2,3)
    # B rows 2,3 (B bottom half)
    # Output k in {0, 1, 2, 3, 4, 5, 6}
    
    # Block 2.1: 2x2x2 for k=0,1
    # Note: j0 is now 2 (cols 2,3 of A, rows 2,3 of B)
    emit_strassen(0, 2, 0)
    # Block 2.2: 2x2x2 for k=2,3
    emit_strassen(0, 2, 2)
    # Block 2.3: 2x2x2 for k=4,5
    emit_strassen(0, 2, 4)
    # Block 2.4: 2x2x1 for k=6
    emit_schoolbook_2x2x1(2, 6)
    
    # Verify base decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return None

    # Attempt rank reduction by checking for linear dependencies
    # Each rank-1 term t is represented by (U[:,t], V[:,t], W[:,t])
    # Two terms are linearly dependent if one is a scalar multiple of the other in the tensor sense
    # More generally, we check if the set of 50 tensor vectors is linearly dependent
    
    # Build tensor vectors: each term t gives a vector in R^(n*m * m*p * p*n) 
    # but that's huge. Instead, we can check dependencies via the matrix representation.
    
    # Alternative: for each pair of terms, check if they could be combined.
    # Actually, let's try a direct approach: form the "coefficient matrix" for each output entry.
    
    # For output (i,j,k), the coefficient is sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # This equals 1 for all (i,j,k).
    
    # To find dependencies, we can try: for each term t, check if removing it and adjusting
    # others can still satisfy all constraints. This is a linear system.
    
    # Simpler approach: check if any two terms have identical (U, V, W) patterns up to scaling
    # or if a term is redundant.
    
    # Let's try: for each term r, remove it and see if the remaining 49 terms can be rescaled
    # to still produce the correct output. This is a linear feasibility problem.
    
    # Even simpler: check pairwise if term a = c * term b for some scalar c.
    # If so, we can merge them.
    
    # Check for proportional terms
    proportional_pairs = []
    for a in range(rank):
        for b in range(a+1, rank):
            # Check if U[:,a] is proportional to U[:,b], V[:,a] to V[:,b], W[:,a] to W[:,b]
            # with the same scalar
            ua, ub = U[:, a], U[:, b]
            va, vb = V[:, a], V[:, b]
            wa, wb = W[:, a], W[:, b]
            
            # Find nonzero entries to determine scalar
            nonzero_a = np.nonzero(ua)[0]
            nonzero_b = np.nonzero(ub)[0]
            
            if len(nonzero_a) == 0 or len(nonzero_b) == 0:
                continue
                
            # Get scalar from U
            common = np.intersect1d(nonzero_a, nonzero_b)
            if len(common) == 0:
                continue
            
            # Check if ratios are consistent
            ratios_u = ua[common] / ub[common]
            if np.allclose(ratios_u, ratios_u[0], atol=1e-10):
                scalar = ratios_u[0]
                # Check V and W
                nonzero_va = np.nonzero(va)[0]
                nonzero_vb = np.nonzero(vb)[0]
                common_v = np.intersect1d(nonzero_va, nonzero_vb)
                if len(common_v) > 0:
                    ratios_v = va[common_v] / vb[common_v]
                    if np.allclose(ratios_v, ratios_v[0], atol=1e-10) and abs(ratios_v[0] - scalar) < 1e-10:
                        nonzero_wa = np.nonzero(wa)[0]
                        nonzero_wb = np.nonzero(wb)[0]
                        common_w = np.intersect1d(nonzero_wa, nonzero_wb)
                        if len(common_w) > 0:
                            ratios_w = wa[common_w] / wb[common_w]
                            if np.allclose(ratios_w, ratios_w[0], atol=1e-10) and abs(ratios_w[0] - scalar) < 1e-10:
                                proportional_pairs.append((a, b, scalar))
    
    # No proportional terms found in this structured decomposition, so rank stays at 50.
    # The Strassen blocks have distinct structures that don't allow simple merging.
    
    if time.time() > DEADLINE - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
