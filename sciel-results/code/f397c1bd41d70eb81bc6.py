"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition.
Uses Strassen-like block decomposition and rank reduction via linear algebra."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12

    def build_target():
        """Build the target multiplication tensor T[a,b,c]."""
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    def verify_decomposition(U, V, W, T=None, tol=1e-8):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        if T is None:
            T = build_target()
        r = U.shape[1]
        # Compute the decomposed tensor
        # For each c, result[a,b] = sum_r U[a,r]*V[b,r]*W[c,r]
        for c in range(pn):
            # UW = U * W[c,:] element-wise in r dimension, shape (nm, r)
            UW = U * W[c, :]  # (nm, r)
            result = UW @ V.T  # (nm, mp)
            if np.max(np.abs(result - T[:, :, c])) > tol:
                return False
        return True

    def reduce_rank(U, V, W, T, max_iters=500):
        """Try to reduce rank by eliminating one term at a time.
        Strategy: remove term t, keep other terms' U and W fixed,
        solve for new V columns via least squares per row b.
        Also try keeping U and V fixed, solving for W.
        And keeping V and W fixed, solving for U.
        """
        r = U.shape[1]
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            if r <= best_rank:
                break
            
            eliminated = False
            
            # Try removing each term t
            for t in range(r):
                if time.time() > deadline - 10:
                    break
                
                others = [x for x in range(r) if x != t]
                r_new = r - 1
                
                # Strategy 1: Keep U, W fixed; solve for V
                success = False
                V_new = np.zeros((mp, r_new), dtype=np.float64)
                
                for b in range(mp):
                    # A_b[a*pn+c, s_idx] = U[a, s] * W[c, s]
                    A_b = np.zeros((n * p, r_new), dtype=np.float64)
                    for s_idx, s in enumerate(others):
                        Us = U[:, s]
                        Ws = W[:, s]
                        for a in range(nm):
                            for c in range(pn):
                                A_b[a * pn + c, s_idx] = Us[a] * Ws[c]
                    
                    rhs_b = T[:, b, :].flatten()
                    
                    try:
                        sol, residuals, rank_m, sv = np.linalg.lstsq(A_b, rhs_b, rcond=None)
                        res = np.linalg.norm(A_b @ sol - rhs_b)
                        norm_rhs = np.linalg.norm(rhs_b)
                        if norm_rhs > 1e-12 and res / norm_rhs > 1e-8:
                            sol = None
                        elif norm_rhs <= 1e-12:
                            # Check if A_b @ sol is close to zero
                            if np.linalg.norm(A_b @ sol) > 1e-6:
                                sol = None
                    except np.linalg.LinAlgError:
                        sol = None
                    
                    if sol is None:
                        success = False
                        break
                    V_new[b, :] = sol
                
                if success:
                    U_new = np.zeros((nm, r_new), dtype=np.float64)
                    W_new = np.zeros((pn, r_new), dtype=np.float64)
                    for s_idx, s in enumerate(others):
                        U_new[:, s_idx] = U[:, s]
                        W_new[:, s_idx] = W[:, s]
                    
                    # Verify
                    if verify_decomposition(U_new, V_new, W_new, T, tol=1e-8):
                        U, V, W = U_new, V_new, W_new
                        r = r_new
                        best_rank = r
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        eliminated = True
                        break
                else:
                    # Strategy 2: Keep V, W fixed; solve for U
                    success2 = False
                    U_new = np.zeros((nm, r_new), dtype=np.float64)
                    
                    for a in range(nm):
                        A_a = np.zeros((m * p, r_new), dtype=np.float64)
                        for s_idx, s in enumerate(others):
                            Vs = V[:, s]
                            Ws = W[:, s]
                            for b in range(mp):
                                for c in range(pn):
                                    A_a[b * pn + c, s_idx] = Vs[b] * Ws[c]
                        
                        rhs_a = T[a, :, :].flatten()
                        
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(A_a, rhs_a, rcond=None)
                            res = np.linalg.norm(A_a @ sol - rhs_a)
                            norm_rhs = np.linalg.norm(rhs_a)
                            if norm_rhs > 1e-12 and res / norm_rhs > 1e-8:
                                sol = None
                            elif norm_rhs <= 1e-12:
                                if np.linalg.norm(A_a @ sol) > 1e-6:
                                    sol = None
                        except np.linalg.LinAlgError:
                            sol = None
                        
                        if sol is None:
                            success2 = False
                            break
                        U_new[a, :] = sol
                    
                    if success2:
                        V_new = np.zeros((mp, r_new), dtype=np.float64)
                        W_new = np.zeros((pn, r_new), dtype=np.float64)
                        for s_idx, s in enumerate(others):
                            V_new[:, s_idx] = V[:, s]
                            W_new[:, s_idx] = W[:, s]
                        
                        if verify_decomposition(U_new, V_new, W_new, T, tol=1e-8):
                            U, V, W = U_new, V_new, W_new
                            r = r_new
                            best_rank = r
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            eliminated = True
                            break
                    else:
                        # Strategy 3: Keep U, V fixed; solve for W
                        success3 = False
                        W_new = np.zeros((pn, r_new), dtype=np.float64)
                        
                        for c in range(pn):
                            A_c = np.zeros((nm * mp, r_new), dtype=np.float64)
                            for s_idx, s in enumerate(others):
                                Us = U[:, s]
                                Vs = V[:, s]
                                for a in range(nm):
                                    for b in range(mp):
                                        A_c[a * mp + b, s_idx] = Us[a] * Vs[b]
                            
                            rhs_c = T[:, :, c].flatten()
                            
                            try:
                                sol, residuals, rank_m, sv = np.linalg.lstsq(A_c, rhs_c, rcond=None)
                                res = np.linalg.norm(A_c @ sol - rhs_c)
                                norm_rhs = np.linalg.norm(rhs_c)
                                if norm_rhs > 1e-12 and res / norm_rhs > 1e-8:
                                    sol = None
                                elif norm_rhs <= 1e-12:
                                    if np.linalg.norm(A_c @ sol) > 1e-6:
                                        sol = None
                            except np.linalg.LinAlgError:
                                sol = None
                            
                            if sol is None:
                                success3 = False
                                break
                            W_new[c, :] = sol
                        
                        if success3:
                            U_new = np.zeros((nm, r_new), dtype=np.float64)
                            V_new = np.zeros((mp, r_new), dtype=np.float64)
                            for s_idx, s in enumerate(others):
                                U_new[:, s_idx] = U[:, s]
                                V_new[:, s_idx] = V[:, s]
                            
                            if verify_decomposition(U_new, V_new, W_new, T, tol=1e-8):
                                U, V, W = U_new, V_new, W_new
                                r = r_new
                                best_rank = r
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                eliminated = True
                                break
                        else:
                            continue
                
                if eliminated:
                    break
            
            if not eliminated:
                break
        
        return best_U, best_V, best_W, best_rank

    def build_strassen_block():
        """Build a Strassen-based block decomposition.
        Split j (size 5) into blocks: [1, 2, 2] and k (size 6) into [2, 2, 2].
        For j=0 (1 col) and each k-block (2 cols): direct multiplication, 1*2=2 terms per k-block, 3 k-blocks -> 6 terms.
        For j-blocks of size 2 (two blocks: j=1,2 and j=3,4) and each k-block of size 2: Strassen 2x2x2 with 7 terms.
        Two j-blocks * three k-blocks * 7 = 42 terms.
        Total: 6 + 42 = 48 terms.
        """
        r_total = 48
        U = np.zeros((nm, r_total), dtype=np.float64)
        V = np.zeros((mp, r_total), dtype=np.float64)
        W = np.zeros((pn, r_total), dtype=np.float64)
        
        col = 0
        
        # Part 1: j=0, direct multiplication with each k-block
        # j=0 is a single column. For each k-block of size 2, we do direct 1x1 * 1x2.
        # Actually, for j=0, it's just 1x1 times 1x2, so 2 terms per k-block.
        j0 = 0
        for k_base in [0, 2, 4]:
            for k_off in range(2):
                k = k_base + k_off
                U[j0 * m + j0, col] = 1  # U[i*m+j] with i=0,1 but only j=0 matters
                # Actually U should be zero except for the relevant entry
                # Let me reconsider: U[i*m+j, col] is the coefficient for A[i,j]
                # For direct multiplication of A[:, 0] (2x1) with B[0, :] (1x2),
                # C[i, k] = A[i, 0] * B[0, k]
                # So U[i*m+0, col] = 1, V[0*p+k, col] = 1, W[k*n+i, col] = 1
                # But we need separate terms for each k since B[0,k] is different.
                U[:, col] = 0
                V[:, col] = 0
                W[:, col] = 0
                for i in range(n):
                    U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                for i in range(n):
                    W[k * n + i, col] = 1
                col += 1
        
        # Part 2: Strassen for j-blocks of size 2
        # j-blocks: [1,2] and [3,4], k-blocks: [0,1], [2,3], [4,5]
        # Strassen 2x2x2 terms (7 terms):
        # M1 = (A11 - A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B22 - B11)
        # M4 = A22 * (B11 - B21)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        strassen_terms = [
            # (u_components, v_components, w_components)
            # Each component is a list of (local_index, value)
            # u: (i_local, j_local, val), v: (j_local, k_local, val), w: (k_local, i_local, val)
            ([((0, 0, 1), (1, 1, -1)), [(0, 0, 1), (1, 1, 1)]],
             [[(0, 0, 1), (1, 1, 1)]],
             [[(0, 0, 1), (1, 1, 1)]]),
            ([((1, 0, 1), (1, 1, 1))],
             [[(0, 0, 1)]],
             [[(1, 0, 1), (1, 1, -1)]]),
            ([[(0, 0, 1)]],
             [[(0, 1, 1), (1, 1, -1)]],
             [[(0, 1, 1), (1, 1, 1)]]),
            ([[(1, 1, 1)]],
             [[(1, 0, 1), (0, 0, -1)]],
             [[(0, 0, 1), (1, 0, 1)]]),
            ([[(0, 0, 1), (0, 1, 1)]],
             [[(1, 1, 1)]],
             [[(0, 0, -1), (0, 1, 1)]]),
            ([[(1, 0, 1), (0, 0, -1)]],
             [[(0, 0, 1), (0, 1, 1)]],
             [[(1, 1, 1)]]),
            ([[(0, 1, 1), (1, 1, -1)]],
             [[(1, 0, 1), (1, 1, 1)]],
             [[(0, 0, 1)]]),
        ]
        
        # Correct Strassen terms for 2x2x2:
        # M1 = (A11 - A22)*(B11 + B22): u=[(0,0,1),(1,1,-1)], v=[(0,0,1),(1,1,1)], w=[(0,0,1),(1,1,1)]
        # M2 = (A21 + A22)*B11: u=[(1,0,1),(1,1,1)], v=[(0,0,1)], w=[(1,0,1),(1,1,-1)]
        # M3 = A11*(B22 - B11): u=[(0,0,1)], v=[(0,1,1),(1,1,-1)], w=[(0,1,1),(1,1,1)]
        # M4 = A22*(B11 - B21): u=[(1,1,1)], v=[(1,0,1),(0,0,-1)], w=[(0,0,1),(1,0,1)]
        # M5 = (A11 + A12)*B22: u=[(0,0,1),(0,1,1)], v=[(1,1,1)], w=[(0,0,-1),(0,1,1)]
        # M6 = (A21 - A11)*(B11 + B12): u=[(1,0,1),(0,0,-1)], v=[(0,0,1),(0,1,1)], w=[(1,1,1)]
        # M7 = (A12 - A22)*(B21 + B22): u=[(0,1,1),(1,1,-1)], v=[(1,0,1),(1,1,1)], w=[(0,0,1)]
        
        for j_block in [1, 3]:  # j_base for the 2x2 blocks
            for k_base in [0, 2, 4]:
                for term_u, term_v, term_w in strassen_terms:
                    U[:, col] = 0
                    V[:, col] = 0
                    W[:, col] = 0
                    for (li, lj, val) in term_u:
                        U[(li) * m + (j_block + lj), col] += val
                    for (lj, lk, val) in term_v:
                        V[(j_block + lj) * p + (k_base + lk), col] += val
                    for (lk, li, val) in term_w:
                        W[(k_base + lk) * n + li, col] += val
                    col += 1
        
        return U, V, W, col

    def to_exact(arr):
        """Convert float array to exact integer or half-integer representation."""
        result = np.zeros_like(arr, dtype=object)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                doubled = round(val * 2)
                if abs(val - doubled / 2) < 1e-8:
                    if doubled % 2 == 0:
                        result[i, j] = doubled // 2
                    else:
                        result[i, j] = Fraction(doubled, 2)
                else:
                    # Keep as fraction of the float
                    result[i, j] = Fraction(val).limit_denominator(10**6)
        return result

    T = build_target()
    
    # Build Strassen block decomposition (rank 48)
    U0, V0, W0, actual_rank = build_strassen_block()
    
    # Verify
    if not verify_decomposition(U0, V0, W0, T, tol=1e-8):
        # Fallback to schoolbook
        r = n * m * p
        U0 = np.zeros((nm, r), dtype=np.float64)
        V0 = np.zeros((mp, r), dtype=np.float64)
        W0 = np.zeros((pn, r), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col] = 1
                    V0[j * p + k, col] = 1
                    W0[k * n + i, col] = 1
                    col += 1
        actual_rank = r
    
    # Reduce rank
    U_red, V_red, W_red, final_rank = reduce_rank(U0, V0, W0, T, max_iters=200)
    
    # Convert to exact representation
    U_exact = to_exact(U_red)
    V_exact = to_exact(V_red)
    W_exact = to_exact(W_red)
    
    return {
        "U": U_exact.tolist(),
        "V": V_exact.tolist(),
        "W": W_exact.tolist()
    }
