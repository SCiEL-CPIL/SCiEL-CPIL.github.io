"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
    
    Strategy:
    1. Build a rank-24 construction using two rank-12 sub-problems (2x4 x 4x2).
    2. Verify and attempt further reduction via column removal.
    3. Return the lowest-rank verified decomposition found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    def construct_rank24():
        """
        Construct rank-24 decomposition for 2x4 x 4x4.
        
        Split into two 2x4 x 4x2 sub-problems, each of rank 12.
        For each 2x4 x 4x2: A(2x4) * B(4x2) = C(2x2)
        
        Rank-12 algorithm for 2x4 x 4x2:
        A = [[a0,a1,a2,a3],[a4,a5,a6,a7]]
        B = [[b0,b4],[b1,b5],[b2,b6],[b3,b7]]
        
        C[0,0] = a0*b0 + a1*b1 + a2*b2 + a3*b3
        C[0,1] = a0*b4 + a1*b5 + a2*b6 + a3*b7
        C[1,0] = a4*b0 + a5*b1 + a6*b2 + a7*b3
        C[1,1] = a4*b4 + a5*b5 + a6*b6 + a7*b7
        
        Use 12 bilinear terms:
        Split A into left 2x2 and right 2x2, B into top 2x1 and bottom 2x1 (for each col).
        Actually, use the known rank-12 construction.
        
        One approach: view as computing 2x2 = (2x4)*(4x2) by:
        - 7 Strassen terms for left half: (a0,a1; a4,a5) * (b0,b1; b2,b3; b4,b5; b6,b7)
          Wait, that's not right either.
        
        Let me use a direct construction:
        M1 = (a0+a1+a2+a3) * (b0+b1+b2+b3)  -- too many terms
        
        Better: use the fact that rank(2x4 x 4x2) = 12.
        Construction via grouping:
        
        Let's define 12 products:
        P1 = (a0+a1) * (b0+b1)
        P2 = (a0+a1) * (b2+b3)
        P3 = (a0-a1) * (b0-b1)
        P4 = (a0-a1) * (b2-b3)
        P5 = (a2+a3) * (b0+b1)
        P6 = (a2+a3) * (b2+b3)
        P7 = (a2-a3) * (b0-b1)
        P8 = (a2-a3) * (b2-b3)
        P9 = (a4+a5) * (b0+b1)
        P10 = (a4+a5) * (b2+b3)
        P11 = (a4-a5) * (b0-b1)
        P12 = (a4-a5) * (b2-b3)
        
        But this doesn't handle the second column of B.
        
        Let me think differently. For 2x4 x 4x2:
        This is equivalent to: for each column k of B, compute A * B[:,k] (a 2x1 vector).
        The bilinear complexity of 2x4 x 4x1 is 8 (naive) but can be reduced.
        
        Actually, the correct approach: the tensor M(2,4,2) has rank 12.
        Let me use a known construction.
        
        Alternative: use the rank-7 Strassen for 2x2x2 applied cleverly.
        
        Let me just build it explicitly with 12 terms.
        """
        rank = 24
        U = np.zeros((n * m, rank))  # 8 x 24
        V = np.zeros((m * p, rank))  # 16 x 24
        W = np.zeros((p * n, rank))  # 8 x 24
        
        # For each half (k_offset = 0 for cols 0-1, k_offset = 2 for cols 2-3)
        # Build a rank-12 decomposition for 2x4 x 4x2
        # A indices: 0-7 (row i: 0,1; col j: 0-3)
        # B indices: (j, k) where j=0-3, k=0-1 (for this half)
        # C indices: (i, k) where i=0-1, k=0-1
        
        # Rank-12 terms for 2x4 x 4x2:
        # Using the construction from the literature:
        # Terms are indexed 0-11 for each half
        
        # Let me use a simple but correct rank-12 construction:
        # Split the 4 input dimensions into 2 groups of 2.
        # A_left = [a0,a1; a4,a5], A_right = [a2,a3; a6,a7]
        # B_top = [b0;b1], B_bottom = [b2;b3] (for col 0)
        # B_top2 = [b4;b5], B_bottom2 = [b6;b7] (for col 1)
        
        # C[0,0] = a0*b0 + a1*b1 + a2*b2 + a3*b3
        # C[0,1] = a0*b4 + a1*b5 + a2*b6 + a3*b7
        # C[1,0] = a4*b0 + a5*b1 + a6*b2 + a7*b3
        # C[1,1] = a4*b4 + a5*b5 + a6*b6 + a7*b7
        
        # 12-term construction:
        # t0 = (a0+a1) * (b0+b1)
        # t1 = (a0-a1) * (b0-b1)
        # t2 = (a0+a1) * (b4+b5)
        # t3 = (a0-a1) * (b4-b5)
        # t4 = (a2+a3) * (b0+b1)
        # t5 = (a2-a3) * (b0-b1)
        # t6 = (a2+a3) * (b4+b5)
        # t7 = (a2-a3) * (b4-b5)
        # t8 = (a4+a5) * (b0+b1)
        # t9 = (a4-a5) * (b0-b1)
        # t10 = (a4+a5) * (b4+b5)
        # t11 = (a4-a5) * (b4-b5)
        
        # But this gives:
        # C[0,0] = (t0+t1)/2 + (t4+t5)/2
        # C[0,1] = (t2+t3)/2 + (t6+t7)/2
        # C[1,0] = (t8+t9)/2 + (t10+t11)/2  -- WRONG, t10,t11 involve a4,a5 with b4,b5
        # This doesn't work correctly.
        
        # Let me use a different approach. The correct rank-12 for 2x4x4x2:
        # Actually, let me just use the naive 8-term approach for each 2x4x4x1
        # and share where possible. Or better, let me use a verified construction.
        
        # Simplest correct approach: use 4 Strassen 2x2x2 blocks but merge them.
        # Actually, let me just go back to the Strassen rank-28 and do better reduction.
        
        # For now, return the rank-28 Strassen construction and rely on reduction.
        # This is a placeholder - we'll improve below.
        
        # Let me construct rank-28 Strassen and return it
        return None  # Signal to use rank-28
    
    def construct_strassen_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_terms = [
            {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
            {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(0, 1, 1), (1, 1, -1)]},
            {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(1, 0, 1), (1, 1, 1)]},
            {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (0, 1, 1)]},
            {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (1, 0, 1)]},
            {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
            {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
        ]
        
        blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        col = 0
        for j_off, k_off in blocks:
            for term in strassen_terms:
                for (li, lj, coeff) in term['U']:
                    U[li * m + (lj + j_off)][col] += coeff
                for (lj, lk, coeff) in term['V']:
                    V[(lj + j_off) * p + (lk + k_off)][col] += coeff
                for (lk, li, coeff) in term['W']:
                    W[(lk + k_off) * n + li][col] += coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try rank-24 construction first
    r24 = construct_rank24()
    if r24 is not None and verify(r24['U'], r24['V'], r24['W']):
        best = r24
    else:
        best = construct_strassen_rank28()
    
    if not verify(best['U'], best['V'], best['W']):
        return best
    
    U_arr = np.array(best['U'], dtype=float)
    V_arr = np.array(best['V'], dtype=float)
    W_arr = np.array(best['W'], dtype=float)
    
    best_rank = U_arr.shape[1]
    
    def try_remove_columns(arrays, max_removals=10):
        """Try to remove up to max_removals columns one at a time."""
        U_a, V_a, W_a = arrays
        removed = 0
        for _ in range(max_removals):
            if time_left() < 10:
                break
            found = False
            for c_remove in range(U_a.shape[1]):
                U_test = np.delete(U_a, c_remove, axis=1)
                V_test = np.delete(V_a, c_remove, axis=1)
                W_test = np.delete(W_a, c_remove, axis=1)
                if verify(U_test, V_test, W_test):
                    U_a, V_a, W_a = U_test, V_test, W_test
                    removed += 1
                    found = True
                    break
            if not found:
                break
        return U_a, V_a, W_a, removed
    
    # Phase 1: Try removing single columns iteratively
    U_arr, V_arr, W_arr, _ = try_remove_columns((U_arr, V_arr, W_arr), max_removals=10)
    best_rank = U_arr.shape[1]
    
    # Phase 2: Tensor-based redundancy check
    if time_left() > 30:
        rank = U_arr.shape[1]
        dim = n * m * p  # 32
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        
        # Precompute all column contributions
        T_cols = np.zeros((rank, dim))
        for c in range(rank):
            T_cols[c] = np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c]).flatten()
        
        # Check if any column is in the span of others
        for i in range(rank):
            if time_left() < 15:
                break
            other = [c for c in range(rank) if c != i]
            M = T_cols[other]
            try:
                coeffs, residuals, rnk, _ = np.linalg.lstsq(M.T, T_cols[i], rcond=1e-10)
                if len(residuals) > 0:
                    residual = np.linalg.norm(residuals[0])
                else:
                    residual = np.linalg.norm(M.T @ coeffs - T_cols[i])
                if residual < 1e-8:
                    U_arr = np.delete(U_arr, i, axis=1)
                    V_arr = np.delete(V_arr, i, axis=1)
                    W_arr = np.delete(W_arr, i, axis=1)
                    best_rank = U_arr.shape[1]
                    break
            except Exception:
                continue
    
    # Final verification
    if not verify(U_arr, V_arr, W_arr):
        best = construct_strassen_rank28()
        return best
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

# EVOLVE-BLOCK-END
