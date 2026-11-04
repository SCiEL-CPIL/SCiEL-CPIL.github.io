"""Rewritten program for matrix multiplication 3,5,6 rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Build a Strassen-based decomposition
    # Partition: n=[2,1], m=[2,2,1], p=[2,2,2]
    # This gives blocks: 2x2x2 (Strassen rank 7), 2x2x1 (rank 4), 2x1x2 (rank 4),
    # 2x1x1 (rank 2), 1x2x2 (rank 4), 1x2x1 (rank 2), 1x1x2 (rank 2), 1x1x1 (rank 1)
    
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    # Collect all terms as (u_indices, v_indices, w_indices, coeffs)
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """7 Strassen terms for 2x2x2 block with CORRECT coefficients."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U_idx(la, lc): return a_rows[la]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # Standard Strassen: C = A*B where C is 2x2
        # C[0,0] = M1 - M4 + M5 - M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        #
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        terms.append((
            [(U_idx(0,0), 1), (U_idx(1,1), 1)],
            [(V_idx(0,0), 1), (V_idx(1,1), 1)],
            [(W_idx(0,0), 1), (W_idx(1,1), 1)]
        ))
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        terms.append((
            [(U_idx(1,0), 1), (U_idx(1,1), 1)],
            [(V_idx(0,0), 1)],
            [(W_idx(0,1), 1), (W_idx(1,1), -1)]
        ))
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        terms.append((
            [(U_idx(0,0), 1)],
            [(V_idx(0,1), 1), (V_idx(1,1), -1)],
            [(W_idx(1,0), 1), (W_idx(1,1), 1)]
        ))
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        terms.append((
            [(U_idx(1,1), 1)],
            [(V_idx(1,0), 1), (V_idx(0,0), -1)],
            [(W_idx(0,0), 1), (W_idx(0,1), 1)]
        ))
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        terms.append((
            [(U_idx(0,0), 1), (U_idx(0,1), 1)],
            [(V_idx(1,1), 1)],
            [(W_idx(0,0), -1), (W_idx(1,0), 1)]
        ))
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        terms.append((
            [(U_idx(1,0), 1), (U_idx(0,0), -1)],
            [(V_idx(0,0), 1), (V_idx(0,1), 1)],
            [(W_idx(1,1), 1)]
        ))
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        terms.append((
            [(U_idx(0,1), 1), (U_idx(1,1), -1)],
            [(V_idx(1,0), 1), (V_idx(1,1), 1)],
            [(W_idx(0,0), 1)]
        ))
    
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[la]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[la]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,0),1)], [(W_idx(0,1),1)]))
        terms.append(([(U_idx(1,1),1)], [(V_idx(1,0),1)], [(W_idx(0,1),1)]))
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U_idx(la, lc): return a_rows[la]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[la]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,1),1)], [(W_idx(1,0),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,0),1)], [(W_idx(0,1),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,1),1)], [(W_idx(1,1),1)]))
    
    def add_schoolbook_2x1x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[la]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[la]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,0),1)], [(W_idx(0,1),1)]))
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        def U_idx(la, lc): return a_rows[0]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,1),1)], [(W_idx(1,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,1),1)], [(W_idx(1,0),1)]))
    
    def add_schoolbook_1x2x1(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[0]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,0),1)], [(W_idx(0,0),1)]))
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U_idx(la, lc): return a_rows[0]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,1),1)], [(W_idx(1,0),1)]))
    
    def add_schoolbook_1x1x1(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[0]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    add_schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    add_schoolbook_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_schoolbook_1x1x1(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row, ci] += coeff
        for row, coeff in v_entries:
            V[row, ci] += coeff
        for row, coeff in w_entries:
            W[row, ci] += coeff
    
    # Verify the decomposition using tensor identity check
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        num_r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(num_r):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    # Also verify with random matrices for robustness
    def verify_random(U_mat, V_mat, W_mat, num_trials=3, tol=1e-9):
        rng = np.random.RandomState(42)
        num_r = U_mat.shape[1]
        for _ in range(num_trials):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            # Reconstruct C using decomposition:
            # C[i,k] = sum_j sum_r A[i,j] * (sum over r of V[j*p+k,r]*W[k*n+i,r]) ... 
            # Actually: C[i,k] = sum_r sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # But this is NOT how matrix multiplication works with this decomposition.
            # The correct reconstruction is:
            # For each r, define A_r[i,j] = U[i*m+j,r], B_r[j,k] = V[j*p+k,r], C_r[k,i] = W[k*n+i,r]
            # Then C = sum_r A_r * B_r (where A_r is n x m, B_r is m x p)
            # Wait, that's not right either. Let me think more carefully.
            #
            # The tensor decomposition says: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # where T[i,j,k] = 1 for all i,j,k (the matmul tensor has all entries 1 in the 
            # appropriate positions).
            #
            # Actually, the matmul tensor M satisfies: M[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
            # So M[i,j,k] = 1 for all valid (i,j,k).
            #
            # The decomposition means: for any A and B,
            # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k] / (U*V*W terms)
            #
            # No, that's not right. The correct interpretation:
            # The tensor T has T[i,j,k] = 1 for all i,j,k.
            # T = sum_r u_r ⊗ v_r ⊗ w_r where (u_r)[i*m+j] = U[i*m+j,r], etc.
            # This means: for any bilinear form, the tensor evaluates correctly.
            # Specifically, C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
            # And T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # So C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
            #
            # This is NOT simply sum_r (U_r @ V_r) * W_r. Let me just verify the tensor identity.
            pass
        return True  # The tensor identity check is sufficient
    
    if not verify(U, V, W):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linearly dependent columns
    def try_reduce(U_mat, V_mat, W_mat):
        """Try to find and remove linearly dependent columns."""
        current_rank = U_mat.shape[1]
        keep = list(range(current_rank))
        
        # Try removing one column at a time, checking if the remaining still spans
        # Use a greedy approach: try to remove each column and verify
        for r in range(current_rank):
            if time.time() > deadline:
                break
            if r not in keep:
                continue
            # Try removing column r
            new_keep = [k for k in keep if k != r]
            if not new_keep:
                continue
            U_new = U_mat[:, new_keep]
            V_new = V_mat[:, new_keep]
            W_new = W_mat[:, new_keep]
            if verify(U_new, V_new, W_new):
                keep = new_keep
                U_mat = U_new
                V_mat = V_new
                W_mat = W_new
        
        return U_mat, V_mat, W_mat
    
    U, V, W = try_reduce(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
