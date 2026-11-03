"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for smaller blocks.
Then applies peeling to remove linearly dependent rank-1 terms.
Target rank: 78 (possibly reducible).
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222_local(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        i0, j0, k0 are the starting indices.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Rank 25.
        """
        strassen_222_local(i0, j0, k0)
        strassen_222_local(i0, j0, k0 + 2)
        strassen_222_local(i0, j0, k0 + 4)
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m347_decomp():
        """
        Full 3x4 * 4x7 decomposition.
        Split n=3 into 1+2.
        n=1 part: 1x4 * 4x7. Split m=4 into 2+2. Two 1x2x7 blocks.
        n=2 part: 2x4 * 4x7. Split m=4 into 2+2. Two 2x2x7 blocks.
        """
        # n=1 part (i=0)
        # m split: j0=0 (size 2), j0=2 (size 2)
        m127_decomp(0, 0, 0)
        m127_decomp(0, 2, 0)
        
        # n=2 part (i=1,2)
        # m split: j0=0 (size 2), j0=2 (size 2)
        m227_decomp(1, 0, 0)
        m227_decomp(1, 2, 0)

    m347_decomp()

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Peeling: try to remove linearly dependent rank-1 terms
    def try_peel(U_mat, V_mat, W_mat):
        """
        Attempt to reduce rank by identifying and removing linearly dependent
        rank-1 terms. A term r is redundant if removing it still preserves
        the decomposition (i.e., the remaining terms still sum to 1 for all
        (i,j,k) where the original had 1, and 0 elsewhere).
        
        Key insight: term r is redundant if for ALL (i,j,k):
          sum_{r' != r} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] == delta_{i,j,k}
        where delta is 1 if (i,j,k) corresponds to a multiplication entry, 0 otherwise.
        But since our tensor IS the multiplication tensor, delta is always 1 for valid entries.
        Actually no - the tensor has 1 at positions (i,j,k) where the multiplication
        produces that entry, which is ALL (i,j,k) in range. So we need sum == 1 for all.
        
        Wait, that's wrong. The multiplication tensor T[i,j,k] = 1 for all i,j,k in range
        because (A*B)[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) triple contributes
        A[i,j]*B[j,k] to C[i,k]. The tensor is T[i,j,k] = 1 for all valid (i,j,k).
        
        So term r is redundant if: for all (i,j,k), sum_{r' != r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] == 1
        
        But that means the remaining terms must STILL produce the full tensor, which
        means term r must contribute 0 everywhere, i.e., it must be the zero term.
        
        That can't be right either. Let me reconsider.
        
        Actually, the correct condition is: the tensor decomposition requires
        sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all (i,j,k).
        
        Term r is redundant if we can find coefficients c_s (s != r) such that
        U[:,r] = sum_{s!=r} c_s * U[:,s], etc. But that's not how peeling works.
        
        The correct peeling: term r is removable if there exist coefficients
        alpha_s (s != r) such that for all (i,j,k):
          U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == sum_{s!=r} alpha_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
        
        This is equivalent to: the rank-1 tensor u_r ⊗ v_r ⊗ w_r lies in the span
        of {u_s ⊗ v_s ⊗ w_s : s != r}.
        
        A practical test: compute the full tensor T = sum_r u_r⊗v_r⊗w_r.
        Then T - u_r⊗v_r⊗w_r should equal sum_{s!=r} u_s⊗v_s⊗w_r.
        But T is always the multiplication tensor (all 1s), so T - u_r⊗v_r⊗w_r
        must equal sum_{s!=r} u_s⊗v_s⊗w_s.
        
        So the condition is: for all (i,j,k):
          1 - U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == sum_{s!=r} U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
        
        Which simplifies to: sum_{s!=r} ... == 1 - U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        This is still expensive. Let's use a different approach:
        Build the matrix M where M[(i,j,k), r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        Then term r is redundant if column r of M is in the span of the other columns.
        """
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        n_terms = U_curr.shape[1]
        
        # Precompute the tensor matrix M: shape (n*m*p, n_terms)
        # M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where idx = (i*m+j)*p + (j*p+k) ... actually let's use a flat index
        # idx = i * (m*p) + j * p + k
        
        total_entries = n * m * p
        M = np.zeros((total_entries, n_terms))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    for r in range(n_terms):
                        M[idx, r] = U_curr[i*m+j, r] * V_curr[j*p+k, r] * W_curr[k*n+i, r]
        
        removed = set()
        active_cols = list(range(n_terms))
        
        for r in range(n_terms):
            if time.time() > deadline - 15:
                break
            if r in removed:
                continue
            
            # Check if column r is in the span of other active columns
            # Use numpy lstsq or rank check
            other_cols = [c for c in active_cols if c != r]
            if len(other_cols) == 0:
                continue
            
            A = M[:, other_cols]
            b = M[:, r]
            
            # Check if b is in the column space of A
            # Use: ||A * lstsq(A, b) - b|| < tol
            try:
                coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                residual = np.linalg.norm(A @ coeffs - b)
                if residual < 1e-8:
                    removed.add(r)
                    active_cols = [c for c in active_cols if c != r]
            except:
                pass
        
        if not removed:
            return U_mat, V_mat, W_mat
        
        keep = [r for r in range(n_terms) if r not in removed]
        return U_curr[:, keep], V_curr[:, keep], W_curr[:, keep]

    # Apply peeling
    U, V, W = try_peel(U, V, W)

    rank = U.shape[1]
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    # If peeling broke something, fall back to original
                    U_orig = np.array(U_vecs).T
                    V_orig = np.array(V_vecs).T
                    W_orig = np.array(W_vecs).T
                    return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}
                    break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
