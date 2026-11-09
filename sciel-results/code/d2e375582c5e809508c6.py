"""Matrix multiplication 2x3 x 3x6: search for lower-rank exact decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-8):
    """Check if U, V, W exactly decompose the 2x3 x 3x6 multiplication tensor."""
    n, m, p = DIMENSIONS
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # A rank-r decomposition means: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # Equivalently: for all (a, b, c): sum_{d} U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
    rank = U.shape[1]
    if rank == 0:
        return False
    # Build the tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute the decomposition
    # sum_d U[a,d] * V[b,d] * W[c,d]
    # This is a 3-mode contraction
    # Efficient: for each d, outer product of U[:,d], V[:,d], W[:,d]
    decomp = np.zeros_like(T)
    for d in range(rank):
        decomp += np.einsum('a,b,c->abc', U[:, d], V[:, d], W[:, d])
    return np.allclose(decomp, T, atol=tol)

def _to_half_integer_matrix(M):
    """Try to convert a floating-point matrix to half-integer entries.
    Returns (success, converted_matrix) where entries are multiples of 0.5."""
    # Scale by 2, round to nearest integer, divide by 2
    M2 = np.round(M * 2.0)
    M2 = np.round(M2)  # ensure integer
    if not np.allclose(M * 2.0, M2, atol=1e-6):
        return False, M
    # Verify exactness with the half-integer version
    return True, M2 / 2.0

def _snap_to_half_int(M):
    """Snap matrix to nearest half-integer values."""
    return np.round(M * 2.0) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s budget, but evaluator kills at 150s
    safety_margin = 5.0  # return by 115s to be safe
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 36
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition via optimization
    # Strategy: use random initialization + gradient-free local search (Nelder-Mead style)
    # on the constraint that entries are half-integers.
    # Actually, for exact decomposition, let's try a different approach:
    # Use the structure of the problem to construct a lower-rank solution.
    
    # Key insight: M_{2,3,6} can be viewed as two M_{2,3,3} problems in parallel
    # (splitting the 6 columns of B into two groups of 3).
    # The rank of M_{2,3,3} is at most 18 (schoolbook), but might be lower.
    # If we can find a rank-r decomposition for M_{2,3,3}, we get rank 2r for M_{2,3,6}.
    
    # Let's try to use a known result: the rank of M_{m,n,p} is at most m*n*p.
    # For small cases, there are known improvements.
    
    # Alternative approach: use bilinear algorithm search.
    # Represent the decomposition as U @ diag(1) @ V^T ⊗ ... 
    # Actually, let's use a direct numerical search.
    
    # Approach: parameterize a rank-r decomposition and use least-squares to find
    # U, V, W such that the tensor is exactly matched.
    # Since we need exact half-integer entries, we'll do:
    # 1. Find a real-valued rank-r decomposition (may exist even if not half-integer)
    # 2. Snap to half-integers and verify
    
    # For rank r, we have r * (nm + mp + pn) = r * (6 + 18 + 12) = 36r variables
    # and nm * mp * pn = 6 * 18 * 12 = 1296 constraints (but many are zero)
    # Number of nonzero tensor entries: n*m*p = 36
    # So we need to satisfy 36 equations with 36r variables.
    # For r = 36: 36*36 = 1296 vars, 36 eqs -> very underdetermined (schoolbook works)
    # For r < 36: still underdetermined
    
    # Let's use a randomized search: pick random rank-r U, V, W and solve for the
    # remaining factor, or use alternating minimization.
    
    # Better approach: use the fact that the tensor is sparse (only 36 nonzeros out of 1296)
    # to do a structured search.
    
    # Actually, the simplest improvement: use the block structure.
    # M_{2,3,6} = M_{2,3,3} ⊕ M_{2,3,3} (two independent 2x3 x 3x3 multiplications)
    # So rank(M_{2,3,6}) ≤ 2 * rank(M_{2,3,3})
    
    # Now, what is rank(M_{2,3,3})? Schoolbook gives 18.
    # Is there a rank-17 or lower decomposition?
    # The tensor M_{2,3,3} has 2*3*3 = 18 nonzero entries.
    # Schoolbook rank is 18.
    # By the submultiplicativity, rank(M_{2,3,6}) ≤ rank(M_{2,3,3}) * 2.
    
    # Let's try to find a rank-17 decomposition for M_{2,3,3} via search.
    # If found, we get rank 34 for M_{2,3,6}.
    
    # Search for rank-r decomposition of M_{2,3,3}:
    # Tensor T_{2,3,3}[i*3+j, j*3+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    def _search_rank_r(n_, m_, p_, r, time_limit=10.0):
        """Search for a rank-r half-integer decomposition of M_{n_,m_,p_}."""
        nm_, mp_, pn_ = n_ * m_, m_ * p_, p_ * n_
        # Build target tensor
        T = np.zeros((nm_, mp_, pn_))
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    T[i * m_ + j, j * p_ + k, k * n_ + i] = 1.0
        
        t_start = time.time()
        rng = np.random.RandomState(42)
        
        for attempt in range(200):
            if time.time() - t_start > time_limit:
                break
            
            # Random initialization
            U = rng.randn(nm_, r)
            V = rng.randn(mp_, r)
            W = rng.randn(pn_, r)
            
            # Alternating minimization: fix two, solve for the third
            # Given V and W, solve for U:
            # For each (a, b, c): sum_d U[a,d] V[b,d] W[c,d] = T[a,b,c]
            # This is linear in U for fixed V, W.
            # Rearrange: for fixed b, c: sum_d U[a,d] * (V[b,d] * W[c,d]) = T[a,b,c]
            # So: U[:, :] * diag-like = T slice
            # More precisely: define M_{bc}[a, d] = V[b,d] * W[c,d]
            # Then: sum_d U[a,d] * M_{bc}[a,d]... no that's not right.
            # Actually: sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
            # Let's define for each (b,c): the equation is sum_d U[a,d] * (V[b,d]*W[c,d]) = T[a,b,c]
            # This is: U[a,:] dot (V[b,:]*W[c,:]) = T[a,b,c] for all a,b,c
            # So U is a solution to a system: for each (b,c), U @ (V[b,:]*W[c,:])^T = T[:,b,c]
            # Stack all (b,c): this is a large linear system.
            
            # For fixed V, W: 
            # A linear system in U: 
            # For each (b,c): sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
            # This gives nm_ equations for each (b,c), so nm_ * mp_ * pn_ equations total
            # but many are zero (T is sparse).
            # Nonzero entries: n_*m_*p_ equations.
            # Unknowns: nm_ * r = n_*m_*r
            # For n_=2, m_=3, p_=3, r=17: unknowns = 6*17=102, eqs=18. Underdetermined.
            
            # Use least squares: find U that minimizes ||decomp - T||_F
            # decomp[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
            # Vectorize: for each d, the contribution is U[:,d] ⊗ V[:,d] ⊗ W[:,d]
            # Let's build a design matrix.
            
            # Flatten T: T_flat of size nm_*mp_*pn_
            # Design matrix X of size (nm_*mp_*pn_, nm_*r) where each column corresponds to (a, d)
            # X[flat_idx, a*n... ] hmm, this is getting complex.
            
            # Simpler: just use numpy lstsq with the full linear system.
            # Number of nonzero T entries: nnz = n_*m_*p_
            # For each nonzero (a,b,c) where T[a,b,c]=1:
            #   sum_d U[a,d] * V[b,d] * W[c,d] = 1
            # Unknowns: U[a,d] for a in range(nm_), d in range(r) -> nm_*r unknowns
            # Equations: nnz = n_*m_*p_
            
            if r >= n_ * m_ * p_:
                # Schoolbook-like, skip
                continue
            
            # Build linear system for U
            # Equations: for each (a,b,c) with T[a,b,c]!=0: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
            A = np.zeros((n_ * m_ * p_, nm_ * r))
            rhs = np.zeros(n_ * m_ * p_)
            idx = 0
            for i in range(n_):
                for j in range(m_):
                    for k in range(p_):
                        a = i * m_ + j
                        b = j * p_ + k
                        c = k * n_ + i
                        # Coefficients for U[a, d]
                        for d in range(r):
                            A[idx, a * r + d] = V[b, d] * W[c, d]
                        rhs[idx] = T[a, b, c]
                        idx += 1
            
            # Solve least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                U = U_new.reshape(nm_, r)
            except:
                continue
            
            # Now similarly solve for V and W
            # For V: sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
            # Unknowns: V[b,d], equations: for each (a,c)
            B_mat = np.zeros((n_ * m_ * p_, mp_ * r))
            rhs2 = np.zeros(n_ * m_ * p_)
            idx = 0
            for i in range(n_):
                for j in range(m_):
                    for k in range(p_):
                        a = i * m_ + j
                        b = j * p_ + k
                        c = k * n_ + i
                        for d in range(r):
                            B_mat[idx, b * r + d] = U[a, d] * W[c, d]
                        rhs2[idx] = T[a, b, c]
                        idx += 1
            try:
                V_new, _, _, _ = np.linalg.lstsq(B_mat, rhs2, rcond=None)
                V = V_new.reshape(mp_, r)
            except:
                continue
            
            # For W: sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
            C_mat = np.zeros((n_ * m_ * p_, pn_ * r))
            rhs3 = np.zeros(n_ * m_ * p_)
            idx = 0
            for i in range(n_):
                for j in range(m_):
                    for k in range(p_):
                        a = i * m_ + j
                        b = j * p_ + k
                        c = k * n_ + i
                        for d in range(r):
                            C_mat[idx, c * r + d] = U[a, d] * V[b, d]
                        rhs3[idx] = T[a, b, c]
                        idx += 1
            try:
                W_new, _, _, _ = np.linalg.lstsq(C_mat, rhs3, rcond=None)
                W = W_new.reshape(pn_, r)
            except:
                continue
            
            # Check if this gives an exact (or near-exact) solution
            decomp = np.zeros_like(T)
            for d in range(r):
                decomp += np.einsum('a,b,c->abc', U[:, d], V[:, d], W[:, d])
            err = np.abs(decomp - T).max()
            if err < 1e-6:
                # Snap to half-integers and re-verify
                U_hi = _snap_to_half_int(U)
                V_hi = _snap_to_half_int(V)
                W_hi = _snap_to_half_int(W)
                # Rebuild and check
                decomp2 = np.zeros_like(T)
                for d in range(r):
                    decomp2 += np.einsum('a,b,c->abc', U_hi[:, d], V_hi[:, d], W_hi[:, d])
                if np.allclose(decomp2, T, atol=1e-8):
                    return U_hi, V_hi, W_hi
        
        return None
    
    # Try to find rank-17 decomposition for M_{2,3,3}
    if time.time() < deadline - safety_margin:
        result_233 = _search_rank_r(2, 3, 3, 17, time_limit=min(30.0, deadline - time.time() - safety_margin))
        if result_233 is not None:
            U_233, V_233, W_233 = result_233
            # Build M_{2,3,6} from two copies of M_{2,3,3}
            # M_{2,3,6}: A is 2x3, B is 3x6 = [B1, B2] where B1, B2 are 3x3
            # C = A * B = [A*B1, A*B2]
            # U for M_{2,3,6}: same U as M_{2,3,3} for both halves -> U is 6x34
            # V for M_{2,3,6}: V1 (for B1) in rows 0-8 (j*p+k for p=3... wait)
            
            # For M_{2,3,6}: n=2, m=3, p=6
            # U: shape (6, r) where r = 2*17 = 34
            # V: shape (18, 34)
            # W: shape (12, 34)
            
            U_236 = np.zeros((6, 34))
            V_236 = np.zeros((18, 34))
            W_236 = np.zeros((12, 34))
            
            # First half (columns 0-2 of B, i.e., B1):
            # U rows: i*3+j for i in {0,1}, j in {0,1,2} -> same as M_{2,3,3}
            U_236[:, 0:17] = U_233
            # V rows for first half: j*6+k for k in {0,1,2}, j in {0,1,2}
            # In M_{2,3,3}: V rows are j*3+k
            # In M_{2,3,6}: V rows are j*6+k
            for j in range(3):
                for k in range(3):
                    V_236[j * 6 + k, 0:17] = V_233[j * 3 + k, :]
            # W rows for first half: k*2+i for k in {0,1,2}, i in {0,1}
            # In M_{2,3,3}: W rows are k*2+i
            # In M_{2,3,6}: W rows are k*2+i (same! since p=6 but k only goes to 2 here)
            # Wait, W has shape (p*n, r) = (12, 34). W[k*n+i] where k is column of B, i is row of A.
            # For the first half, k in {0,1,2}, so W rows 0-5 correspond to k=0,1,2
            for k in range(3):
                for i in range(2):
                    W_236[k * 2 + i, 0:17] = W_233[k * 2 + i, :]
            
            # Second half (columns 3-5 of B, i.e., B2):
            U_236[:, 17:34] = U_233
            for j in range(3):
                for k in range(3):
                    V_236[j * 6 + (k + 3), 17:34] = V_233[j * 3 + k, :]
            for k in range(3):
                for i in range(2):
                    W_236[(k + 3) * 2 + i, 17:34] = W_233[k * 2 + i, :]
            
            # Verify the full M_{2,3,6} decomposition
            if _verify(U_236, V_236, W_236):
                best_U, best_V, best_W = U_236, V_236, W_236
                best_rank = 34
    
    # Try rank-16 for M_{2,3,3} -> rank 32 for M_{2,3,6}
    if time.time() < deadline - safety_margin and best_rank > 32:
        result_233_16 = _search_rank_r(2, 3, 3, 16, time_limit=min(20.0, deadline - time.time() - safety_margin))
        if result_233_16 is not None:
            U_233, V_233, W_233 = result_233_16
            U_236 = np.zeros((6, 32))
            V_236 = np.zeros((18, 32))
            W_236 = np.zeros((12, 32))
            U_236[:, 0:16] = U_233
            U_236[:, 16:32] = U_233
            for j in range(3):
                for k in range(3):
                    V_236[j * 6 + k, 0:16] = V_233[j * 3 + k, :]
                    V_236[j * 6 + (k + 3), 16:32] = V_233[j * 3 + k, :]
            for k in range(3):
                for i in range(2):
                    W_236[k * 2 + i, 0:16] = W_233[k * 2 + i, :]
                    W_236[(k + 3) * 2 + i, 16:32] = W_233[k * 2 + i, :]
            if _verify(U_236, V_236, W_236):
                best_U, best_V, best_W = U_236, V_236, W_236
                best_rank = 32
    
    # Try direct search for rank < 36 on M_{2,3,6}
    # This is harder but let's try a few ranks
    if time.time() < deadline - safety_margin and best_rank > 35:
        # Try rank 35 directly
        U = rng.randn(6, 35) if 'rng' in dir() else np.random.RandomState(42).randn(6, 35)
        V = np.random.RandomState(43).randn(18, 35)
        W = np.random.RandomState(44).randn(12, 35)
        
        T = np.zeros((6, 18, 12))
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    T[i * 3 + j, j * 6 + k, k * 2 + i] = 1.0
        
        # One round of alternating minimization
        for _ in range(3):
            # Solve for U
            A = np.zeros((36, 6 * 35))
            rhs = np.zeros(36)
            idx = 0
            for i in range(2):
                for j in range(3):
                    for k in range(6):
                        a = i * 3 + j
                        b = j * 6 + k
                        c = k * 2 + i
                        for d in range(35):
                            A[idx, a * 35 + d] = V[b, d] * W[c, d]
                        rhs[idx] = T[a, b, c]
                        idx += 1
            U, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            U = U.reshape(6, 35)
            
            # Solve for V
            B_mat = np.zeros((36, 18 * 35))
            rhs2 = np.zeros(36)
            idx = 0
            for i in range(2):
                for j in range(3):
                    for k in range(6):
                        a = i * 3 + j
                        b = j * 6 + k
                        c = k * 2 + i
                        for d in range(35):
                            B_mat[idx, b * 35 + d] = U[a, d] * W[c, d]
                        rhs2[idx] = T[a, b, c]
                        idx += 1
            V, _, _, _ = np.linalg.lstsq(B_mat, rhs2, rcond=None)
            V = V.reshape(18, 35)
            
            # Solve for W
            C_mat = np.zeros((36, 12 * 35))
            rhs3 = np.zeros(36)
            idx = 0
            for i in range(2):
                for j in range(3):
                    for k in range(6):
                        a = i * 3 + j
                        b = j * 6 + k
                        c = k * 2 + i
                        for d in range(35):
                            C_mat[idx, c * 35 + d] = U[a, d] * V[b, d]
                        rhs3[idx] = T[a, b, c]
                        idx += 1
            W, _, _, _ = np.linalg.lstsq(C_mat, rhs3, rcond=None)
            W = W.reshape(12, 35)
        
        # Check and snap
        decomp = np.zeros_like(T)
        for d in range(35):
            decomp += np.einsum('a,b,c->abc', U[:, d], V[:, d], W[:, d])
        if np.abs(decomp - T).max() < 1e-6:
            U_hi = _snap_to_half_int(U)
            V_hi = _snap_to_half_int(V)
            W_hi = _snap_to_half_int(W)
            decomp2 = np.zeros_like(T)
            for d in range(35):
                decomp2 += np.einsum('a,b,c->abc', U_hi[:, d], V_hi[:, d], W_hi[:, d])
            if np.allclose(decomp2, T, atol=1e-8):
                if _verify(U_hi, V_hi, W_hi):
                    best_U, best_V, best_W = U_hi, V_hi, W_hi
                    best_rank = 35
    
    # Return the best result
    # Convert to lists as expected by the interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
