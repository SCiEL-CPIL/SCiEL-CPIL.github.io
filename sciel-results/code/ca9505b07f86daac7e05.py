"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 140.0  # Leave 10 seconds margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
        
        U: shape (4, 7), indexed by (i, j) -> row i*2+j
        V: shape (4, 7), indexed by (j, k) -> row j*2+k
        W: shape (4, 7), indexed by (k, i) -> row k*2+i  (trace convention)
        """
        # Strassen's 7 products for 2x2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # U entries: coefficient of A[i,j] in M_r
        # V entries: coefficient of B[j,k] in M_r
        # W entries: coefficient of C[i,k] in M_r (with trace convention W[k*2+i, r])
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00+A11)*(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00 (k=0, i=0)
        W[3, 0] = 1  # C11 (k=1, i=1)
        
        # M2: (A10+A11)*B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = -1 # C10 (k=0, i=1)
        W[3, 1] = 1  # C11 (k=1, i=1)
        
        # M3: A00*(B01-B11)
        U[0, 2] = 1  # A00
        V[1, 2] = 1  # B01
        V[3, 2] = -1 # B11
        W[1, 2] = 1  # C01 (k=1, i=0)
        W[3, 2] = 1  # C11 (k=1, i=1)
        
        # M4: A11*(B10-B00)
        U[3, 3] = 1  # A11
        V[2, 3] = 1  # B10
        V[0, 3] = -1 # B00
        W[0, 3] = 1  # C00 (k=0, i=0)
        W[2, 3] = 1  # C10 (k=0, i=1)
        
        # M5: (A00+A01)*B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1 # C00 (k=0, i=0)
        W[1, 4] = 1  # C01 (k=1, i=0)
        
        # M6: (A10-A00)*(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        W[3, 5] = 1  # C11 (k=1, i=1)
        
        # M7: (A01-A11)*(B10+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1 # A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00 (k=0, i=0)
        W[1, 6] = -1 # C01 (k=1, i=0)
        
        return U, V, W
    
    def strassen_2x1x1():
        """Return (U, V, W) for 2x1x1 matrix multiplication (scalar mult).
        
        This is just 2 scalar multiplications, rank 2.
        U: shape (2, 2), V: shape (1, 2), W: shape (2, 2)
        """
        U = np.zeros((2, 2))
        V = np.zeros((1, 2))
        W = np.zeros((2, 2))
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        U[1, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        return U, V, W
    
    def strassen_1x2x1():
        """Return (U, V, W) for 1x2x1 matrix multiplication.
        
        U: shape (2, 2), V: shape (2, 2), W: shape (2, 2)
        """
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        U[1, 1] = 1
        V[1, 1] = 1
        W[1, 1] = 1
        return U, V, W
    
    def block_decompose(n_blocks, m_blocks, p_blocks):
        """Decompose matrix multiplication using block structure.
        
        n_blocks: list of block sizes along n dimension (must sum to n)
        m_blocks: list of block sizes along m dimension (must sum to m)
        p_blocks: list of block sizes along p dimension (must sum to p)
        
        For each triple (nb, mb, pb), use:
        - Strassen 2x2x2 if all are 2
        - Schoolbook otherwise
        """
        if sum(n_blocks) != n or sum(m_blocks) != m or sum(p_blocks) != p:
            return None
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        for nb in n_blocks:
            for mb in m_blocks:
                for pb in p_blocks:
                    if nb == 2 and mb == 2 and pb == 2:
                        u, v, w = strassen_2x2x2()
                    elif nb == 2 and mb == 1 and pb == 1:
                        u, v, w = strassen_2x1x1()
                    elif nb == 1 and mb == 2 and pb == 1:
                        u, v, w = strassen_1x2x1()
                    else:
                        # Schoolbook for this block
                        rank = nb * mb * pb
                        u = np.zeros((nb * mb, rank))
                        v = np.zeros((mb * pb, rank))
                        w = np.zeros((pb * nb, rank))
                        col = 0
                        for i in range(nb):
                            for j in range(mb):
                                for k in range(pb):
                                    u[i * mb + j, col] = 1
                                    v[j * pb + k, col] = 1
                                    w[k * nb + i, col] = 1
                                    col += 1
                    U_parts.append(u)
                    V_parts.append(v)
                    W_parts.append(w)
        
        # Combine parts
        total_rank = sum(u.shape[1] for u in U_parts)
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        col = 0
        for u, v, w in zip(U_parts, V_parts, W_parts):
            r = u.shape[1]
            # Determine offsets
            # U rows: i*m + j, where i is within n_blocks, j within m_blocks
            # V rows: j*p + k, where j is within m_blocks, k within p_blocks
            # W rows: k*n + i, where k is within p_blocks, i within n_blocks
            
            # We need to figure out the global offsets
            # This is complex; let's use a different approach
            pass
        
        return None  # Placeholder - we'll use a simpler assembly
    
    def assemble_block_decomposition(n_blocks, m_blocks, p_blocks):
        """Assemble a block decomposition into full U, V, W matrices."""
        if sum(n_blocks) != n or sum(m_blocks) != m or sum(p_blocks) != p:
            return None
        
        U = np.zeros((nm, 0))
        V = np.zeros((mp, 0))
        W = np.zeros((pn, 0))
        
        n_offset = 0
        for nb in n_blocks:
            m_offset = 0
            for mb in m_blocks:
                p_offset = 0
                for pb in p_blocks:
                    if nb == 2 and mb == 2 and pb == 2:
                        u, v, w = strassen_2x2x2()
                    elif nb == 2 and mb == 1 and pb == 1:
                        u, v, w = strassen_2x1x1()
                    elif nb == 1 and mb == 2 and pb == 1:
                        u, v, w = strassen_1x2x1()
                    else:
                        rank = nb * mb * pb
                        u = np.zeros((nb * mb, rank))
                        v = np.zeros((mb * pb, rank))
                        w = np.zeros((pb * nb, rank))
                        col = 0
                        for i in range(nb):
                            for j in range(mb):
                                for k in range(pb):
                                    u[i * mb + j, col] = 1
                                    v[j * pb + k, col] = 1
                                    w[k * nb + i, col] = 1
                                    col += 1
                    
                    r = u.shape[1]
                    U_new = np.zeros((nm, r))
                    V_new = np.zeros((mp, r))
                    W_new = np.zeros((pn, r))
                    
                    # Place U: rows i*m + j where i in [n_offset, n_offset+nb), j in [m_offset, m_offset+mb)
                    for i in range(nb):
                        for j in range(mb):
                            U_new[(n_offset + i) * m + (m_offset + j), :] = u[i * mb + j, :]
                    
                    # Place V: rows j*p + k where j in [m_offset, m_offset+mb), k in [p_offset, p_offset+pb)
                    for j in range(mb):
                        for k in range(pb):
                            V_new[(m_offset + j) * p + (p_offset + k), :] = v[j * pb + k, :]
                    
                    # Place W: rows k*n + i where k in [p_offset, p_offset+pb), i in [n_offset, n_offset+nb)
                    for k in range(pb):
                        for i in range(nb):
                            W_new[(p_offset + k) * n + (n_offset + i), :] = w[k * nb + i, :]
                    
                    U = np.hstack([U, U_new])
                    V = np.hstack([V, V_new])
                    W = np.hstack([W, W_new])
                    
                    p_offset += pb
                m_offset += mb
            n_offset += nb
        
        return {"U": U, "V": V, "W": W}
    
    def compute_error(U, V, W):
        """Compute the sum of squared errors between the reconstructed tensor and the target."""
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        # Target: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = T_rec[i * m + j, j * p + k, k * n + i]
                    error += (actual - expected) ** 2
        # Also check that off-diagonal entries are 0
        for u_idx in range(nm):
            for v_idx in range(mp):
                for w_idx in range(pn):
                    i = u_idx // m
                    j = u_idx % m
                    j2 = v_idx // p
                    k = v_idx % p
                    k2 = w_idx // n
                    i2 = w_idx % n
                    if j != j2 or k != k2 or i != i2:
                        error += T_rec[u_idx, v_idx, w_idx] ** 2
        return error
    
    # Build target tensor for verification
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        max_err = np.max(np.abs(T_rec - T_target))
        return max_err < tol
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try various block decompositions
    block_configs = [
        # (n_blocks, m_blocks, p_blocks)
        ([1, 2], [2, 3], [2, 3]),
        ([1, 2], [2, 1, 2], [2, 1, 2]),
        ([1, 2], [1, 2, 2], [1, 2, 2]),
        ([2, 1], [2, 3], [2, 3]),
        ([2, 1], [1, 2, 2], [1, 2, 2]),
        ([1, 1, 1], [2, 3], [2, 3]),
        ([1, 1, 1], [1, 2, 2], [1, 2, 2]),
        ([3], [2, 3], [2, 3]),
        ([3], [1, 2, 2], [1, 2, 2]),
        ([1, 2], [5], [5]),
        ([3], [5], [5]),
    ]
    
    for config in block_configs:
        if time.time() - start_time > time_limit - 30:
            break
        result = assemble_block_decomposition(*config)
        if result is not None:
            U, V, W = result['U'], result['V'], result['W']
            if verify_decomposition(U, V, W):
                rank = U.shape[1]
                if rank < best_rank:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    if best_rank == max(nm, mp, pn):
                        break
    
    # Now try optimization to further reduce rank from the best block decomposition
    if time.time() - start_time < time_limit - 20 and best_rank > max(nm, mp, pn):
        def try_optimize(rank, max_iters=500, ftol=1e-10, gtol=1e-6):
            """Try to find a rank-`rank` decomposition using optimization."""
            if time.time() - start_time > time_limit - 10:
                return None
            
            n_restarts = 3
            best_local_error = float('inf')
            best_local_params = None
            
            for restart in range(n_restarts):
                if time.time() - start_time > time_limit - 15:
                    break
                
                if restart == 0:
                    U_init = np.random.randn(nm, rank) * 0.1
                    V_init = np.random.randn(mp, rank) * 0.1
                    W_init = np.random.randn(pn, rank) * 0.1
                elif restart == 1:
                    U_init = np.random.randn(nm, rank) * 0.5
                    V_init = np.random.randn(mp, rank) * 0.5
                    W_init = np.random.randn(pn, rank) * 0.5
                else:
                    U_init = np.random.randn(nm, rank) * 1.0
                    V_init = np.random.randn(mp, rank) * 1.0
                    W_init = np.random.randn(pn, rank) * 1.0
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                def objective(params):
                    if time.time() - start_time > time_limit - 5:
                        return 1e10
                    U = params[:nm * rank].reshape(nm, rank)
                    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                    W = params[nm * rank + mp * rank:].reshape(pn, rank)
                    T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                    error = np.sum((T_rec - T_target) ** 2)
                    return error
                
                try:
                    result = minimize(objective, x0, method='L-BFGS-B', 
                                    options={'maxiter': max_iters, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
                    
                    if result.fun < best_local_error:
                        best_local_error = result.fun
                        best_local_params = result.x.copy()
                    
                    if result.fun < 1e-6:
                        U = result.x[:nm * rank].reshape(nm, rank)
                        V = result.x[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                        W = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                        
                        # Round to nearest half-integer
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if verify_decomposition(U_rounded, V_rounded, W_rounded):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                except Exception:
                    pass
            
            if best_local_params is not None and best_local_error < 1e-4:
                try:
                    U = best_local_params[:nm * rank].reshape(nm, rank)
                    V = best_local_params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                    W = best_local_params[nm * rank + mp * rank:].reshape(pn, rank)
                    
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                except Exception:
                    pass
            
            return None
        
        # Try ranks from lower bound up to current best
        lower_bound = max(nm, mp, pn)
        max_ranks_to_try = 10
        for rank in range(lower_bound, min(lower_bound + max_ranks_to_try, best_rank)):
            if time.time() - start_time > time_limit - 20:
                break
            
            result = try_optimize(rank)
            if result is not None:
                best_result = result
                best_rank = rank
                if best_rank == lower_bound:
                    break
    
    if time.time() - start_time > time_limit:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
