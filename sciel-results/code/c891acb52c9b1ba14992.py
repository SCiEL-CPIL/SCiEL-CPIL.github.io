"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition, rank = n*m*p."""
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

def _strassen_decomposition(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2x2 blocks where possible.
    This is the baseline 'tiling' strategy which achieves rank 88 for 3x4x8.
    """
    def splits_of(N):
        if N == 0:
            return [[]]
        result = []
        for first in range(1, N+1):
            for rest in splits_of(N - first):
                result.append([first] + rest)
        return result
    
    def block_rank(di, dj, dk):
        if di == 2 and dj == 2 and dk == 2:
            return 7
        return di * dj * dk
    
    best_rank = n * m * p
    best_split = None
    
    i_splits = splits_of(n)
    m_splits = splits_of(m)
    p_splits = splits_of(p)
    
    for iseg in i_splits:
        for mseg in m_splits:
            for pseg in p_splits:
                total_rank = 0
                for di in iseg:
                    for dj in mseg:
                        for dk in pseg:
                            total_rank += block_rank(di, dj, dk)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_split = (iseg, mseg, pseg)
    
    if best_split is None:
        return None
    
    iseg, mseg, pseg = best_split
    rank = best_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    i0 = 0
    for di in iseg:
        j0 = 0
        for dj in mseg:
            k0 = 0
            for dk in pseg:
                if di == 2 and dj == 2 and dk == 2:
                    strassen_terms = [
                        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
                        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
                        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
                        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
                        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
                        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
                        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
                    ]
                    for t in range(7):
                        a_c, b_c, c_c = strassen_terms[t]
                        for a_idx in range(4):
                            ai, aj = a_idx // 2, a_idx % 2
                            U[(i0+ai)*m + (j0+aj), col] = a_c[a_idx]
                        for b_idx in range(4):
                            bj, bk = b_idx // 2, b_idx % 2
                            V[(j0+bj)*p + (k0+bk), col] = b_c[b_idx]
                        for c_idx in range(4):
                            ck, ci = c_idx // 2, c_idx % 2
                            W[(k0+ck)*n + (i0+ci), col] = c_c[c_idx]
                        col += 1
                else:
                    for ai in range(di):
                        for aj in range(dj):
                            for ak in range(dk):
                                U[(i0+ai)*m + (j0+aj), col] = 1
                                V[(j0+aj)*p + (k0+ak), col] = 1
                                W[(k0+ak)*n + (i0+ai), col] = 1
                                col += 1
                k0 += dk
            j0 += dj
        i0 += di
    
    if col != rank:
        return None
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _tensor_matrix_mult(A, B, C):
    """
    Compute the matrix representation of the tensor contraction.
    T is represented as a 3D array T[i, j, k] where:
    i indexes U (n*m), j indexes V (m*p), k indexes W (p*n).
    The multiplication T * C is a sum over k: sum_k T[i,j,k] C[k,l].
    This is equivalent to matmul(T.reshape(i, j, k), C.reshape(k, l)).reshape(i, j, l)
    """
    return np.matmul(A.reshape(A.shape[0], A.shape[1], -1), C).reshape(A.shape[0], A.shape[1], C.shape[1])

def _als_optimize(T, rank, max_iter=1000, tol=1e-10):
    """
    Alternating Least Squares to minimize ||T - U*V*W||_F.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    dims = [n*m, m*p, p*n]
    
    # Initialize with random small values
    U = np.random.randn(dims[0], rank) * 0.1
    V = np.random.randn(dims[1], rank) * 0.1
    W = np.random.randn(dims[2], rank) * 0.1
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U: minimize ||T - U (V*W)||_F
        # V*W is (m*p, p*n). T is (n*m, m*p, p*n).
        # We need to solve for U such that U @ (V*W) approx T.
        # This is a linear system: U * M = T_flat where M = V*W and T_flat is T reshaped.
        # Actually, T is a 3rd order tensor. The contraction T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}.
        # Fix V, W. For each i, we have a matrix T_i (m*p, p*n) which should equal sum_c U_ic (V_c outer W_c).
        # Let M = V @ W.T ? No.
        # T_{i, j, k} = sum_c U_{i, c} V_{j, c} W_{k, c}.
        # Let M_{c, j, k} = V_{j, c} W_{k, c}. This is the outer product of V_c and W_c.
        # So T_{i, :, :} = sum_c U_{i, c} (V_c outer W_c).
        # This is a linear system for U_i.
        # The basis matrices are B_c = V_c outer W_c.
        # We want U_i such that sum_c U_{i,c} B_c approx T_i.
        # Vectorize: vec(T_i) = sum_c U_{i,c} vec(B_c).
        # Let B_mat be (m*p*p*n, rank) where column c is vec(B_c).
        # Then U_i^T B_mat = vec(T_i)^T.
        # B_mat = kron(W, V.T) ? Let's check.
        # vec(V_c outer W_c) = W_c kron V_c.
        # So B_mat columns are W_c kron V_c.
        # B_mat = kron(W, V.T) is (p*n * m*p, rank).
        # Wait, vec(A outer B) = B kron A.
        # Here A=V_c, B=W_c. So vec(V_c outer W_c) = W_c kron V_c.
        # B_mat = [W_1 kron V_1, ..., W_r kron V_r].
        # B_mat = kron(W, V.T) ?
        # (kron(W, V.T))_{(k,j), c} = W_{k,c} V.T_{j,c} = W_{k,c} V_{c,j}.
        # Yes, this matches.
        
        B_mat = np.kron(W, V.T) # Shape: (p*n * m*p, rank)
        # T_flat: (n*m, m*p * p*n).
        # We want to solve X B_mat^T = T_flat where X is U^T (rank, n*m).
        # U = (B_mat^T X^T)^T = (X B_mat)^T.
        # Least squares: U^T = T_flat @ pinv(B_mat^T) ? No.
        # Minimize ||T_flat - U^T B_mat^T||_F.
        # Solution: U^T = T_flat @ (B_mat @ B_mat.T)^-1 B_mat.T
        # Or simply use lstsq.
        # T_flat is (n*m, N). B_mat.T is (N, rank).
        # We want U^T (rank, n*m) such that U^T B_mat.T approx T_flat.
        # This is (U^T B_mat.T) = T_flat.
        # Transpose: B_mat U = T_flat.T.
        # U (rank, n*m). B_mat (N, rank). T_flat.T (N, n*m).
        # Solve B_mat U = T_flat.T.
        
        # To save time, we can solve for each row of U separately?
        # U is (rank, n*m).
        # B_mat is (N, rank).
        # We want U such that B_mat @ U = T_flat.T.
        # This is a linear system for U.
        # Since rank is small, we can use lstsq.
        
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat, T_flat.T, rcond=None)
        except:
            break
            
        U = U_new.T # Back to (n*m, rank)
        
        # Update V
        # T_{i, j, k} = sum_c U_{i, c} V_{j, c} W_{k, c}.
        # Fix U, W.
        # T_{:, j, :} = sum_c V_{j, c} (U_c outer W_c).
        # vec(T_{:, j, :}) = sum_c V_{j, c} vec(U_c outer W_c) = sum_c V_{j, c} (W_c kron U_c).
        # B_mat_V = kron(W, U.T). Shape (N, rank).
        # T_flat_V: We need T reshaped such that rows correspond to j.
        # T is (n*m, m*p, p*n).
        # We want a matrix where row j contains vec(T_{:, j, :}).
        # T.transpose(1, 0, 2).reshape(m*p, n*m * p*n).
        
        T_flat_V = T.transpose(1, 0, 2).reshape(dims[1], dims[0] * dims[2])
        B_mat_V = np.kron(W, U.T)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat_V, T_flat_V.T, rcond=None)
        except:
            break
        V = V_new.T
        
        # Update W
        # T_{i, j, k} = sum_c U_{i, c} V_{j, c} W_{k, c}.
        # Fix U, V.
        # T_{:, :, k} = sum_c W_{k, c} (U_c outer V_c).
        # vec(T_{:, :, k}) = sum_c W_{k, c} vec(U_c outer V_c) = sum_c W_{k, c} (V_c kron U_c).
        # B_mat_W = kron(V, U.T). Shape (N, rank).
        # T_flat_W: row k contains vec(T_{:, :, k}).
        # T.transpose(2, 0, 1).reshape(p*n, n*m * m*p).
        
        T_flat_W = T.transpose(2, 0, 1).reshape(dims[2], dims[0] * dims[1])
        B_mat_W = np.kron(V, U.T)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat_W, T_flat_W.T, rcond=None)
        except:
            break
        W = W_new.T
        
        # Check convergence
        # Compute error
        # T_approx = sum_c U_c outer V_c outer W_c
        # Efficient check:
        # err = ||T - T_approx||_F
        # T_approx_{i,j,k} = sum_c U_ic V_jc W_kc
        # We can compute T_approx by contracting.
        # T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        # This might be slow for large rank.
        # Alternative: compute norm squared.
        # ||T_approx||^2 = trace(T_approx^T T_approx)
        # = sum_{i,j,k} (sum_c U_ic V_jc W_kc)^2
        # = sum_c,c' (sum_i U_ic U_ic') (sum_j V_jc V_jc') (sum_k W_kc W_kc')
        # = sum_c,c' (U^T U)_{cc'} (V^T V)_{cc'} (W^T W)_{cc'}
        # = trace( (U^T U) * (V^T V) * (W^T W) ) ? No, element-wise product sum.
        # = sum_c (U^T U)_{cc} (V^T V)_{cc} (W^T W)_{cc} ? No.
        # It is sum_c,c' A_cc' B_cc' C_cc'.
        # This is trace(A B C) if A,B,C are symmetric?
        # Let's just compute the residual norm via matmul trick.
        # R = T - T_approx.
        # ||R||^2 = ||T||^2 - 2 <T, T_approx> + ||T_approx||^2.
        
        # <T, T_approx> = sum_{i,j,k} T_{ijk} sum_c U_ic V_jc W_kc
        # = sum_c (sum_i T_{i,:,c} U_ic) ... no.
        # = sum_c (sum_{i,j,k} T_{ijk} U_ic V_jc W_kc)
        # = sum_c ( (T * U_c) * V_c ) * W_c ?
        # Let M_c = T * U_c (contract over i). M_c is (m*p, p*n).
        # M_c[j,k] = sum_i T_{i,j,k} U_{i,c}.
        # Then <T, T_approx> = sum_c sum_{j,k} M_c[j,k] V_{j,c} W_{k,c}.
        
        # ||T_approx||^2 = sum_c (U_c^T U_c) (V_c^T V_c) (W_c^T W_c) ? No.
        # = sum_c,c' (U^T U)_{cc'} (V^T V)_{cc'} (W^T W)_{cc'}.
        
        # This is getting complicated. Let's just compute T_approx for small rank.
        # Rank is likely < 100.
        # T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        # This is fast enough for rank < 100 and dims 12, 32, 24.
        
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T - T_approx)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W, err

def _try_als_decomposition(n, m, p, deadline):
    """
    Try to find a low rank decomposition using ALS.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_rank = n*m*p
    
    # Try ranks from a heuristic lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # Also, rank >= 2 * max(n,m,p) / min(n,m,p) ? No.
    # Just try a range.
    for rank in range(32, 97):
        if time.time() > deadline - 10:
            break
            
        # Run ALS with a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            U, V, W, err = _als_optimize(T, rank, max_iter=200, tol=1e-12)
            
            if err < 1e-8:
                # Found a numerical solution. Now try to recover exact integers.
                # The entries should be integers or half-integers.
                # Scale by 2 to check for integers.
                U_int = np.round(U * 2)
                V_int = np.round(V * 2)
                W_int = np.round(W * 2)
                
                # Check if U_int/2, V_int/2, W_int/2 is a valid solution
                U_check = U_int / 2.0
                V_check = V_int / 2.0
                W_check = W_int / 2.0
                
                T_approx = np.einsum('ic,jc,kc->ijk', U_check, V_check, W_check)
                if np.linalg.norm(T - T_approx) < 1e-8:
                    # Valid exact solution!
                    best_result = {
                        "U": U_check.tolist(),
                        "V": V_check.tolist(),
                        "W": W_check.tolist()
                    }
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # We found a solution for this rank. Can we go lower?
            # Since we iterate upwards, the first found is the minimal rank 
            # among the tried ranks.
            # But maybe a lower rank was missed due to local minima.
            # Given time constraints, we return the first found.
            return best_result
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy 1: Tiling with Strassen (known to give 88)
    result = _strassen_decomposition(n, m, p)
    
    if result is not None:
        # We have a baseline of 88.
        # Try to improve with ALS.
        # If ALS finds something better, return it.
        better_result = _try_als_decomposition(n, m, p, deadline)
        if better_result is not None:
            # Verify the better result
            if _verify(better_result['U'], better_result['V'], better_result['W'], n, m, p):
                return better_result
        return result
    
    # Fallback
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
