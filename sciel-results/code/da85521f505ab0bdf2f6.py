import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 140  # seconds, with 10s safety margin

_START_TIME = time.time()

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _construct_rank55():
    """Construct the known rank-55 decomposition."""
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1
        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1
        
        # Strassen M1-M7
        U[a00, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        U[a10, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0; W[c11, col] = -1.0
        col += 1
        
        U[a00, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        U[a11, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        U[a00, col] = 1.0; U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0; W[c01, col] = 1.0
        col += 1
        
        U[a10, col] = 1.0; U[a00, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # Schoolbook P1-P4
        U[a02, col] = 1.0; V[b20, col] = 1.0; W[c00, col] = 1.0
        col += 1
        U[a02, col] = 1.0; V[b21, col] = 1.0; W[c01, col] = 1.0
        col += 1
        U[a12, col] = 1.0; V[b20, col] = 1.0; W[c10, col] = 1.0
        col += 1
        U[a12, col] = 1.0; V[b21, col] = 1.0; W[c11, col] = 1.0
        col += 1
    
    return {'U': U, 'V': V, 'W': W}

def _verify_decomposition(U, V, W, n=2, m=3, p=10, tol=1e-10):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = U.shape[1]
    # Check: for all i,k: sum_r U[i,r]*V[r,k] should give W[k*n+i, r] contributions
    # The tensor T[i,j,k] = delta(i,j) for matmul, where i indexes A, j indexes B, k indexes C
    # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Sample check with random matrices
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        C_computed += U[:, r] @ np.diag(V[:, r]) @ W[:, r]  # This is wrong
    
    # Correct check: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (U[i,r]*V[j,r]*W[k,r] / (A[i,j]*B[j,k]))
    # Better: verify tensor equality
    # T[i,j,k] = 1 if this is a valid matmul term, else 0
    # For matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = delta_{j is the connecting index}
    # Actually T[i,j,k] = 1 for all valid (i,j,k) where j connects i and k
    
    # The matmul tensor: T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # and the contraction is sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
    
    # Decomposition: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # So we need: for all i,j,k: sum_r U[i,r] V[j,r] W[k,r] = 1
    
    # Wait, that's not right either. Let me re-read the contract.
    # "exactly decompose the matrix multiplication tensor"
    # The matmul tensor T has T[i,j,k] = 1 for all valid indices
    # and C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k
    
    # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # So we need sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    # That can't be right because the LHS is a product of three separate sums...
    # No, it's a single sum over r of products.
    
    # Let me verify with the rank-55 construction
    # For the P1 term: U[a02, col]=1, V[b20, col]=1, W[c00, col]=1
    # This contributes 1*1*1=1 to T[a02, b20, c00]
    # a02=2 (A[0,2]), b20=20 (B[2,0]), c00=0 (C[0,0])
    # T[2,20,0] should be 1, and indeed A[0,2]*B[2,0] contributes to C[0,0] ✓
    
    # So the check is: for all i in [0,6), j in [0,30), k in [0,20):
    # sum_r U[i,r] * V[j,r] * W[k,r] = 1 if the term A[i] * B[j] contributes to C[k]
    # A[i] where i = a_row * 3 + a_col, contributes to C[c_row, c_col] 
    #   where c_row = a_row, and the B index j = b_row * 10 + b_col
    #   contributes to C[a_row, b_col]
    # So T[i,j,k] = 1 iff a_row == c_row AND b_row == a_col AND c_col == b_col
    # i.e., i = a_row*3+a_col, j = a_col*10+b_col, k = b_col*2+a_row
    
    # Let's build the target tensor and check
    T_target = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_col in range(p):
                i = a_row * m + a_col
                j = a_col * p + b_col
                k = b_col * n + a_row
                T_target[i, j, k] = 1.0
    
    # Check decomposition
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # This is a 3-tensor, check all entries
    T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    return np.allclose(T_decomp, T_target, atol=tol)

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Tries to find rank-50 (10 per group), falls back to rank-55.
    """
    n, m, p = DIMENSIONS
    baseline = _construct_rank55()
    
    if _time_remaining() < 15:
        return {'U': baseline['U'].tolist(), 'V': baseline['V'].tolist(), 'W': baseline['W'].tolist()}
    
    # Sub-problem: 2x3 by 3x2 matmul
    # A: 2x3 (indices 0-5), B: 3x2 (indices 0-5), C: 2x2 (indices 0-3)
    # T[i,j,k] = 1 iff a_row==c_row and a_col==b_row
    # where i=a_row*3+a_col, j=b_row*2+b_col, k=c_row*2+c_col
    
    target_sub = np.zeros((6, 6, 4))
    for a_row in range(2):
        for a_col in range(3):
            for b_col in range(2):
                i = a_row * 3 + a_col
                j = a_col * 2 + b_col
                k = a_row * 2 + b_col
                target_sub[i, j, k] = 1.0
    
    def _try_rank10():
        """Try to find rank-10 decomposition for sub-problem using structured init."""
        # Strategy: use the known rank-11 solution and try to eliminate one term
        # Or use a better optimization with multiple restarts
        
        rng = np.random.RandomState(42)
        best_obj = np.inf
        best_params = None
        
        def _obj(params):
            u_sz = 60
            v_sz = 60
            w_sz = 40
            U = params[:u_sz].reshape(6, 10)
            V = params[u_sz:u_sz+v_sz].reshape(6, 10)
            W = params[u_sz+v_sz:].reshape(4, 10)
            T = np.einsum('it,jt,kt->ijk', U, V, W)
            return np.sum((T - target_sub) ** 2)
        
        def _obj_grad(params):
            u_sz = 60
            v_sz = 60
            w_sz = 40
            U = params[:u_sz].reshape(6, 10)
            V = params[u_sz:u_sz+v_sz].reshape(6, 10)
            W = params[u_sz+v_sz:].reshape(4, 10)
            T = np.einsum('it,jt,kt->ijk', U, V, W)
            diff = T - target_sub
            # Gradient w.r.t. U[i,t]: sum_{j,k} diff[i,j,k]*V[j,t]*W[k,t]
            gU = np.einsum('ijk,jt,kt->it', diff, V, W)
            gV = np.einsum('ijk,it,kt->jt', diff, U, W)
            gW = np.einsum('ijk,it,jt->kt', diff, U, V)
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        # Multiple restarts with different strategies
        for attempt in range(30):
            if _time_remaining() < 20:
                break
            
            if attempt < 10:
                # Small random init
                scale = 0.5
            elif attempt < 20:
                # Medium random init
                scale = 1.0
            else:
                # Start from perturbed rank-11 solution
                # Take rank-11 solution, set last column to small values, optimize
                scale = 0.8
            
            U_init = rng.randn(6, 10) * scale
            V_init = rng.randn(6, 10) * scale
            W_init = rng.randn(4, 10) * scale
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = minimize(_obj, params_init, jac=_obj_grad, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        
        if best_params is None:
            return None
        
        U = best_params[:60].reshape(6, 10)
        V = best_params[60:120].reshape(6, 10)
        W = best_params[120:].reshape(4, 10)
        T = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T - target_sub))
        
        if error < 1e-10:
            return (U, V, W)
        return None
    
    # Try rank 10
    if _time_remaining() > 25:
        sub_decomp = _try_rank10()
        if sub_decomp is not None:
            U_sub, V_sub, W_sub = sub_decomp
            rank_total = 50
            U_full = np.zeros((6, rank_total))
            V_full = np.zeros((30, rank_total))
            W_full = np.zeros((20, rank_total))
            
            for b in range(5):
                k0 = 2 * b
                k1 = 2 * b + 1
                col_offset = b * 10
                b_index_map = [k0, k1, 10+k0, 10+k1, 20+k0, 20+k1]
                c_index_map = [k0*2+0, k1*2+0, k0*2+1, k1*2+1]
                
                for t in range(10):
                    col = col_offset + t
                    U_full[:, col] = U_sub[:, t]
                    for j_sub in range(6):
                        V_full[b_index_map[j_sub], col] = V_sub[j_sub, t]
                    for k_sub in range(4):
                        W_full[c_index_map[k_sub], col] = W_sub[k_sub, t]
            
            if _verify_decomposition(U_full, V_full, W_full):
                return {'U': U_full.tolist(), 'V': V_full.tolist(), 'W': W_full.tolist()}
    
    # Fall back to rank 55
    return {'U': baseline['U'].tolist(), 'V': baseline['V'].tolist(), 'W': baseline['W'].tolist()}
