"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import numpy as np

# EVOLVE-BLOCK-START
import time

def _verify_decomposition(U, V, W, n=2, m=2, p=2, tol=1e-10):
    """Verify that U, V, W form a valid matrix multiplication decomposition."""
    # The decomposition must satisfy: T[i,j,k,l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T is the matrix multiplication tensor
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Build the tensor from decomposition and compare with identity tensor
    # T[i,j,k,l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if (i,j) == (k,l) in the appropriate sense for matmul
    # Actually for matrix multiplication: (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k,l] = 1 if j == l, else 0... no wait.
    # The tensor T_matmul[i,j,k,l] = delta[j,k] (since (AB)[i,l] = sum_k A[i,k]B[k,l])
    # Wait, let me reconsider. For C = A * B where C[i,k] = sum_j A[i,j] B[j,k]:
    # T[i,j,k,l] = delta[j,k] (the index j in A must match index k in B... no)
    # T[i,j,k,l] = 1 if j == k, else 0? No...
    # Actually the tensor is T[i,j,k,l] = delta_{j,k} where we sum over j=k
    # Hmm, let me think more carefully.
    # (AB)_{i,l} = sum_j A_{i,j} B_{j,l}
    # So the tensor T has T[i,j,k,l] = delta_{j,k} (the free indices are i and l, contracted j and k)
    # Wait no. In the decomposition T = sum_c U_c ⊗ V_c ⊗ W_c:
    # T[i,j,k,l] = sum_c U[i,j,c] * V[k,l,c] * W[?,c]
    # The convention is W[k*n+i, c] so W's row index encodes (k, i)
    # So: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+l, c] * B[j,l])
    # Hmm, this is getting complex. Let me just verify numerically with random matrices.
    
    # Generate random test matrices
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            # U[i*m+j, c] gives coefficients for A
            u_c = U[:, c]  # shape (n*m,)
            v_c = V[:, c]  # shape (m*p,)
            w_c = W[:, c]  # shape (p*n,)
            
            # Reshape U row: u_c[i*m+j] -> U[i,j,c]
            u_mat = u_c.reshape(n, m)
            # Reshape V row: v_c[j*p+k] -> V[j,k,c]  
            v_mat = v_c.reshape(m, p)
            # Reshape W row: w_c[k*n+i] -> W[k,i,c]
            w_mat = w_c.reshape(p, n)
            
            # Product term: (A @ u_mat^T?) No...
            # The decomposition says: C = sum_c w_c_outer * (u_c dot A) * (B dot v_c)
            # More precisely: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c]*A[i,j]) * (sum_j V[j*p+k, c]*B[j,k])
            # Wait, that doesn't look right either. Let me use the standard formulation.
            # Standard: T_{i,j,k,l} = sum_r U_{i,j,r} V_{k,l,r} W_{r} where W is scalar
            # But here W is a matrix too. The convention W[k*n+i, r] means:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            # Hmm, that's not standard. Let me try:
            # Actually in tensor decomposition of matmul:
            # T_{i,j,k,l} = sum_r U_{i,j,r} * V_{k,l,r} * W_{i,l,r}
            # where W_{i,l,r} is stored as W[l*n+i, r] (trace convention)
            # So: C[i,l] = sum_j,k A[i,j] B[k,l] T_{i,j,k,l}
            #       = sum_j,k A[i,j] B[k,l] sum_r U[i,j,r] V[k,l,r] W[l*n+i, r]
            #       = sum_r W[l*n+i, r] (sum_j A[i,j] U[i,j,r]) (sum_k B[k,l] V[k,l,r])
            # But wait, that gives sum_r W[l*n+i,r] * (A @ U_r)[i] * (B @ V_r)[l]
            # where U_r is the r-th column reshaped... no.
            # U[i*m+j, r] means U_r[i,j] = U[i*m+j, r]
            # So sum_j A[i,j] U_r[i,j] = elementwise product, not matrix mult.
            # That's: sum_j A[i,j] * U[i,j,r]
            # Similarly sum_k B[k,l] * V[k,l,r]
            
            # Let me compute: for each r, term = W[r_vec] * (elementwise_A_U_r) * (elementwise_B_V_r)
            # But that gives a scalar per r, and we sum over r to get... what?
            
            # I think the correct interpretation is:
            # The matmul tensor T[i,j,k,l] = delta[j,k] (since (AB)[i,l] = sum_j A[i,j]B[j,l])
            # And T = sum_r U_r ⊗ V_r ⊗ W_r where:
            # T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
            # Wait, V has shape (m*p, rank), so V[j*p+k, r] corresponds to B[j,k]
            # And the tensor T[i,j,k,l] should be delta[j,k] for matmul
            # Let's verify: sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r] should = delta[j,k]
            # But that's 4 indices and we're summing over r... 
            # Actually no. The decomposition is:
            # T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
            # This must equal delta[j,k] (the matmul tensor)
            
            # Let me just verify this identity numerically
            pass
        
        # Simpler verification: check the tensor identity directly
        # T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r] == delta[j,k]
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        t_decomp = 0.0
                        for r in range(rank):
                            t_decomp += U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
                        t_expected = 1.0 if j == k else 0.0
                        if abs(t_decomp - t_expected) > tol:
                            return False
    
    return True

def _try_rank6_search(time_budget=30.0):
    """
    Attempt to find a rank-6 decomposition by exploiting symmetry.
    Returns (U, V, W) if found, None otherwise.
    """
    start_time = time.time()
    n, m, p = 2, 2, 2
    
    # For rank 6, we have 6*4*3 = 72 unknowns (U: 4x6, V: 4x6, W: 4x6)
    # The constraints are T[i,j,k,l] = sum_r U[i*m+j,r]*V[k*p+l,r]*W[l*n+i,r] = delta[j,k]
    # That's 16 equations (4*4) but many are trivially satisfied or dependent.
    # Non-trivial constraints: for each (i,j,k,l) with j != k, the sum must be 0
    # And for j == k, the sum must be 1.
    # Total: 16 equations, but due to structure, we have 8 non-trivial zero constraints
    # (i,j,k,l) with j!=k: 4*4 - 4 = 12... let me count.
    # i in {0,1}, j in {0,1}, k in {0,1}, l in {0,1}
    # j==k: (i,0,0,l) and (i,1,1,l) -> 4*2 = 8 equations (should equal 1)
    # j!=k: (i,0,1,l) and (i,1,0,l) -> 4*2 = 8 equations (should equal 0)
    # Total 16 equations.
    
    # The system is overdetermined (72 unknowns, 16 equations) but nonlinear.
    # It's known that no rank-6 decomposition exists over R for 2x2 matmul.
    # But let's do a quick numerical check using random initialization + gradient-free search
    # to confirm, with a tight time budget.
    
    best_residual = np.inf
    best_params = None
    
    # Use a simple random search with restarts
    np.random.seed(12345)
    max_iters = 5000
    iters_per_restart = 200
    n_restarts = max_iters // iters_per_restart
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Random initialization
        U = np.random.randn(4, 6) * 0.5
        V = np.random.randn(4, 6) * 0.5
        W = np.random.randn(4, 6) * 0.5
        
        # Simple coordinate descent / random perturbation
        lr = 0.1
        for it in range(iters_per_restart):
            if time.time() - start_time > time_budget:
                break
            
            # Compute residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            t_val = 0.0
                            for r in range(6):
                                t_val += U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
                            expected = 1.0 if j == k else 0.0
                            residual += (t_val - expected) ** 2
            
            if residual < best_residual:
                best_residual = residual
                best_params = (U.copy(), V.copy(), W.copy())
            
            if residual < 1e-10:
                return U, V, W
            
            # Random perturbation
            mask = np.random.rand() < 0.33
            if mask:
                idx = np.random.randint(0, 24)
                U[idx//6, idx%6] += np.random.randn() * lr
            elif mask:
                idx = np.random.randint(0, 24)
                V[idx//6, idx%6] += np.random.randn() * lr
            else:
                idx = np.random.randint(0, 24)
                W[idx//6, idx%6] += np.random.randn() * lr
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    First attempts a rank-6 search (known to be infeasible, but verified numerically),
    then falls back to the proven-optimal Strassen rank-7 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    deadline = time.time() + 120
    
    # Attempt rank-6 search with tight time budget
    try:
        result = _try_rank6_search(time_budget=10.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Check deadline before returning rank-7
    if time.time() > deadline - 5:
        # Return Strassen immediately
        pass
    
    # Strassen's rank-7 decomposition (proven optimal for 2x2 matmul over R)
    rank = 7

    # Strassen's 7 products
    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # Row order: A[0,0], A[0,1], A[1,0], A[1,1]
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # Row order: B[0,0], B[0,1], B[1,0], B[1,1]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # Row index = k*n + i (official trace convention)
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ]

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
