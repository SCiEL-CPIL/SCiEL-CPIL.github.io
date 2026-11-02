"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def get_schoolbook():
        """Return schoolbook decomposition as lists."""
        rank = n * m * p  # 50
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)
    
    def try_structured_rank(rank):
        """
        Try to find a structured decomposition using block decomposition.
        For 2x5x5, split the 5 dimension into 2+3 blocks and use Strassen-like tricks.
        """
        if time.time() > deadline:
            return None
        
        # Try a block decomposition: split j into j1 (size 2) and j2 (size 3)
        # A is 2x5 = [A1 | A2] where A1 is 2x2, A2 is 2x3
        # B is 5x5 = [B1; B2] where B1 is 2x5, B2 is 3x5
        # C = A1*B1 + A2*B2
        
        # For the 2x2x5 part (A1*B1): rank at most 2*2*5=20, but we can do better
        # For the 2x3x5 part (A2*B2): rank at most 2*3*5=30
        # Total: 50 (schoolbook)
        
        # Better: use Strassen on the 2x2 part
        # A1*B1 where A1 is 2x2, B1 is 2x5
        # Strassen: M1=(A11+A22)*(B11+B22), M2=(A21+A22)*B11, M3=A11*(B12-B22), 
        #           M4=A22*(B21-B11), M5=(A11+A12)*B22, M6=(A21-A11)*(B11+B12), 
        #           M7=(A12-A22)*(B21+B22)
        # But B1 is 2x5, not 2x2. So we need to adapt.
        
        # Alternative: just try random initialization with better structure
        for restart in range(30):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + restart * 7)
            
            # Initialize with a structured pattern
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            # Random initialization with small values
            U[:] = np.random.randn(dim_u, rank) * 0.3
            V[:] = np.random.randn(dim_v, rank) * 0.3
            W[:] = np.random.randn(dim_w, rank) * 0.3
            
            # Add some structure: make first few columns match schoolbook pattern
            for col in range(min(rank, 50)):
                i = col // (m * p)
                rem = col % (m * p)
                j = rem // p
                k = rem % p
                if i < n and j < m and k < p:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def fun(params):
                u_flat = params[:dim_u * rank]
                v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
                w_flat = params[dim_u * rank + dim_v * rank : ]
                
                U_m = u_flat.reshape(dim_u, rank)
                V_m = v_flat.reshape(dim_v, rank)
                W_m = w_flat.reshape(dim_w, rank)
                
                T = np.einsum('ar,br,cr->abc', U_m, V_m, W_m)
                diff = T - target
                loss = np.sum(diff * diff)
                
                grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V_m, W_m)
                grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U_m, W_m)
                grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U_m, V_m)
                
                grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
                return loss, grad
            
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
                
                if res.fun < 1e-8:
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    # Snap to nearest half-integer
                    u_snap = np.round(u_final * 2.0) / 2.0
                    v_snap = np.round(v_final * 2.0) / 2.0
                    w_snap = np.round(w_final * 2.0) / 2.0
                    
                    if verify_decomposition(u_snap, v_snap, w_snap):
                        return (u_snap, v_snap, w_snap)
                    
                    # Try cleaning small values
                    u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
                    v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
                    w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
                    if verify_decomposition(u_clean, v_clean, w_clean):
                        return (u_clean, v_clean, w_clean)
            except Exception:
                continue
        
        return None
    
    # Try a direct algebraic construction for rank 35
    # Split 5 = 2 + 3
    # C = A1*B1 + A2*B2 where A1 is 2x2, B1 is 2x5, A2 is 2x3, B2 is 3x5
    # Use rank-7 Strassen for 2x2x2 block and standard for remaining
    def try_algebraic_rank_35():
        """
        Try to construct a rank-35 decomposition algebraically.
        2x2x5 part: can use 7*5=35 terms? No, that's not right.
        Let's think: A1*B1 is 2x2 times 2x5. 
        Each of the 5 columns of B1 gives a 2x2 times 2x1 product.
        2x2x1 has rank at most 4, but Strassen gives rank 7 for 2x2x2.
        For 2x2x1: M1=A11*B11, M2=A21*B11, M3=A12*B12, M4=A22*B12
        That's rank 4 for each column, so 4*5=20 for A1*B1.
        A2*B2 is 2x3 times 3x5: rank at most 2*3*5=30.
        Total: 20+30=50. Not helpful.
        
        Let me try a different approach: use the fact that we only need 2 output rows.
        """
        if time.time() > deadline:
            return None
        
        # Try rank 40: split into 2x2x5 (rank 20) + 2x3x5 (rank 20)
        # 2x2x5: for each of 5 columns, 2x2x1 needs 4 mults, so 20 total
        # 2x3x5: for each of 5 columns, 2x3x1 needs 6 mults, so 30 total
        # That's 50 again.
        
        # Alternative: use shared computation
        # C[0,k] = sum_j A[0,j]*B[j,k]
        # C[1,k] = sum_j A[1,j]*B[j,k]
        
        # Group j=0,1 together: 
        # C[0,k] += A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] += A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # This is a 2x2x5 subproblem.
        
        # Use Strassen-like: define
        # M1 = (A[0,0]+A[1,1]) * B[0,k]  -- but B[0,k] varies with k
        # This doesn't work directly because B entries differ by k.
        
        return None
    
    # Try various ranks from a reasonable lower bound
    # Theoretical lower bound is 25, but let's start from 30
    best_result = None
    
    for target_rank in range(30, 50):
        if time.time() > deadline:
            break
        
        result = try_structured_rank(target_rank)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
