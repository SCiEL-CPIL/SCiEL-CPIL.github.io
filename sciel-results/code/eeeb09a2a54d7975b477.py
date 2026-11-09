import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    R = U.shape[1]
    U = U.reshape(n, m, R)
    V = V.reshape(m, p, R)
    W = W.reshape(p, n, R)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    T_target = np.ones((n, m, p))
    
    # Check exact match with tolerance for floating point
    if np.allclose(T_computed, T_target, atol=1e-10):
        return True
    return False

def _constructive_decomposition():
    """
    Constructive decomposition for 3x4 x 4x5 matrix multiplication.
    
    Strategy: Partition the middle dimension (4) into 2+2.
    A is 3x4 = [A1 | A2] where A1, A2 are 3x2
    B is 4x5 = [B1; B2] where B1, B2 are 2x5
    C = A1*B1 + A2*B2
    
    For each 3x2 * 2x5 product, we can use a decomposition.
    Actually, let's use a direct construction.
    
    The tensor T[i,j,k] = 1 for all i,j,k (in the appropriate ranges).
    
    We'll use the following approach: for each of the 4 values of j,
    we need to express the contribution. Let's use a rank-4 base for the 4-dimension.
    
    Actually, let me use a simpler constructive approach:
    We know that 3*4*5 = 60 is the schoolbook rank.
    We can potentially do better by sharing some factors.
    
    Let's try a rank-50 construction using the following idea:
    Split the 4 middle indices into pairs and use 7 multiplications per pair (Strassen-like),
    but adjusted for rectangular matrices.
    
    For 3x2 * 2x5: we can use the following 7-term decomposition:
    Let A = [[a11, a12], [a21, a22], [a31, a32]] (3x2)
    Let B = [[b11, b12, b13, b14, b15], [b21, b22, b23, b24, b25]] (2x5)
    
    C = A*B is 3x5.
    
    Using a Strassen-like approach for 3x2 * 2x5:
    M1 = (a11 + a21) * (b11 + b21)  -> contributes to c11, c21
    M2 = (a11 + a21) * b21          -> 
    ...
    
    This is getting complex. Let me use a different constructive approach.
    
    I'll use the fact that we can write the identity tensor as a sum of rank-1 tensors.
    For the (3,4,5) case, I'll use a systematic construction.
    
    Key insight: The matrix multiplication tensor for (n,m,p) has a natural
    decomposition where each "slice" corresponds to a value of the middle index.
    
    Let me try rank 49 using a known pattern.
    """
    n, m, p = DIMENSIONS
    
    # Use a constructive approach based on partitioning
    # Split m=4 into 2+2
    # For each half, we do a 3x2 * 2x5 multiplication
    
    # For 3x2 * 2x5, the schoolbook rank is 3*2*2=12, but we can do better.
    # Actually, 3x2 * 2x5: A is 3x2, B is 2x5, C is 3x5.
    # The tensor has 3*2*5=30 entries.
    
    # Let me use a direct construction for the full 3x4x5 case.
    # I'll use rank 50 with a structured initialization and verify.
    
    # Construct U, V, W for rank R
    R = 50
    
    # Initialize with a pattern that's close to the solution
    # Use the schoolbook as a starting point and try to merge some terms
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Start with schoolbook (rank 60) and try to compress
    # For now, return schoolbook as a valid fallback
    rank = n * m * p  # 60
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
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank_optimization(R, time_budget=10.0):
    """Try to find a rank-R decomposition using optimization."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    max_restarts = 3
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with structured pattern
        U_init = np.random.randn(n * m, R) * 0.1
        V_init = np.random.randn(m * p, R) * 0.1
        W_init = np.random.randn(p * n, R) * 0.1
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*R]
            V_flat = x[n*m*R : n*m*R + m*p*R]
            W_flat = x[n*m*R + m*p*R :]
            
            U = U_flat.reshape(n, m, R)
            V = V_flat.reshape(m, p, R)
            W = W_flat.reshape(p, n, R)
            
            T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            T_target = np.ones((n, m, p))
            error = np.sum((T_computed - T_target) ** 2)
            return error
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12})
        except Exception:
            continue
        
        if res.fun < 1e-8:
            U_sol = res.x[:n*m*R].reshape(n*m, R)
            V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Try snapping to half-integers
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            if _verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            
            # Try integers
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    total_start = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    # First, try optimization for lower ranks
    ranks_to_try = [45, 48, 50, 52, 55]
    
    for R in ranks_to_try:
        if time.time() - total_start > deadline:
            break
        
        remaining = deadline - (time.time() - total_start)
        if remaining < 5.0:
            break
        
        time_budget = min(15.0, remaining * 0.7)
        
        result = _try_rank_optimization(R, time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 60)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
