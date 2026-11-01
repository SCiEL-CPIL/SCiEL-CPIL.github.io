"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify(U, V, W, n=2, m=2, p=4, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that off-diagonal terms are zero
    # For a given (a, b, c) where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # The term should be zero unless i=i', j=j', k=k'
    # This is expensive, so we spot-check
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Determine if this is a valid multiplication term
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                # Valid if i==i2, j==j2, k==k2
                if i == i2 and j == j2 and k == k2:
                    continue  # Already checked, should be 1
                
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > tol:
                    return False
    
    return True

def _strassen_rank14():
    """Construct the rank-14 decomposition using two independent Strassen 2x2x2 multiplications."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    # Indexing:
    # U[i*m + j]: A[i,j], i in {0,1}, j in {0,1}
    # V[j*p + k]: B[j,k], j in {0,1}, k in {0,1,2,3}
    # W[k*n + i]: C[i,k], i in {0,1}, k in {0,1,2,3}
    
    # Left half (k=0,1):
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # Right half (k=2,3):
    # B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[6], B[1,3]=V[7]
    # C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    
    # Strassen for 2x2 x 2x2:
    # m1 = (A11+A22)(B11+B22)
    # m2 = (A21+A22)B11
    # m3 = A11(B21-B22)
    # m4 = A22(B12-B11)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Left half (r=0..6)
    # m1_L: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0, r] = 1; V[5, r] = 1
    W[0, r] = 1; W[3, r] = 1  # C[0,0] += 1, C[1,1] += 1
    
    # m2_L: (A[1,0]+A[1,1]) * B[0,0]
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0, r] = 1
    W[2, r] = 1; W[3, r] = -1  # C[1,0] += 1, C[1,1] -= 1
    
    # m3_L: A[0,0] * (B[1,0]-B[1,1])
    r = 2
    U[0, r] = 1
    V[4, r] = 1; V[5, r] = -1
    W[1, r] = 1; W[3, r] = 1  # C[0,1] += 1, C[1,1] += 1
    
    # m4_L: A[1,1] * (B[0,1]-B[0,0])
    r = 3
    U[3, r] = 1
    V[1, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[2, r] = 1  # C[0,0] += 1, C[1,0] += 1
    
    # m5_L: (A[0,0]+A[0,1]) * B[1,1]
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[5, r] = 1
    W[0, r] = -1; W[1, r] = 1  # C[0,0] -= 1, C[0,1] += 1
    
    # m6_L: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[3, r] = 1  # C[1,1] += 1
    
    # m7_L: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[0, r] = 1  # C[0,0] += 1
    
    # Right half (r=7..13)
    # m1_R: (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[7, r] = 1
    W[4, r] = 1; W[7, r] = 1  # C[0,2] += 1, C[1,3] += 1
    
    # m2_R: (A[1,0]+A[1,1]) * B[0,2]
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[6, r] = 1; W[7, r] = -1  # C[1,2] += 1, C[1,3] -= 1
    
    # m3_R: A[0,0] * (B[1,2]-B[1,3])
    r = 9
    U[0, r] = 1
    V[6, r] = 1; V[7, r] = -1
    W[5, r] = 1; W[7, r] = 1  # C[0,3] += 1, C[1,3] += 1
    
    # m4_R: A[1,1] * (B[0,3]-B[0,2])
    r = 10
    U[3, r] = 1
    V[3, r] = 1; V[2, r] = -1
    W[4, r] = 1; W[6, r] = 1  # C[0,2] += 1, C[1,2] += 1
    
    # m5_R: (A[0,0]+A[0,1]) * B[1,3]
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[7, r] = 1
    W[4, r] = -1; W[5, r] = 1  # C[0,2] -= 1, C[0,3] += 1
    
    # m6_R: (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[7, r] = 1  # C[1,3] += 1
    
    # m7_R: (A[0,1]-A[1,1]) * (B[1,2]+B[1,3])
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[4, r] = 1  # C[0,2] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank13_peel():
    """
    Try to find a rank-13 decomposition by peeling off one term from the rank-14 solution.
    
    Strategy: Take the rank-14 Strassen solution and try to express one of the 14 terms
    as a linear combination of the other 13, effectively reducing the rank.
    
    More concretely: We look for a rank-13 decomposition where one of the rank-14 terms
    is replaced by a combination that cancels out.
    
    Alternative: Directly search for rank-13 by fixing the structure.
    """
    n, m, p = 2, 2, 4
    rank = 13
    
    # The 16 equations (one per (i,j,k) triple) are:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # Plus the off-diagonal constraints:
    # For each (a,b,c) not corresponding to a valid (i,j,k): 
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # This is a system of polynomial equations. Let's try a structured approach.
    
    # Idea: Since the problem is 2x2 x 2x4, we can think of it as:
    # C[:, :2] = A * B[:, :2]  (2x2 x 2x2, rank 7)
    # C[:, 2:] = A * B[:, 2:]  (2x2 x 2x2, rank 7)
    # Total rank 14 if independent.
    
    # To get rank 13, we need to share one multiplication between the two halves.
    # This means finding a bilinear form that contributes to both C[:, :2] and C[:, 2:].
    
    # Let's try to construct such a form explicitly.
    # Suppose we have a term that contributes to both halves.
    # For example, a term involving A[0,0] that multiplies a combination of B entries from both halves.
    
    # Let's try a direct optimization approach.
    # We'll parameterize U, V, W with 13 columns and minimize the residual.
    
    np.random.seed(12345)
    
    best_solution = None
    best_residual = np.inf
    
    for attempt in range(200):
        # Random initialization with structure
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        # Initialize with sparse random patterns
        for r in range(rank):
            # U: 1-2 nonzeros per column
            num_u = np.random.choice([1, 2], p=[0.4, 0.6])
            u_idx = np.random.choice(4, size=num_u, replace=False)
            for idx in u_idx:
                U[idx, r] = np.random.choice([-1, 1])
            
            # V: 1-3 nonzeros per column
            num_v = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            v_idx = np.random.choice(8, size=num_v, replace=False)
            for idx in v_idx:
                V[idx, r] = np.random.choice([-1, 1])
            
            # W: 1-2 nonzeros per column
            num_w = np.random.choice([1, 2], p=[0.5, 0.5])
            w_idx = np.random.choice(8, size=num_w, replace=False)
            for idx in w_idx:
                W[idx, r] = np.random.choice([-1, 1])
        
        # Compute residual
        def residual(params):
            U_flat = params[:4*rank]
            V_flat = params[4*rank:4*rank+8*rank]
            W_flat = params[4*rank+8*rank:]
            
            U = U_flat.reshape(4, rank)
            V = V_flat.reshape(8, rank)
            W = W_flat.reshape(8, rank)
            
            res = 0.0
            # Check nonzero entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        res += (val - 1.0) ** 2
            
            # Check some off-diagonal entries (sampled)
            for a in range(4):
                for b in range(8):
                    for c in range(8):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        
                        if i == i2 and j == j2 and k == k2:
                            continue
                        
                        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        res += val ** 2
            
            return res
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use scipy minimize to refine
        try:
            result = minimize(residual, params0, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.fun < best_residual:
                best_residual = result.fun
                U_opt = result.x[:4*rank].reshape(4, rank)
                V_opt = result.x[4*rank:4*rank+8*rank].reshape(8, rank)
                W_opt = result.x[4*rank+8*rank:].reshape(8, rank)
                best_solution = (U_opt, V_opt, W_opt)
                
                if best_residual < 1e-8:
                    # Found a good solution, verify
                    U_check = best_solution[0].tolist()
                    V_check = best_solution[1].tolist()
                    W_check = best_solution[2].tolist()
                    if _verify(U_check, V_check, W_check, n, m, p):
                        return {"U": U_check, "V": V_check, "W": W_check}
        except Exception:
            continue
    
    return None

def _try_rank12_direct():
    """
    Try to find a rank-12 decomposition directly.
    
    This is more ambitious. The theoretical minimum for 2x2 x 2x4 is unknown,
    but rank 12 would be a significant improvement over rank 14.
    """
    n, m, p = 2, 2, 4
    rank = 12
    
    np.random.seed(54321)
    
    best_solution = None
    best_residual = np.inf
    
    for attempt in range(100):
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        for r in range(rank):
            num_u = np.random.choice([1, 2], p=[0.3, 0.7])
            u_idx = np.random.choice(4, size=num_u, replace=False)
            for idx in u_idx:
                U[idx, r] = np.random.choice([-1, 1])
            
            num_v = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            v_idx = np.random.choice(8, size=num_v, replace=False)
            for idx in v_idx:
                V[idx, r] = np.random.choice([-1, 1])
            
            num_w = np.random.choice([1, 2], p=[0.4, 0.6])
            w_idx = np.random.choice(8, size=num_w, replace=False)
            for idx in w_idx:
                W[idx, r] = np.random.choice([-1, 1])
        
        def residual(params):
            U_flat = params[:4*rank]
            V_flat = params[4*rank:4*rank+8*rank]
            W_flat = params[4*rank+8*rank:]
            
            U = U_flat.reshape(4, rank)
            V = V_flat.reshape(8, rank)
            W = W_flat.reshape(8, rank)
            
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        res += (val - 1.0) ** 2
            
            for a in range(4):
                for b in range(8):
                    for c in range(8):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        
                        if i == i2 and j == j2 and k == k2:
                            continue
                        
                        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        res += val ** 2
            
            return res
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(residual, params0, method='Nelder-Mead', 
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.fun < best_residual:
                best_residual = result.fun
                U_opt = result.x[:4*rank].reshape(4, rank)
                V_opt = result.x[4*rank:4*rank+8*rank].reshape(8, rank)
                W_opt = result.x[4*rank+8*rank:].reshape(8, rank)
                best_solution = (U_opt, V_opt, W_opt)
                
                if best_residual < 1e-8:
                    U_check = best_solution[0].tolist()
                    V_check = best_solution[1].tolist()
                    W_check = best_solution[2].tolist()
                    if _verify(U_check, V_check, W_check, n, m, p):
                        return {"U": U_check, "V": V_check, "W": W_check}
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with the known rank-14 solution
    best = _strassen_rank14()
    best_rank = 14
    
    # Verify the rank-14 solution
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-13 solution
    if time.time() < deadline - 60:
        result13 = _try_rank13_peel()
        if result13 is not None:
            return result13
    
    # Try to find rank-12 solution
    if time.time() < deadline - 30:
        result12 = _try_rank12_direct()
        if result12 is not None:
            return result12
    
    # Return the best solution found
    return best
# EVOLVE-BLOCK-END
