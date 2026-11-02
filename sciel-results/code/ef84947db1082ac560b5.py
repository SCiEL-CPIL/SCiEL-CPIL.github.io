"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from scipy.optimize import minimize
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[i, j, k, l] = 1 if j == k else 0
    # Note: The problem definition says W[k*n+i, r] in the prompt text 
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # But the interface says W shape (p*n, rank). 
    # Let's re-read carefully.
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "T[i,j,k,l] = delta_{jk}"
    # Standard matmul: C = A B. C[i,l] = sum_j A[i,j] B[j,l].
    # Tensor T[i,j,k,l] corresponds to A[i,j] * B[j,k] * W[?,?].
    # Wait, the tensor for matmul is usually T[i, j, k, l] = A[i,j] B[k,l] delta_{j,k}? No.
    # The multiplication tensor $M_{n,m,p}$ has indices $i \in [n], j \in [m], k \in [m], l \in [p]$.
    # $M_{i,j,k,l} = 1$ if $j=k$, else 0.
    # Decomposition: $M_{i,j,k,l} = \sum_r U_{i,j,r} V_{k,l,r} W_{?}$
    # The prompt says:
    # U shape (n*m, rank) -> Index i*m+j
    # V shape (m*p, rank) -> Index j*p+k ?? 
    # W shape (p*n, rank) -> Index k*n+l ??
    
    # Let's check the indices in the prompt's compute function:
    # T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
    # This matches:
    # U[i*m+j]
    # V[j*p+k]  (Here j is from A, k is from B? No, j is middle index of A, k is middle index of B)
    # In matmul C = AB, C[i,l] = sum_j A[i,j] B[j,l].
    # So the middle indices must match.
    # A has indices (i, j). B has indices (j, l).
    # So V should depend on j and l?
    # But the prompt says V[j*p+k]. This implies V depends on j and k.
    # And W[k*n+l]. This implies W depends on k and l.
    # And U[i*m+j].
    # Sum over r.
    # If j=k, then we have U[i,j] V[j,l] W[j,l]? No.
    
    # Let's look at the target: delta_{jk}.
    # If j=k, T=1.
    # The term is U[i*m+j] V[j*p+k] W[k*n+l].
    # If j=k, this is U[i,j] V[j,l] W[j,l]? No, V index is j*p+k = j*p+j. W index is k*n+l = j*n+l.
    # This doesn't look like standard matmul decomposition.
    
    # Standard decomposition for C = AB:
    # C[i,l] = sum_j A[i,j] B[j,l].
    # We want to represent the operation.
    # Usually, we define the tensor $T_{i, j, k, l}$ such that $T_{i, j, k, l} = 1$ if $j=k$ else 0.
    # Then $C[i,l] = \sum_{j,k} A[i,j] B[k,l] T_{i,j,k,l}$? No, that would sum over j,k.
    # The decomposition is $T_{i,j,k,l} = \sum_r U_{i,j,r} V_{k,l,r}$.
    # But we have 3 factors U, V, W.
    # This is a 3-way decomposition?
    # The prompt says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
    # This is a 3-tensor decomposition of the matmul tensor?
    # The matmul tensor is a 4-index object $T_{i,j,k,l}$.
    # It can be viewed as a 3-tensor if we group indices?
    # Or is it a decomposition of the form:
    # $\sum_r U_{i,j,r} V_{k,l,r} = \delta_{jk}$?
    # If so, U and V are the factors. Where does W come in?
    
    # Let's re-read the prompt's compute function in the current code:
    # T[i,j,k,l] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+l]
    # This is a 3-fold product summing to a 4-index tensor.
    # This is valid if the indices align.
    # U depends on (i,j).
    # V depends on (j,k).
    # W depends on (k,l).
    # Summing over r, we get a function of (i,j,k,l).
    # We want this to be $\delta_{jk}$.
    # Note that $\delta_{jk}$ does not depend on i or l.
    # So for any i, l, the sum must be 1 if j=k, 0 otherwise.
    
    # This is a valid formulation.
    # The "rank" is R.
    # We need to find U, V, W.
    
    # My previous analysis of indices:
    # U: (n*m, R)
    # V: (m*p, R) -> Index j*p+k? 
    # Wait, if V index is j*p+k, and j in [0,m), k in [0,m)?
    # But V shape is (m*p, R). So index ranges [0, m*p).
    # If p=3, m=2, size 6.
    # Indices j*p+k. j in [0,2), k in [0,3)?
    # But in the tensor T[i,j,k,l], k is the middle index of B?
    # In matmul, B is m x p. Indices (j, l) with j in [0,m), l in [0,p).
    # So the middle index of B is j? No, B[j,l].
    # A[i,j].
    # So the shared index is j.
    # So T should depend on i, j, j, l?
    # The prompt uses k as the second middle index.
    # T[i,j,k,l] = delta_{jk}.
    # This implies j and k are the indices to be matched.
    # So A has (i,j), B has (k,l).
    # So B is indexed by (k,l). k in [0,m), l in [0,p).
    # So V depends on (j,k). j from A, k from B.
    # V shape (m*p, R). Index j*p+k?
    # If j in [0,m) and k in [0,p)? No, k should be in [0,m) to match j.
    # But V size is m*p.
    # If V index is j*p+k, and j in [0,m), k in [0,p), then V covers all pairs (j,k) where j is row of A, k is col of B?
    # But we only care about j=k? No, V is a factor.
    
    # Let's stick to the prompt's definition:
    # U[i*m+j, r]
    # V[j*p+k, r]
    # W[k*n+l, r]
    # Target: delta_{jk}.
    
    # This means:
    # For fixed i, l:
    # sum_{j,k} U[i,j] V[j,k] W[k,l] (in matrix notation if we reshape)
    # Wait, sum is over r.
    # T[i,j,k,l] = sum_r U[i,j,r] V[j,k,r] W[k,l,r].
    # We want T[i,j,k,l] = 1 if j=k, else 0.
    
    # This is exactly the decomposition of the identity tensor in the middle indices.
    # This is possible.
    
    # Vectorized computation:
    # U_3d: (n, m, R)
    # V_3d: (m, p, R)  -- Wait, V index is j*p+k. j in [0,m), k in [0,p)?
    # If V index is j*p+k, then j is the first index, k is the second.
    # But in T[i,j,k,l], k is the index that must equal j.
    # So k ranges [0,m)?
    # But V shape is m*p. If p=3, m=2.
    # If k ranges [0,m)=2, then V only uses indices 0..3?
    # Indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4.
    # Indices 2 and 5 are unused?
    # W index k*n+l. k in [0,m)=2. l in [0,p)=3.
    # W shape p*n = 6.
    # Indices: 0*2+0=0, 0*2+1=1, 0*2+2=2, 1*2+0=3, 1*2+1=4, 1*2+2=5.
    
    # So V is effectively a (m, m) matrix embedded in (m, p)?
    # Or is k in [0,p)?
    # If k in [0,p), then delta_{jk} is only defined if j and k are in same range?
    # j in [0,m). k in [0,p).
    # If m != p, delta_{jk} is 0 if j != k.
    # If m=2, p=3.
    # j=0, k=0 -> 1.
    # j=0, k=1 -> 0.
    # j=0, k=2 -> 0.
    # j=1, k=0 -> 0.
    # j=1, k=1 -> 1.
    # j=1, k=2 -> 0.
    
    # So the target tensor is sparse.
    
    # Vectorization:
    # U: (n, m, R)
    # V: (m, p, R)  (Index j, k)
    # W: (p, n, R)  (Index k, l)
    
    # T[i, j, k, l] = sum_r U[i,j,r] V[j,k,r] W[k,l,r]
    # np.einsum('ijr,jkr,klr->ijkl', U, V, W)
    
    # This is correct.
    
    def solve_rank(R):
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(n, m, R) * 0.5
        V = np.random.randn(m, p, R) * 0.5
        W = np.random.randn(p, n, R) * 0.5
        
        # Flatten for optimizer
        def flatten(x):
            return x.flatten()
        
        def unflatten(x):
            return x.reshape(n, m, R)
        def unflatten_v(x):
            return x.reshape(m, p, R)
        def unflatten_w(x):
            return x.reshape(p, n, R)
            
        # Objective
        def objective(x):
            U_3d = x[:n*m*R].reshape(n, m, R)
            V_3d = x[n*m*R:n*m*R+m*p*R].reshape(m, p, R)
            W_3d = x[n*m*R+m*p*R:].reshape(p, n, R)
            
            T = np.einsum('ijr,jkr,klr->ijkl', U_3d, V_3d, W_3d)
            
            # Target
            T_target = np.zeros((n, m, p, n)) # Wait, T is (n, m, p, n)?
            # T[i,j,k,l]. i in n, j in m, k in p, l in n.
            # delta_{jk}. j in m, k in p.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if j == k:
                            T_target[i, j, k, :] = 1.0 # l can be anything?
                            # Wait, T[i,j,k,l] = 1 if j==k.
                            # It does not depend on l.
                            # So for all l, T[i,j,k,l] = 1.
                            pass
            
            # Construct T_target efficiently
            T_target = np.zeros((n, m, p, n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if j == k:
                            T_target[i, j, k, :] = 1.0
                        
            error = T - T_target
            return np.sum(error**2)
        
        # Optimize
        res = minimize(objective, flatten(np.concatenate([U.flatten(), V.flatten(), W.flatten()])), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success and res.fun < 1e-8:
            x = res.x
            U_sol = x[:n*m*R].reshape(n, m, R)
            V_sol = x[n*m*R:n*m*R+m*p*R].reshape(m, p, R)
            W_sol = x[n*m*R+m*p*R:].reshape(p, n, R)
            
            # Rationalize
            # Try to find integer/half-integer solution
            for scale in [1, 2, 4, 8, 16]:
                U_s = U_sol * scale
                V_s = V_sol * scale
                W_s = W_sol * scale
                
                U_r = np.round(U_s)
                V_r = np.round(V_s)
                W_r = np.round(W_s)
                
                # Check error
                T_check = np.einsum('ijr,jkr,klr->ijkl', U_r, V_r, W_r)
                if np.max(np.abs(T_check - T_target)) < 1e-6:
                    # Return solution
                    # Convert to lists
                    U_list = [[int(x) for x in row] for row in U_r.reshape(n*m, R)]
                    V_list = [[int(x) for x in row] for row in V_r.reshape(m*p, R)]
                    W_list = [[int(x) for x in row] for row in W_r.reshape(p*n, R)]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try half-integers if scale=1 failed? 
            # If scale=2, U_s is integer, so U_sol is half-integer.
            # The returned values are integers (scaled).
            # But the interface expects the values that sum to 1.
            # If I return U_r (which is 2*U_sol), then the sum will be 2*1 = 2.
            # I must return the actual values U_sol, which are half-integers.
            # So I should return U_r / scale.
            
            # Let's re-evaluate.
            # If U_r is integer, and U_r = U_sol * scale.
            # Then U_sol = U_r / scale.
            # The decomposition uses U_sol.
            # So I should return U_r / scale.
            
            for scale in [1, 2, 4, 8, 16]:
                U_s = U_sol * scale
                V_s = V_sol * scale
                W_s = W_sol * scale
                
                U_r = np.round(U_s)
                V_r = np.round(V_s)
                W_r = np.round(W_s)
                
                # Check if U_r is close to U_s
                if np.max(np.abs(U_r - U_s)) < 1e-4 and np.max(np.abs(V_r - V_s)) < 1e-4 and np.max(np.abs(W_r - W_s)) < 1e-4:
                    # Verify with rationalized values
                    U_rat = U_r / scale
                    V_rat = V_r / scale
                    W_rat = W_r / scale
                    
                    T_check = np.einsum('ijr,jkr,klr->ijkl', U_rat, V_rat, W_rat)
                    if np.max(np.abs(T_check - T_target)) < 1e-6:
                        U_list = [[float(x) for x in row] for row in U_rat.reshape(n*m, R)]
                        V_list = [[float(x) for x in row] for row in V_rat.reshape(m*p, R)]
                        W_list = [[float(x) for x in row] for row in W_rat.reshape(p*n, R)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        return None

    # Try ranks
    for R in range(8, 13):
        sol = solve_rank(R)
        if sol:
            return sol
            
    # Fallback
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
# EVOLVE-BLOCK-END
