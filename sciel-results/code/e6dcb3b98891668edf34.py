import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition is valid using vectorized operations."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the full tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Create target tensor: 1 if (i,j,k) is in support, 0 otherwise
    # Support is defined by the matrix multiplication structure:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The tensor T should be 1 at all (i,j,k) and 0 elsewhere? 
    # Wait, the tensor for matrix multiplication M*A is T_{i,j,k} = delta_{jk}? No.
    # The tensor T_{i,j,k} such that (M A)_{i,k} = sum_j M_{i,j} A_{j,k}.
    # The decomposition is sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)} = delta_{j,k}? No.
    # The standard tensor for matmul (n x m) * (m x p) -> (n x p) is T_{i,j,k} = 1 if the indices match the multiplication pattern?
    # Actually, the tensor is T_{i,j,k} = delta_{j,k} is wrong.
    # The tensor is T_{i,j,k} = 1 for all i,j,k? No.
    # Let's look at the current code's _verify.
    # It checks val == 1.0 for all i,j,k.
    # And val == 0.0 for others?
    # The loop:
    # for i in range(n): for j in range(m): for k in range(p):
    #   val = ...
    #   if abs(val - 1.0) > tol: return False
    # This implies the target tensor is ALL ONES.
    # Is that correct for matrix multiplication?
    # If T_{i,j,k} = 1, then sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1.
    # This is NOT matrix multiplication. Matrix multiplication tensor is T_{i,j,k} = delta_{j,k} is also not right.
    # The tensor for matrix multiplication (n x m) (m x p) is T_{i,j,k} = 1 if the indices are coupled?
    # No, the tensor T is defined such that T_{i,j,k} = 1 if (i,j,k) corresponds to a term in the product.
    # The product C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T_{i,j,k} = delta_{j,j} ? No.
    # The tensor T_{i,j,k} is 1 for all i,j,k? No, that would be a rank-1 tensor.
    # The rank of the matrix multiplication tensor is what we are looking for.
    # The tensor is T_{i,j,k} = 1 for all i,j,k? No.
    # Let's re-read the task description.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
    # The tensor is T_{i,j,k} = 1 if the indices form a valid multiplication term?
    # Actually, the standard definition of the matrix multiplication tensor T_{n,m,p} is:
    # T_{i,j,k} = 1 if the indices are such that they contribute to the product?
    # No, the tensor is T_{i,j,k} = 1 for all i,j,k is WRONG.
    # The tensor is T_{i,j,k} = delta_{j,k} is WRONG.
    # The tensor is T_{i,j,k} = 1 for all i,j,k? 
    # Let's check the schoolbook decomposition in the current code.
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1.
    # For a specific col, U is non-zero only at (i,j), V at (j,k), W at (k,i).
    # The sum is 1 only when (i,j,k) matches the specific col's indices.
    # So for each (i,j,k), there is exactly one col where U,V,W are 1.
    # So the sum is 1 for all (i,j,k).
    # So the target tensor IS ALL ONES.
    # This means the tensor T_{i,j,k} = 1 for all i,j,k.
    # This is the tensor for the identity map on R^{n*m*p}?
    # No, this is the tensor for matrix multiplication?
    # Wait. If T_{i,j,k} = 1, then the bilinear form is sum_{i,j,k} x_i y_j z_k = (sum x)(sum y)(sum z).
    # This is NOT matrix multiplication.
    # Matrix multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor for this is T_{i,j,k} = delta_{j,k} is not right.
    # The tensor is T_{i,j,k} = 1 if the indices are linked?
    # Actually, the tensor T_{i,j,k} is 1 for all i,j,k is the tensor for the product of three vectors.
    # Let's look at the indices in the current code's _verify again.
    # It checks val == 1.0 for all i,j,k.
    # And it checks val == 0.0 for "others".
    # But the loop covers ALL i,j,k in range(n), range(m), range(p).
    # So it checks that T_{i,j,k} = 1 for ALL i,j,k.
    # This implies the target tensor is the all-ones tensor.
    # This is strange for matrix multiplication.
    # Unless... the indices in the task description are different?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Maybe the tensor is NOT all ones?
    # Let's re-read the task carefully.
    # "Matrix multiplication 3,4,6"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
    # Usually, the tensor T_{i,j,k} is 1 if the indices (i,j,k) are such that j is the shared dimension.
    # But here, the indices are i (1..3), j (1..4), k (1..6).
    # The multiplication is A (3x4) * B (4x6) = C (3x6).
    # C_{i,k} = sum_{j=1}^4 A_{i,j} B_{j,k}.
    # The tensor T_{i,j,k} is defined such that C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}.
    # This implies T_{i,j,k} = 1 for all i,j,k?
    # Yes! Because for each i,k, we sum over j. The coefficient of A_{i,j} B_{j,k} is 1.
    # So T_{i,j,k} = 1 for all i,j,k.
    # So the target tensor IS the all-ones tensor.
    # My confusion was unfounded. The tensor for matrix multiplication M_{n,m,p} is indeed the all-ones tensor of shape (n,m,p).
    # The "zeros" check in the original code:
    # "support" is the set of all (i,j,k).
    # "for u in range(n*m): for v in range(m*p): for w in range(p*n):"
    # This iterates over ALL possible index combinations for the factors.
    # It checks if (u,v,w) is in support.
    # Support contains (i*m+j, j*p+k, k*n+i) for all i,j,k.
    # Are there (u,v,w) that are NOT in support?
    # u = i*m+j. v = j*p+k. w = k*n+i.
    # Given u, v, w, can we always find i,j,k?
    # i = u // m, j = u % m.
    # j' = v // p, k = v % p.
    # i' = w % n, k' = w // n.
    # If j == j' and i == i' and k == k', then it is in support.
    # If not, it is not in support.
    # So there ARE zeros in the tensor space if we consider the full hypercube (n*m) x (m*p) x (p*n).
    # The tensor T is defined on the full space of factors, but the matrix multiplication only cares about the "diagonal" where indices match.
    # So the target is:
    # T[u,v,w] = 1 if u,v,w correspond to consistent i,j,k.
    # T[u,v,w] = 0 otherwise.
    # So the target is NOT all ones. It is a sparse tensor with 1s on the "diagonal" and 0s elsewhere.
    # The original code's _verify checks:
    # 1. For consistent (i,j,k), val == 1.
    # 2. For inconsistent (u,v,w), val == 0.
    # This is the correct interpretation.
    
    # So, for the vectorized check:
    # We need to construct the target tensor T_target of shape (n*m, m*p, p*n).
    # Or rather, we can just compute the error on the "1s" and "0s" separately.
    
    # Let's create the target tensor.
    # T_target = np.zeros((n*m, m*p, p*n))
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       u = i*m + j
    #       v = j*p + k
    #       w = k*n + i
    #       T_target[u, v, w] = 1.0
       
    # But this is large. (12 * 24 * 18) = 5184 elements. Not too bad.
    # However, we can be smarter.
    # The error is sum_{u,v,w} (T[u,v,w] - T_target[u,v,w])^2.
    
    # Let's just compute the full tensor T of shape (n*m, m*p, p*n) using einsum?
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c].
    # This is an outer product sum.
    # T = np.einsum('uc,vc,wc->uvw', U, V, W)
    
    # Then compare with T_target.
    
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    err = np.sum((T - T_target)**2)
    return err < tol

def _try_optimize(n, m, p, rank, seed, max_iter=2000, tol=1e-10):
    """Try to find a valid decomposition of given rank using L-BFGS-B."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Flatten for optimization
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    def objective(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank:n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        T = np.einsum('uc,vc,wc->uvw', U, V, W)
        err = np.sum((T - T_target)**2)
        return err
        
    def gradient(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank:n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        T = np.einsum('uc,vc,wc->uvw', U, V, W)
        diff = T - T_target # (n*m, m*p, p*n)
        
        # dU[u,c] = sum_{v,w} diff[u,v,w] * V[v,c] * W[w,c]
        dU = np.einsum('uvw,vc,wc->uc', diff, V, W)
        # dV[v,c] = sum_{u,w} diff[u,v,w] * U[u,c] * W[w,c]
        dV = np.einsum('uvw,uc,wc->vc', diff, U, W)
        # dW[w,c] = sum_{u,v} diff[u,v,w] * U[u,c] * V[v,c]
        dW = np.einsum('uvw,uc,vc->wc', diff, U, V)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if res.success or res.fun < 1e-8:
            x_opt = res.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest 0.5
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if _verify(U_round, V_round, W_round, n, m, p):
                return U_round, V_round, W_round, res.fun
    except Exception:
        pass
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward
    max_rank_to_try = min(60, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds for each rank
        max_seeds = 5 if rank < 30 else 3
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            
            result = _try_optimize(n, m, p, rank, seed, max_iter=1000)
            if result is not None:
                U_opt, V_opt, W_opt, err = result
                best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                best_rank = rank
                return best_result
    
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W
