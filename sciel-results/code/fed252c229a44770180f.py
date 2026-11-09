import numpy as np
import random
from fractions import Fraction
import time
import scipy.optimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a low-rank decomposition using optimization
    # Target rank: try a few values below 63
    # The schoolbook rank is 63.
    # We know 3x3x3 has rank 23. 3x3x7 can be split into 3x3x3 + 3x3x4.
    # Let's try to find a decomposition with rank 50-60.
    
    # We will use a randomized search with gradient descent for a specific rank.
    # To increase the chance of success, we try a few ranks.
    for rank in [55, 56, 58, 60]:
        start_time = time.time()
        result = find_decomposition_optimization(n, m, p, rank, timeout=20)
        if result is not None:
            return result
        if time.time() - start_time > 100:
            break
            
    # Fallback to schoolbook
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
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

def find_decomposition_optimization(n, m, p, rank, timeout=20):
    # Flatten the tensor T
    # T[i, j, k] = 1 if it corresponds to the multiplication term
    # The multiplication tensor M_{n,m,p} has entries M_{i,j; j,k; k,i} = 1?
    # Let's define the tensor T of shape (n*m, m*p, p*n)
    # T[(i,j), (j,k), (k,i)] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # We want U (nm x R), V (mp x R), W (pn x R)
    # such that sum_r U[:,r] outer V[:,r] outer W[:,r] = T
    # This is equivalent to:
    # For all a, b, c: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Parameters: flatten U, V, W
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_params = size_U + size_V + size_W
    
    def objective(params):
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Compute the reconstructed tensor
        # Rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This can be computed efficiently?
        # Rec = sum_r np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        # This is O(R * nm * mp * pn).
        # nm=9, mp=21, pn=21. R=55.
        # 55 * 9 * 21 * 21 = 215k operations.
        # Very fast.
        
        Rec = np.zeros_like(T)
        for r in range(rank):
            Rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        # Return squared error
        diff = Rec - T
        return np.sum(diff * diff)

    # Random initialization
    np.random.seed(42)
    params = np.random.randn(total_params) * 0.1
    
    # Use scipy minimize
    # L-BFGS-B is good for large unconstrained problems
    start = time.time()
    try:
        res = scipy.optimize.minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 10000})
        if time.time() - start > timeout:
            return None
        
        final_cost = res.fun
        # Check if the cost is close to 0
        if final_cost < 1e-6:
            U_flat = res.x[:size_U]
            V_flat = res.x[size_U:size_U+size_V]
            W_flat = res.x[size_U+size_V:]
            
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            # Convert to list of lists with integers/halves if possible?
            # The prompt says "integer/half-integer entries".
            # The optimization gives floats.
            # We need to check if they are close to half-integers.
            # If not, we might need to round or reject.
            # For now, let's check if they are close to 0.5 * integer.
            
            U_list = []
            for i in range(n*m):
                row = []
                for r in range(rank):
                    val = U[i,r]
                    # Round to nearest half-integer
                    rounded = round(val * 2) / 2.0
                    if abs(val - rounded) > 1e-3:
                        return None
                    row.append(rounded)
                U_list.append(row)
                
            V_list = []
            for i in range(m*p):
                row = []
                for r in range(rank):
                    val = V[i,r]
                    rounded = round(val * 2) / 2.0
                    if abs(val - rounded) > 1e-3:
                        return None
                    row.append(rounded)
                V_list.append(row)
                
            W_list = []
            for i in range(p*n):
                row = []
                for r in range(rank):
                    val = W[i,r]
                    rounded = round(val * 2) / 2.0
                    if abs(val - rounded) > 1e-3:
                        return None
                    row.append(rounded)
                W_list.append(row)
                
            return {"U": U_list, "V": V_list, "W": W_list}
            
    except Exception as e:
        pass
        
    return None
# EVOLVE-BLOCK-END
