"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check a few entries for exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if True else 0.0  # T[i,j,k] = 1 for all (i,j,k) in matrix mult
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_to_half_integer(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 100
    
    # Start with the trivial rank-100 decomposition as fallback
    U_fallback = [[0] * trivial_rank for _ in range(n * m)]
    V_fallback = [[0] * trivial_rank for _ in range(m * p)]
    W_fallback = [[0] * trivial_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = trivial_rank
    
    # Try SGD with l1 regularization to find lower rank solutions
    # Use a smaller rank target and see if we can find a valid decomposition
    np.random.seed(42)
    
    for target_rank in [99, 98, 97, 96, 95]:
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        lr = 0.01
        l1_lambda = 0.001
        max_iters = 2000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradient of loss = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            # This is expensive, so sample a subset
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Sample some (i,j,k) triples
            num_samples = min(100, n * m * p)
            for _ in range(num_samples):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                # Compute contribution
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                val = np.sum(prod)
                diff = val - 1.0
                
                # Gradients
                grad_U[u_idx, :] += 2 * diff * V[v_idx, :] * W[w_idx, :]
                grad_V[v_idx, :] += 2 * diff * U[u_idx, :] * W[w_idx, :]
                grad_W[w_idx, :] += 2 * diff * U[u_idx, :] * V[v_idx, :]
                
                loss += diff * diff
            
            # Add L1 regularization gradient (subgradient)
            grad_U += l1_lambda * np.sign(U)
            grad_V += l1_lambda * np.sign(V)
            grad_W += l1_lambda * np.sign(W)
            
            # SGD update
            U -= lr * grad_U / num_samples
            V -= lr * grad_V / num_samples
            W -= lr * grad_W / num_samples
            
            # Check if we've converged and can round to half-integers
            if it % 200 == 0 and it > 0:
                # Try rounding to half-integers
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.sum(U_rounded[u_idx, :] * V_rounded[v_idx, :] * W_rounded[w_idx, :])
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Found a valid lower-rank decomposition!
                    best_rank = target_rank
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    break
        
        if best_rank < trivial_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
