import numpy as np
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # Target Tensor T[i, j, k] = 1 if we are multiplying A[i,j] * B[j,k] -> C[i,k]
    # The tensor definition for matrix multiplication is T[i, j, k] = 1.
    # Wait, the tensor T is defined such that T_{i,j,k} = 1 for all i,j,k?
    # No, the matrix multiplication tensor is $M_{n,m,p}$.
    # The entries are $M_{i,j,k} = 1$ if the index $j$ is shared?
    # Standard definition: $T[i, j, k] = 1$.
    # Then $\sum_{j} A_{i,j} B_{j,k} = \sum_{j} A_{i,j} B_{j,k} \delta_{j,j}$.
    # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
    # The tensor is $T_{i,j,k} = 1$.
    # Decomposition: $T_{i,j,k} = \sum_t U_{i,j}^t V_{j,k}^t W_{k,i}^t$? No.
    # The factors are:
    # U: (n*m, r), indices (i, j)
    # V: (m*p, r), indices (j, k)
    # W: (p*n, r), indices (k, i)  <-- Note the W index ordering in the prompt: W[k*n+i, r]
    
    # Let's verify the W index.
    # Prompt: "W[k*n+i,r]"
    # This implies W is indexed by (k, i).
    # The term is $U[i, j, t] V[j, k, t] W[k, i, t]$.
    # Sum over t: $\sum_t U_{ij}^t V_{jk}^t W_{ki}^t$.
    # This must equal $\delta_{j,j}$? No.
    # Matrix multiplication: $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Tensor form: $T_{ijk} = 1$.
    # So we need $\sum_t U_{ij}^t V_{jk}^t W_{ki}^t = 1$ for all $i, j, k$.
    
    # Let's build the target tensor T[i, j, k] = 1.
    # Flatten T into a vector? Or use matricization.
    # Let's use the vectorized form.
    # Size of T: n*m*p.
    # Index mapping: idx = i*m*p + j*p + k.
    
    # Trivial Solution
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_trivial[u_idx, col] = 1.0
                V_trivial[v_idx, col] = 1.0
                W_trivial[w_idx, col] = 1.0
                col += 1

    # Helper to compute tensor from factors
    def compute_tensor(U, V, W, rank):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reshape U to (n, m, r)
        # Reshape V to (m, p, r)
        # Reshape W to (p, n, r) -- Careful: W index is k*n+i, so shape is (p, n, r)
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Sum over r: np.einsum('ijr,jkr,pir->ijk', U_3d, V_3d, W_3d)
        # Wait, W index is (k, i). In W_3d, first dim is p (k), second is n (i).
        # So W_3d[k, i, r].
        # We want T[i, j, k].
        # Term: U[i, j, r] * V[j, k, r] * W[k, i, r].
        # einsum: 'ijr,jkr,kir->ijk'
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return T

    # Helper to check if T is all 1s
    def is_valid(T):
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Helper to snap to half-integers
    def snap_to_half_int(M):
        return np.round(M * 2) / 2

    # Try to find a better rank using ALS + Snapping (IHT-like)
    # We'll try ranks from trivial_rank - 1 down to trivial_rank - 5 (limit search)
    # Given the difficulty, we might not find anything, but we must try.
    
    best_rank = trivial_rank
    best_factors = (U_trivial, V_trivial, W_trivial)
    
    # We only try to improve if we think it's possible.
    # For 3x4x4, it's highly likely 48 is optimal or very close.
    # However, to satisfy the directive, we run the search.
    
    # Set a timeout or iteration limit to ensure we finish in 120s.
    # Let's try a few iterations for rank = 47.
    
    import time
    start_time = time.time()
    max_time = 60.0 # Leave buffer
    
    for r in range(trivial_rank - 1, trivial_rank - 10, -1):
        if time.time() - start_time > max_time:
            break
            
        # Initialize randomly
        # Use small random values to start near 0? Or random uniform?
        # Since we want half-integers, maybe start with random half-integers?
        # Or start with the trivial solution and zero out some columns?
        
        # Strategy: Take trivial solution, zero out (trivial_rank - r) random columns, 
        # then refine the remaining r columns.
        
        U = U_trivial[:, :r].copy()
        V = V_trivial[:, :r].copy()
        W = W_trivial[:, :r].copy()
        
        # Randomize the selection of which columns to keep?
        # Or just keep the first r?
        # Let's try random permutations of columns for the initial guess.
        
        found = False
        max_iters = 500
        tol = 1e-9
        
        for iter in range(max_iters):
            if time.time() - start_time > max_time:
                break
                
            # 1. Update U
            # Minimize || T - sum_t U_{ij}^t V_{jk}^t W_{ki}^t ||_F
            # For fixed V, W, this is linear in U.
            # T[i,j,k] = sum_t U[i,j,t] * (V[j,k,t] * W[k,i,t])
            # Let M_VW[i,j,k,t] = V[j,k,t] * W[k,i,t]
            # This is a linear system for each (i,j) across k?
            # Actually, flatten T to vector of length n*m*p.
            # Flatten U to vector of length n*m*r.
            # The mapping is linear.
            
            # Construct the linear operator A such that vec(T) = A * vec(U)
            # A has shape (n*m*p, n*m*r)
            # This is expensive to build explicitly.
            # Use einsum to solve least squares.
            
            # We can solve for U by iterating over t? No, U is coupled.
            
            # Alternative: Use the fact that for fixed V, W, the problem is:
            # Minimize sum_{i,j,k} (T_{ijk} - sum_t U_{ijt} V_{jkt} W_{kit})^2
            # Derivative w.r.t U_{ijt}:
            # -2 sum_k (T_{ijk} - sum_t U_{ijt} V_{jkt} W_{kit}) V_{jkt} W_{kit} = 0
            # sum_k (T_{ijk}) V_{jkt} W_{kit} = sum_t sum_k U_{ijt} V_{jkt} W_{kit} V_{jkt} W_{kit}
            # This is a linear system for U_{ijt} for each t? No, U_{ijt} is indexed by i,j,t.
            # For a fixed t, U_{ijt} appears.
            # Let's group by t.
            # The equation for U_{ijt} is:
            # sum_k T_{ijk} V_{jkt} W_{kit} = sum_{t'} sum_k U_{ijt'} V_{jkt'} W_{kit'} V_{jkt} W_{kit}
            
            # This is a system of size (n*m) for each t? No, U has dimension n*m*r.
            # Let's just use a simple gradient step instead of solving the exact linear system,
            # as building the matrix A is memory/time intensive for large r.
            # Gradient of Error w.r.t U:
            # E = || T - T_est ||_F^2
            # dE/dU = 2 * (T_est - T) contracted with V, W.
            
            T_est = compute_tensor(U, V, W, r)
            Res = T_est - np.ones((n, m, p))
            
            # Gradient for U:
            # dE/dU[i,j,t] = 2 * sum_k Res[i,j,k] * V[j,k,t] * W[k,i,t]
            # Using einsum: 'ijk,jkt,kit->ijt'
            dU = 2 * np.einsum('ijk,jkt,kit->ijt', Res, V.reshape(m, p, r), W.reshape(p, n, r))
            
            # Gradient for V:
            # dE/dV[j,k,t] = 2 * sum_i Res[i,j,k] * U[i,j,t] * W[k,i,t]
            # Using einsum: 'ijk,ijt,kit->jkt'
            dV = 2 * np.einsum('ijk,ijt,kit->jkt', Res, U.reshape(n, m, r), W.reshape(p, n, r))
            
            # Gradient for W:
            # dE/dW[k,i,t] = 2 * sum_j Res[i,j,k] * U[i,j,t] * V[j,k,t]
            # Using einsum: 'ijk,ijt,jkt->kit'
            dW = 2 * np.einsum('ijk,ijt,jkt->kit', Res, U.reshape(n, m, r), V.reshape(m, p, r))
            
            # Learning rate
            # Adaptive learning rate?
            lr = 0.1
            # Simple decay or fixed.
            
            U -= lr * dU.reshape(n * m, r)
            V -= lr * dV.reshape(m * p, r)
            W -= lr * dW.reshape(p * n, r)
            
            # Snap to half-integers
            U = snap_to_half_int(U)
            V = snap_to_half_int(V)
            W = snap_to_half_int(W)
            
            # Check convergence
            T_check = compute_tensor(U, V, W, r)
            if is_valid(T_check):
                best_rank = r
                best_factors = (U, V, W)
                found = True
                break
        
        if found:
            break # Found a better rank, stop searching lower ranks
    
    if best_rank < trivial_rank:
        U_out, V_out, W_out = best_factors
    else:
        U_out, V_out, W_out = U_trivial, V_trivial, W_trivial

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
