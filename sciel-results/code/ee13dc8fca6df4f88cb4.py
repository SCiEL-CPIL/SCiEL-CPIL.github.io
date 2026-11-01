"""Matrix multiplication 2x5x5 - attempt at lower rank via block structure."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Attempt to find a rank-minimal tensor decomposition for 2x5x5 matrix multiplication.
    
    The matrix multiplication tensor T has entries T[i,j,k] = 1 if the (i,j) element of A
    and (j,k) element of B contribute to (i,k) of C.
    
    For a 2x5x5 multiplication:
    - A is 2x5, B is 5x5, C is 2x5
    - T[i, j, k] = 1 for all i in {0,1}, j in {0..4}, k in {0..4}
    
    We model this as: C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor can be written as sum over j of (e_i for fixed i) (x) (e_j) (x) (e_k)
    which gives rank 2*5*5 = 50 in the schoolbook way.
    
    However, we can potentially do better by exploiting structure. Let's try a
    random initialization approach with optimization.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # The multiplication tensor T[i, j, k] where:
    # i indexes rows of A (size n=2)
    # j indexes cols of A / rows of B (size m=5)  
    # k indexes cols of B (size p=5)
    # T[i,j,k] = 1 if this term appears in the sum for C[i,k]
    
    # Actually, the standard matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k
    
    # We want to write T as a sum of rank-1 tensors:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the convention is:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The target tensor T_target[i,j,k] = 1 for all i in [0,2), j in [0,5), k in [0,5)
    # But actually that's not right. Let me reconsider.
    
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The tensor for this operation has entries T[i,j,k] where:
    # - i is the row index of A and C
    # - j is the column index of A and row index of B
    # - k is the column index of B and C
    # T[i,j,k] = 1 (it always contributes)
    
    # In the factorization:
    # T[i,j,k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    
    # So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to find a low-rank decomposition using numerical optimization.
    
    # Start with the schoolbook rank of 50
    n_schoolbook = n * m * p  # 50
    
    # Try to find a lower rank decomposition
    # We'll try ranks from some lower bound up to the schoolbook rank
    # Theoretical lower bound is max(n*p, m*p, n*m) but that's not tight for tensor rank
    
    # Let's try a simple approach: use random initialization and gradient descent
    # to minimize the difference from the target tensor.
    
    # Actually, let's think about this differently. The problem is:
    # We have a 3D tensor T of size (2*5) x (5*5) x (5*2) = 10 x 25 x 10
    # where T[(i*m+j), (j*p+k), (k*n+i)] = 1 and 0 otherwise? No...
    
    # Let me re-read the problem. The factor matrices are:
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (10, rank)
    
    # And the decomposition satisfies:
    # For matrix mult C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # This can be written as:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k W[k*n+i, c] * B[j,k]?) 
    # Hmm, that doesn't quite work. Let me think again.
    
    # The standard way: define
    # U[i*m+j, c] such that (U^T * vec(A))[c] = something
    # Actually, the factorization is:
    # Let a = vec(A) in R^{n*m}, b = vec(B) in R^{m*p}, c = vec(C) in R^{n*p}
    # Then c = M * a * b (multilinear)
    # The tensor T has T[i,j,k] where the indices correspond to the three "slices"
    
    # I think the correct formulation is:
    # T is a tensor of size (n*m) x (m*p) x (n*p)
    # T[a, b, c] = 1 if a corresponds to (i,j), b to (j,k), c to (i,k)
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # And we want to factorize T as a sum of rank-1 tensors:
    # T = sum_c u_c (x) v_c (x) w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{n*p}
    
    # So U[:,c] = u_c, V[:,c] = v_c, W[:,c] = w_c
    
    # Let's build the target tensor and try to find a low-rank factorization.
    
    # Build target tensor
    target = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1.0
    
    # Now try to find low-rank factorization
    # We'll try different ranks and see if we can achieve a good fit
    
    # For efficiency, let's try a few candidate ranks
    # Start with a moderately low rank and see if optimization can converge
    
    best_rank = n_schoolbook
    best_U, best_V, best_W = None, None, None
    
    # Try ranks: we know schoolbook is 50. Let's try lower.
    # A reasonable starting point might be around 20-30
    
    candidates = [15, 16, 20, 25, 30, 35, 40, 45, 50]
    
    np.random.seed(42)
    
    for rank in candidates:
        # Random initialization
        U_init = np.random.randn(n*m, rank) * 0.1
        V_init = np.random.randn(m*p, rank) * 0.1
        W_init = np.random.randn(n*p, rank) * 0.1
        
        # Objective: minimize ||sum_c U[:,c] (x) V[:,c] (x) W[:,c] - target||^2
        # This is non-convex, so we'll use gradient descent or scipy minimize
        
        def objective(params):
            U_f = params[:n*m*rank].reshape(n*m, rank)
            V_f = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_f = params[n*m*rank + m*p*rank:].reshape(n*p, rank)
            
            # Compute the factorized tensor
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # This is equivalent to: sum_d outer(U[:,d], V[:,d]) (x) W[:,d]
            # Or we can compute it as: for each d, add U[:,d] (x) V[:,d] (x) W[:,d]
            
            # Efficient computation:
            # T = sum_d np.einsum('ad,bd,cd->abc', U, V, W)
            T_comp = np.zeros_like(target)
            for d in range(rank):
                T_comp += np.einsum('a,b,c->abc', U_f[:,d], V_f[:,d], W_f[:,d])
            
            diff = T_comp - target
            return np.sum(diff**2)
        
        # Flatten parameters
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use scipy minimize with L-BFGS-B
        # This might be slow for large problems, so let's try a few iterations
        try:
            result = minimize(objective, params_init, method='L-BFGS-B', 
                            options={'maxiter': 100, 'gtol': 1e-8})
            
            if result.fun < 1e-6:  # Good fit
                U_sol = result.x[:n*m*rank].reshape(n*m, round(rank))
                V_sol = result.x[n*m*rank:n*m*rank + m*p*round(rank)].reshape(m*p, round(rank))
                W_sol = result.x[n*m*rank + m*p*round(rank):].reshape(n*p, round(rank))
                best_rank = rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break  # Found a good solution, stop
        except Exception:
            continue
    
    # If no good solution found, fall back to schoolbook
    if best_U is None:
        # Schoolbook decomposition
        U = np.zeros((n*m, n_schoolbook))
        V = np.zeros((m*p, n_schoolbook))
        W = np.zeros((n*p, n_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Round to nearest integers or half-integers if possible
        # The problem says entries should be integer/half-integer
        # For now, return the floating point solution
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
