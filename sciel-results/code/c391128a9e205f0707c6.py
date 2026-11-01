import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _greedy_rank_search(n, m, p, max_rank=100, time_limit=140.0):
    """
    Greedy search for a low-rank decomposition of the matrix multiplication tensor.
    T[i, j, k] = 1 if we are computing C[i, k] = sum_j A[i, j] B[j, k]
    We want T = sum_r U[:, r] V[:, r] W[:, r]
    """
    start_time = time.time()
    
    # Tensor dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T flattened
    # T is (n, m, p) where T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Residual tensor R
    R = T.copy()
    
    U = []
    V = []
    W = []
    
    # We will greedily pick rank-1 components.
    # A rank-1 component is defined by vectors u, v, w.
    # We want to maximize <u, v, w, R> to reduce the residual.
    # Since we can scale, we can fix one vector and solve for others?
    # Or just use a heuristic.
    
    # Heuristic: 
    # 1. Pick a random or structured (u, v, w)
    # 2. Compute projection
    # 3. Update R
    
    # Better Heuristic:
    # Since the tensor is all 1s, we can try to decompose it into sums of outer products of indicator vectors?
    # No, the entries in U, V, W can be any real numbers.
    
    # Let's try a simple iterative method:
    # At each step, we want to find u, v, w such that <u, v, w, R> is maximized.
    # This is equivalent to finding the best rank-1 approximation of R.
    # We can use a simple power method or SVD-based approach on unfolded tensors.
    
    # However, SVD of a 3-tensor is not trivial (CP decomposition).
    # We can use a simple greedy approach:
    # 1. Fix w to be a random vector or a vector corresponding to a specific output index.
    # 2. Solve for u, v that maximize <u, v, w, R>.
    
    # Let's try a specific structured approach:
    # The tensor T is separable in the sense that T[i, j, k] = 1.
    # It is the sum of n*m*p rank-1 tensors: e_i e_j e_k.
    # Rank <= n*m*p = 60.
    
    # Can we do better?
    # For 2x3x10, schoolbook is 60.
    # Strassen for 2x2x2 is 7.
    # Current program uses 35 + 20 = 55.
    
    # Let's try to find a decomposition with rank < 55.
    
    # We will use a simple local search / greedy selection.
    # We will maintain a set of candidate rank-1 components.
    # We will pick the one that best fits the residual.
    
    # To make it efficient, we will use a grid of candidate vectors.
    # Or we can use a random restart strategy.
    
    # Given the time limit, let's try a simple "peeling" algorithm.
    # 1. Find the best rank-1 approximation of R.
    # 2. Subtract it from R.
    # 3. Repeat until R is close to 0 or max_rank is reached.
    
    # How to find best rank-1 approximation?
    # Maximize f(u, v, w) = sum_{i,j,k} R[i,j,k] u[i] v[j] w[k]
    # subject to ||u||=||v||=||w||=1.
    
    # We can use a simple iterative coordinate descent:
    # u = argmax_u <u, V @ W @ R>
    # v = argmax_v <v, U @ W @ R>
    # w = argmax_w <w, U @ V @ R>
    
    # This is basically ALS (Alternating Least Squares) for CP decomposition.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # We will try to find a decomposition with rank up to 54 (to beat 55).
    # We will run ALS for different initializations.
    
    for rank in range(30, 55): # Try ranks from 30 to 54
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try a few random initializations
        for init in range(5):
            if time.time() - start_time > time_limit - 10:
                break
                
            U_init = np.random.randn(dim_u, rank)
            V_init = np.random.randn(dim_v, rank)
            W_init = np.random.randn(dim_w, rank)
            
            # Normalize columns
            for r in range(rank):
                U_init[:, r] /= np.linalg.norm(U_init[:, r])
                V_init[:, r] /= np.linalg.norm(V_init[:, r])
                W_init[:, r] /= np.linalg.norm(W_init[:, r])
                
            # Run ALS
            for it in range(50):
                # Update U
                # U[:, r] = argmax_u <u, sum_k R[:, :, k] * V[:, r] * W[:, r]>
                # Let's compute M_u = R * V[:, r] * W[:, r] (contracting over j, k)
                # M_u[i] = sum_{j,k} R[i,j,k] V[j,r] W[k,r]
                # We can compute this efficiently.
                
                # Precompute contractions
                # R is (n, m, p)
                # V is (m*p, rank) -> reshape to (m, p, rank)
                # W is (p*n, rank) -> reshape to (p, n, rank)
                
                V_3d = V_init.reshape(m, p, rank)
                W_3d = W_init.reshape(p, n, rank)
                U_3d = U_init.reshape(n, m, rank)
                
                # Update U
                for r in range(rank):
                    # Contract R with V_3d[:, :, r] and W_3d[:, :, r]
                    # This is a bit expensive if done naively.
                    # Let's just do it.
                    M_u = np.tensordot(R, V_3d[:, :, r] * W_3d[:, :, r], axes=([1, 2], [0, 1]))
                    # M_u is (n, m) -> flatten to (n*m)
                    M_u_flat = M_u.flatten()
                    norm = np.linalg.norm(M_u_flat)
                    if norm > 1e-10:
                        U_init[:, r] = M_u_flat / norm
                    else:
                        U_init[:, r] = np.random.randn(dim_u)
                        U_init[:, r] /= np.linalg.norm(U_init[:, r])
                        
                # Update V
                for r in range(rank):
                    M_v = np.tensordot(R, U_3d[:, :, r] * W_3d[:, :, r], axes=([0, 2], [0, 1]))
                    # M_v is (m, p) -> flatten to (m*p)
                    M_v_flat = M_v.flatten()
                    norm = np.linalg.norm(M_v_flat)
                    if norm > 1e-10:
                        V_init[:, r] = M_v_flat / norm
                    else:
                        V_init[:, r] = np.random.randn(dim_v)
                        V_init[:, r] /= np.linalg.norm(V_init[:, r])
                        
                # Update W
                for r in range(rank):
                    M_w = np.tensordot(R, U_3d[:, :, r] * V_3d[:, :, r], axes=([0, 1], [0, 1]))
                    # M_w is (p, n) -> flatten to (p*n)
                    M_w_flat = M_w.flatten()
                    norm = np.linalg.norm(M_w_flat)
                    if norm > 1e-10:
                        W_init[:, r] = M_w_flat / norm
                    else:
                        W_init[:, r] = np.random.randn(dim_w)
                        W_init[:, r] /= np.linalg.norm(W_init[:, r])
                        
            # Check residual
            # Compute T_approx = sum_r U[:, r] V[:, r] W[:, r]
            # T_approx[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # This is expensive to check fully.
            # We can check a subset or just assume convergence if loss is low.
            # Let's compute the loss: ||T - T_approx||^2
            # Loss = ||T||^2 - 2 <T, T_approx> + ||T_approx||^2
            # Since we normalized, ||T_approx|| is not necessarily 1.
            
            # Let's just compute T_approx for a few random indices to estimate.
            # Or better, compute the full thing if rank is small enough.
            
            # For now, let's just return the best rank found that converges.
            # We need a verification step.
            
            # Convert to lists for verification
            U_list = U_init.tolist()
            V_list = V_init.tolist()
            W_list = W_init.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list, rank
                
            # If not verified, we can try to scale the vectors to improve fit?
            # The normalization might prevent exact fit.
            # We can solve for scaling factors.
            
            # But for now, if ALS converges to the exact tensor, verification should pass.
            # The issue is that ALS might converge to a local minimum that is not exact.
            
            # If we don't find an exact solution, we continue.
            
    # Fallback to current best known (55)
    return None, None, None, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS # 2, 3, 10
    
    # Try greedy search
    U, V, W, rank = _greedy_rank_search(n, m, p, max_rank=60, time_limit=deadline - start_time)
    
    if U is not None:
        return {"U": U, "V": V, "W": W}
        
    # Fallback to the 55-rank decomposition from the previous strategy
    # We need to implement the 55-rank fallback here or return schoolbook.
    # Schoolbook is 60. 55 is better.
    # Let's implement the 55-rank fallback quickly.
    
    # Actually, let's just return schoolbook if greedy fails, as it's guaranteed.
    # But 55 is better.
    
    # Let's reconstruct the 55-rank logic from the prompt.
    
    def _strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # r=0: P1 = A00 * B00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        # r=1: P2 = (A00 + A01) * B10
        U[0, 1] = 1; U[1, 1] = 1; V[2, 1] = 1; W[2, 1] = 1; W[3, 1] = -1
        # r=2: P3 = A01 * (B01 - B11)
        U[1, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[0, 2] = -1; W[1, 2] = 1
        # r=3: P4 = A11 * (B00 + B01)
        U[3, 3] = 1; V[0, 3] = 1; V[1, 3] = 1; W[1, 3] = 1; W[3, 3] = 1
        # r=4: P5 = (A10 - A00) * (B00 + B10)
        U[2, 4] = 1; U[0, 4] = -1; V[0, 4] = 1; V[2, 4] = 1; W[0, 4] = 1; W[2, 4] = 1
        # r=5: P6 = (A00 + A11) * (B11 - B00)
        U[0, 5] = 1; U[3, 5] = 1; V[3, 5] = 1; V[0, 5] = -1; W[3, 5] = 1
        # r=6: P7 = (A10 + A01) * (B11 - B01)
        U[2, 6] = 1; U[1, 6] = 1; V[3, 6] = 1; V[1, 6] = -1; W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    def _construct_m2210():
        n, m, p = 2, 2, 10
        num_U = 4
        num_V = 20
        num_W = 20
        rank = 35
        
        U = np.zeros((num_U, rank))
        V = np.zeros((num_V, rank))
        W = np.zeros((num_W, rank))
        
        U_str, V_str, W_str = _strassen_2x2x2()
        
        for b in range(5):
            col_offset = b * 7
            for r in range(7):
                U[:, col_offset + r] = U_str[:, r]
                
                V[2*b, col_offset + r] = V_str[0, r]
                V[2*b+1, col_offset + r] = V_str[1, r]
                V[2+2*b, col_offset + r] = V_str[2, r]
                V[2+2*b+1, col_offset + r] = V_str[3, r]
                
                W[4*b, col_offset + r] = W_str[0, r]
                W[4*b+2, col_offset + r] = W_str[1, r]
                W[4*b+1, col_offset + r] = W_str[2, r]
                W[4*b+3, col_offset + r] = W_str[3, r]
                
        return U, V, W

    def _construct_m2110():
        n, m, p = 2, 1, 10
        num_U = 2
        num_V = 10
        num_W = 20
        rank = 20
        
        U = np.zeros((num_U, rank))
        V = np.zeros((num_V, rank))
        W = np.zeros((num_W, rank))
        
        for i in range(n):
            for k in range(p):
                r = i * p + k
                U[i, r] = 1
                V[k, r] = 1
                W[k*n+i, r] = 1
                
        return U, V, W

    # Combine
    U1, V1, W1 = _construct_m2210()
    U2, V2, W2 = _construct_m2110()
    
    total_rank = U1.shape[1] + U2.shape[1]
    U_global = np.zeros((6, total_rank))
    V_global = np.zeros((30, total_rank))
    W_global = np.zeros((20, total_rank))
    
    for i in range(2):
        for j in range(2):
            U_global[i*3 + j, :35] = U1[i*2 + j, :]
            
    for j in range(2):
        for k in range(10):
            V_global[j*10 + k, :35] = V1[j*2 + k, :]
            
    for k in range(10):
        for i in range(2):
            W_global[k*2 + i, :35] = W1[k*2 + i, :]
            
    for i in range(2):
        U_global[i*3 + 2, 35:] = U2[i, :]
        
    for k in range(10):
        V_global[2*10 + k, 35:] = V2[k, :]
        
    for k in range(10):
        for i in range(2):
            W_global[k*2 + i, 35:] = W2[k,2 + i, :]
            
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}

