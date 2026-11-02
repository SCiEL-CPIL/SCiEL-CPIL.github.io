"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    # Index conventions:
    # U: (n*m, rank) with row i*m+j
    # V: (m*p, rank) with row j*p+k
    # W: (p*n, rank) with row k*n+i
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor for 2x2 * 2x6)
    
    # The tensor is just the all-ones tensor of shape (2,2,6) since 
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k]=1 for all valid i,j,k.
    
    # Rank lower bound: max(n*p, m*p, n*m) = max(12, 12, 4) = 12
    # But for 2x2*2x6, we can potentially do better than schoolbook (24).
    
    # Strategy: Try to find low-rank decomposition using randomized linear algebra
    # and then verify. Start from known good decompositions and try to reduce.
    
    # First, let's try a structured approach.
    # For 2x2 * 2x6:
    # A is 2x2, B is 2x6. Product C is 2x6.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # We can group by k. For each k, C[:,k] = A * B[:,k] (2x2 times 2x1).
    # The 2x2 * 2x1 multiplication has tensor rank... let's think.
    # Actually, let's use the fact that this is a product of a 2x2 with a 2x6.
    
    # One approach: use the 2x2 matrix multiplication tensor (rank 7 via Strassen)
    # and extend to the 6 columns.
    
    # Actually, for 2x2 * 2x6, we can think of it as 6 independent 2x2*2x1 multiplications
    # but they share the same A matrix. This structure might allow rank reduction.
    
    # Let me try a direct randomized search for low rank.
    
    def verify(U, V, W):
        """Check if U,V,W correctly decompose the all-ones tensor of shape (n,m,p)."""
        rank = U.shape[1]
        # T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For efficiency, compute via matrix operations
        # Reshape approach:
        # For each c, outer product structure...
        # Simpler: just check a few random entries or all entries
        # All entries:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U_arr, V_arr, W_arr):
        """Vectorized verification using numpy."""
        rank = U_arr.shape[1]
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Build the tensor
        T = np.zeros((n, m, p))
        for c in range(rank):
            # U_c has shape (n*m,), reshape to (n,m)
            Uc = U_arr[:, c].reshape(n, m)
            Vc = V_arr[:, c].reshape(m, p)
            Wc = W_arr[:, c].reshape(p, n)
            # For each (i,j,k): Uc[i,j] * Vc[j,k] * Wc[k,i]
            # This is a 3D operation
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += Uc[i, j] * Vc[j, :] * Wc[:, i]
        return np.allclose(T, 1.0, atol=1e-9)
    
    def to_lists(U_arr, V_arr, W_arr):
        return {
            "U": U_arr.tolist(),
            "V": V_arr.tolist(),
            "W": W_arr.tolist()
        }
    
    # Schoolbook decomposition (rank 24) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = to_lists(U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*p, m*p, n*m) = max(12, 12, 4) = 12
    
    # Try ranks from lower bound up to schoolbook
    min_rank = max(n*p, m*p, n*m)
    
    for target_rank in range(min_rank, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Iterative refinement: minimize ||T_computed - 1||^2
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # Gradient descent on the entries
            
            success = False
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute current T
                T = np.zeros((n, m, p))
                for c in range(target_rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            T[i, j, :] += Uc[i, j] * Vc[j, :] * Wc[:, i]
                
                # Check convergence
                err = np.linalg.norm(T - 1.0)
                if err < 1e-8:
                    success = True
                    break
                
                # Compute gradients
                # dL/dU[i*m+j,c] = sum_k (T[i,j,k] - 1) * V[j*p+k,c] * W[k*n+i,c]
                # dL/dV[j*p+k,c] = sum_i (T[i,j,k] - 1) * U[i*m+j,c] * W[k*n+i,c]
                # dL/dW[k*n+i,c] = sum_j (T[i,j,k] - 1) * U[i*m+j,c] * V[j*p+k,c]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(target_rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                e = T[i, j, k] - 1.0
                                dU[i*m+j, c] += e * Vc[j, k] * Wc[k, i]
                                dV[j*p+k, c] += e * Uc[i, j] * Wc[k, i]
                                dW[k*n+i, c] += e * Uc[i, j] * Vc[j, k]
                
                # Adaptive step size
                lr = 0.01
                # Try to take a step
                U_new = U - lr * dU
                V_new = V - lr * dV
                W_new = W - lr * dW
                
                # Check if error decreased
                T_new = np.zeros((n, m, p))
                for c in range(target_rank):
                    Uc = U_new[:, c].reshape(n, m)
                    Vc = V_new[:, c].reshape(m, p)
                    Wc = W_new[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            T_new[i, j, :] += Uc[i, j] * Vc[j, :] * Wc[:, i]
                err_new = np.linalg.norm(T_new - 1.0)
                
                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    T = T_new
                    # Decay learning rate
                    lr *= 0.95
                else:
                    lr *= 0.5
                    if lr < 1e-6:
                        break
            
            if success:
                # Verify and round if possible
                if verify_fast(U, V, W):
                    best_rank = target_rank
                    best_result = to_lists(U, V, W)
                    break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
