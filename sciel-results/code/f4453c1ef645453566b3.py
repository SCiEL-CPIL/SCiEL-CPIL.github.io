import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s wall-clock

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        """Check if U, V, W form an exact tensor decomposition."""
        try:
            n, m, p = DIMENSIONS
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Build the multiplication tensor T[i,j,k] = 1
            # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
            # For each column c, the term is u_c * v_c * w_c where:
            #   u_c[i*m+j] = U[i*m+j, c]
            #   v_c[j*p+k] = V[j*p+k, c]  
            #   w_c[k*n+i] = W[k*n+i, c]
            
            # For each column c, we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
            
            # Efficient check: for each (i,j), build a vector of size p from U and V, 
            # then check against W
            rank = len(U[0])
            
            # Build the tensor from the decomposition
            # T_decomp[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Vectorized approach
            # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
            U_reshaped = U_arr.reshape(n, m, rank)
            V_reshaped = V_arr.reshape(m, p, rank)
            W_reshaped = W_arr.reshape(p, n, rank)
            
            # For each i, j: 
            #   u_ij = U_reshaped[i, j, :]  (rank,)
            #   v_jk = V_reshaped[j, :, :]  (p, rank)
            #   For each k: v_j = V_reshaped[j, k, :]
            #   w_ki = W_reshaped[k, i, :]  (rank,)
            #   term_k = dot(u_ij, v_j) * ... no wait
            
            # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
            
            # Let's compute this for all (i,j,k)
            # T_decomp[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
            
            # We can do: for each i,j,k, compute the sum
            # Using broadcasting:
            # U_reshaped: (n, m, rank)
            # V_reshaped: (m, p, rank) -> need to align with (n, m, p)
            # W_reshaped: (p, n, rank) -> need to align with (n, m, p)
            
            # U_term: (n, m, p, rank) - U[i,j,c] broadcast over k
            # V_term: (n, m, p, rank) - V[j,k,c] broadcast over i
            # W_term: (n, m, p, rank) - W[k,i,c] broadcast over j
            
            U_bcast = U_reshaped[:, :, np.newaxis, :]  # (n, m, 1, rank)
            V_bcast = V_reshaped[np.newaxis, :, :, :]  # (1, m, p, rank)
            W_bcast = W_reshaped.T[np.newaxis, np.newaxis, :, :]  # (1, 1, p, n, rank) -> reshape
            # W_reshaped is (p, n, rank), so W_reshaped[k, i, c]
            # We want (n, m, p, rank) where index [i, j, k, c] = W_reshaped[k, i, c]
            W_bcast = W_reshaped[np.newaxis, :, :, :]  # (1, n, p, rank) -> transpose to get (n, p, rank) then broadcast
            W_bcast = W_reshaped.T[np.newaxis, :, :, :]  # (1, n, p, rank) where [i, k, c] = W_reshaped[k, i, c]
            # Now broadcast: (1, n, p, rank) to (n, m, p, rank)
            W_bcast = W_bcast[:, np.newaxis, :, :]  # (n, 1, p, rank)
            
            # Now multiply and sum over rank
            T_decomp = np.sum(U_bcast * V_bcast * W_bcast, axis=3)  # (n, m, p)
            
            # Should be all ones
            return np.allclose(T_decomp, 1.0, atol=1e-6)
        except Exception:
            return False

    def try_random_decomposition(rank, max_iters=5000):
        """Try to find a random decomposition of given rank."""
        for iter in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Random matrices with small integer entries
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            
            # Check
            if validate(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # First, try to find a better decomposition than schoolbook
    # Schoolbook rank is 48. Try to find lower ranks.
    
    # Known lower bounds: for 3x4x4, the rank is at least max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But the actual minimal rank for matrix multiplication is unknown for most cases.
    
    # Let's try some known structured decompositions
    
    # Try rank 16, 17, ... up to 47
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try structured approaches
    # One approach: use the fact that we can share some computations
    
    # For 3x4x4, let's try to use a Kronecker-like structure
    # Actually, let's just do random search for lower ranks
    
    for rank in range(16, best_rank):
        if time.time() > deadline - 20:
            break
        
        result = try_random_decomposition(rank, max_iters=200)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a lower rank, use it
    
    return best_result
# EVOLVE-BLOCK-END
