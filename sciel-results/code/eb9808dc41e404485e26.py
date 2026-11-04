"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Subtensor slicing
    # For each of the 2 rows of the 2x5 matrix, we need to compute a 5x6 matrix-vector product
    # where the "vector" is the corresponding row of the 2x5 matrix (length 5)
    # and the "matrix" is the 5x6 matrix.
    # Actually, let's re-examine the dimensions.
    # A is 2x5, B is 5x6. C is 2x6.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T has indices (i, j, k) with i in [0,2), j in [0,5), k in [0,6).
    # T[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k].
    # This is the standard matrix multiplication tensor M_{2,5,6}.
    
    # The prompt suggests:
    # "fixing the two row indices of the 2x5 matrix to create two independent 5x6 matrix-vector multiplication tensors"
    # This means we slice the tensor by i.
    # For i=0: T[0, j, k] = 1. This is a 1x5x6 tensor, which is effectively a 5x6 matrix (the first row of B? No, it's the coefficients for the first row of C).
    # Actually, the tensor for i=0 is T_0[j, k] = 1. This is just a 5x6 matrix of ones? No.
    # The tensor decomposition is T = sum_r u_r v_r^T w_r^T.
    # If we fix i, we get a matrix T_i[j, k] = sum_r u_r[i] v_r[j] w_r[k].
    # For M_{2,5,6}, T_i[j, k] = 1 for all i, j, k? No.
    # T[i, j, k] = 1.
    # So T_i is a 5x6 matrix of all ones for both i=0 and i=1.
    # This doesn't seem right for a useful slicing strategy to reduce rank.
    
    # Let's reconsider the "subtensor slicing" method.
    # Maybe it means fixing j? No, j is the shared index.
    # Maybe it means fixing k?
    
    # Let's look at the "known rank-10 decomposition for 5x6 matrix-vector products".
    # A 5x6 matrix-vector product is C[j] = sum_k B[j, k] v[k] for j in [0,5), k in [0,6).
    # This is a 5x6 matrix times a 6-vector. The result is a 5-vector.
    # The tensor for this is M_{5,1,6} (if we view it as 5x1 matrix times 1x6 matrix? No).
    # It's a bilinear map from (5x6 matrix, 6-vector) to 5-vector.
    # The tensor T[j, k, l] = 1 where j is row of B, k is col of B (and index of v), l is index of output?
    # No, the standard matrix multiplication tensor M_{n,m,p} is for (n x m) * (m x p) -> (n x p).
    # So M_{5,1,6} would be (5x1) * (1x6) -> (5x6). That's not matrix-vector.
    # Matrix-vector is (5x6) * (6x1) -> (5x1). This is M_{5,6,1}.
    # Rank of M_{5,6,1}?
    # It is known that the rank of M_{n,m,1} is n*m. But wait, M_{n,m,1} is just n*m scalars.
    # The bilinear complexity of multiplying an n x m matrix by a vector of length m is n*m.
    # So rank-10 for 5x6 matrix-vector? That seems too low. 5*6=30.
    # Unless the prompt means something else.
    # "known rank-10 decomposition for 5 x 6 matrix-vector products"
    # Maybe it means multiplying a 5x6 matrix by a vector of length 5? No, dimensions must match.
    # Maybe it means a specific structured decomposition?
    
    # Let's look at the current best score: 1.111 with rank 54.
    # Schoolbook is 2*5*6 = 60.
    # 54 is 60 * 0.9.
    # The prompt claims a candidate rank-20 decomposition.
    # If the rank is 20, the score would be 60/20 = 3.0? No, the score is 1.0 for schoolbook.
    # "score is the rank" -> Lower is better.
    # "search score is seed/value, so 1.0 equals the seed program"
    # Seed value is 60.
    # If rank is 54, value is 54. Score = 60/54 = 1.111.
    # If rank is 20, value is 20. Score = 60/20 = 3.0.
    # This would be a huge improvement.
    
    # How to get rank 20?
    # The tensor is M_{2,5,6}.
    # If we can decompose it into 20 rank-1 terms.
    # One way: Use Kronecker product structure?
    # M_{2,5,6} = M_{2,1,1} x M_{1,5,6}? No.
    # M_{2,5,6} can be viewed as 2 copies of M_{1,5,6} summed?
    # T[i, j, k] = 1.
    # If we write T = sum_{r=1}^{20} u_r v_r w_r.
    #
    # Let's try to construct a rank-20 decomposition manually or via search.
    # Given the time limit and the risk of a wrong manual derivation, I will implement a simple search or use a known low-rank structure if available.
    # However, the prompt says "use the known rank-10 decomposition for 5x6 matrix-vector products".
    # If M_{5,6,1} has rank 10, then M_{2,5,6} might have rank 2*10 = 20.
    # How?
    # M_{2,5,6}[i, j, k] = 1.
    # This is not a Kronecker product of M_{2,1,1} and M_{1,5,6}.
    # But notice that for each i, the slice T[i, :, :] is the same matrix (all ones 5x6).
    # So T = e_0 * A + e_1 * A, where A is the 5x6 all-ones matrix.
    # This doesn't help with rank reduction directly in the standard tensor form.
    
    # Let's assume the "rank-10" claim is for a different tensor or I am misinterpreting it.
    # I will stick to a robust approach:
    # 1. Try to verify if a rank-20 decomposition exists via random search or structured guess.
    # 2. If not, fall back to the known rank-54 Strassen-like decomposition or schoolbook.
    
    # Actually, let's look at the "Structured Strassen decomposition" mentioned in the knowledge.
    # It reduced rank from 60 to 54.
    # I can implement that.
    # But the prompt explicitly asks to try the "subtensor slicing" method to get rank 20.
    # I will write a function that attempts to find a rank-20 decomposition using a randomized search with a time limit.
    # If it fails, it returns the schoolbook or a known better one.
    
    n, m, p = DIMENSIONS
    target_rank = 20
    deadline = time.time() + 140  # 10s safety margin
    
    # Define the target tensor T[i, j, k] = 1
    # We want to find U (n*m x R), V (m*p x R), W (p*n x R) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a 3-tensor.
    # We can try to find a sparse solution.
    # Let's use a simple iterative method: 
    # Initialize U, V, W with random small values.
    # Perform gradient descent on the error.
    # But this is hard to implement quickly and reliably.
    
    # Alternative: 
    # If the "rank-10" for 5x6 matrix-vector is true, it likely refers to a specific structured decomposition.
    # I will search for a rank-20 decomposition using a simple randomized algorithm with early stopping.
    # If I can't find it in 20s, I'll return the schoolbook decomposition (rank 60) to ensure validity, 
    # or better, the rank-54 one if I can code it quickly.
    # Actually, coding a correct rank-54 Strassen for 2x5x6 is complex.
    # Let's just return the schoolbook decomposition to be safe, unless I can quickly verify a lower rank.
    
    # Wait, the prompt says "This is a search instruction... Treat these as hypotheses to test."
    # I will test the hypothesis that rank 20 is achievable.
    
    def try_find_decomposition(rank, time_limit=10):
        start_time = time.time()
        # Random initialization
        np.random.seed(42)
        U = np.random.rand(n * m, rank)
        V = np.random.rand(m * p, rank)
        W = np.random.rand(p * n, rank)
        
        # Target tensor T
        T = np.ones((n, m, p))
        
        # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k]
        
        # Gradient descent step
        lr = 0.01
        for _ in range(1000):
            if time.time() > start_time + time_limit:
                return None
            
            # Compute current tensor estimate
            # This is O(n*m*p*rank)
            T_est = np.zeros((n, m, p))
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                # T_est[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                # This is not a standard einsum, need to be careful with indices.
                # u[i,j] * v[j,k] * w[k,i]
                # For fixed r:
                # We can do:
                # tmp = u @ v  -> (n, p)
                # Then T_est[i,k] += tmp[i,k] * w[k,i]
                tmp = u @ v
                # w is (p, n). We need w[k, i].
                # So we need to multiply element-wise by w.T?
                # tmp[i,k] * w[k,i] = tmp[i,k] * w.T[i,k]
                T_est += tmp * w.T
            
            error = T_est - T
            loss = np.sum(error**2)
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                
                # Gradient of sum_{i,j,k} (sum_r u_r[i,j] v_r[j,k] w_r[k,i] - T[i,j,k])^2 w.r.t u_r[i,j]
                # = 2 * sum_{i,j,k} (error[i,j,k]) * v_r[j,k] * w_r[k,i] * (delta_{ii'} delta_{jj'})
                # For u[i,j]:
                # grad_u[i,j] = 2 * sum_{k} error[i,j,k] * v[j,k] * w[k,i]
                
                grad_u = np.zeros((n, m))
                grad_v = np.zeros((m, p))
                grad_w = np.zeros((p, n))
                
                for i in range(n):
                    for j in range(m):
                        # sum_k error[i,j,k] * v[j,k] * w[k,i]
                        grad_u[i, j] = np.sum(error[i, j, :] * v[j, :] * w[:, i])
                        
                for j in range(m):
                    for k in range(p):
                        # sum_i error[i,j,k] * u[i,j] * w[k,i]
                        grad_v[j, k] = np.sum(error[:, j, k] * u[:, j] * w[k, :])
                        
                for k in range(p):
                    for i in range(n):
                        # sum_j error[i,j,k] * u[i,j] * v[j,k]
                        grad_w[k, i] = np.sum(error[i, :, k] * u[i, :] * v[:, k])
                        
                dU[r] = 2 * grad_u.flatten()
                dV[r] = 2 * grad_v.flatten()
                dW[r] = 2 * grad_w.flatten()
                
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < 1e-8:
                return U, V, W
                
        return None
    
    U, V, W = try_find_decomposition(target_rank, time_limit=20)
    
    if U is not None:
        # Convert to lists of integers/half-integers if possible, else float
        # The task says "integer/half-integer entries".
        # If the solution is close to integers, round them.
        U_list = [[int(round(val * 2)) / 2 for val in row] for row in U]
        V_list = [[int(round(val * 2)) / 2 for val in row] for row in V]
        W_list = [[int(round(val * 2)) / 2 for val in row] for row in W]
        
        # Verify
        T_est = np.zeros((n, m, p))
        for r in range(target_rank):
            u = np.array(U_list[:, r]).reshape(n, m)
            v = np.array(V_list[:, r]).reshape(m, p)
            w = np.array(W_list[:, r]).reshape(p, n)
            tmp = u @ v
            T_est += tmp * w.T
        
        if np.allclose(T_est, np.ones((n, m, p)), atol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Rounding failed, return floats? Task says integer/half-integer.
            # If we can't satisfy, better to return a valid integer solution.
            pass
            
    # Fallback: Schoolbook decomposition
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
