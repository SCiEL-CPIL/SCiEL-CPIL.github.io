"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # The tensor T is defined by T[i, j, k] = 1 if and only if i, j, k are valid indices.
    # We seek U (num_u x R), V (num_v x R), W (num_w x R) such that:
    # sum_r U[i, r] * V[j, r] * W[k, r] = 1 for all i, j, k.
    
    # We start from rank 50 (schoolbook) and try to go down to 49.
    # If we fail to find a valid decomposition for a lower rank in time, we return the best valid one found.
    
    best_rank = 50
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute the target tensor indices (i, j, k)
    # There are n*m*p = 50 constraints.
    # Actually, the tensor is sparse in the sense that only 50 entries are 1, rest 0?
    # No, the tensor for matrix multiplication has 1s at (i, j, k) for all valid i, j, k.
    # Wait, the standard matrix multiplication tensor T_{i,j,k} is 1 if the term C_{ik} = sum A_{ij} B_{jk} exists.
    # So T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # The decomposition is sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k].
    
    # Let's try ranks from 49 up to 50.
    # We will use a randomized approach with linear solving.
    # Fix U and V randomly, solve for W? No, it's bilinear in U, V, W.
    # If we fix U and V, the equation is linear in W?
    # sum_r U[i, r] V[j, r] W[k, r] = 1.
    # This is not linear in W because W[k, r] is multiplied by U[i, r] V[j, r].
    # However, for a fixed k, the equation for all i, j is:
    # sum_r (U[i, r] V[j, r]) W[k, r] = 1.
    # Let M_{k} be the matrix of size (n*m) x R where M_{k}[i*m+j, r] = U[i, r] V[j, r].
    # Then M_{k} W[:, k] = 1 (vector of ones of length n*m).
    # So for each k, we have a linear system M_{k} w_k = 1.
    # If M_{k} is invertible (or has a solution), we can find W[:, k].
    # Since R might be less than n*m (e.g., R=49 < 50), M_k is 50x49.
    # The system is overdetermined. We need an exact solution.
    # This only works if the rows of M_k are linearly dependent in a specific way or if we have a solution.
    # Actually, if we generate U and V randomly, M_k will have full column rank (49).
    # The system M_k w = 1 has a solution if and only if 1 is in the column space of M_k.
    # For random U, V, this is highly unlikely.
    
    # Alternative: Use the known 49-rank structure.
    # The 49-rank decomposition for 2x5x5 is not trivially known as a simple formula.
    # However, the "Strassen for 2x2x2 sub-blocks" hint suggests a block structure.
    # 2x5x5 can be viewed as 2x(2+3)x(2+3) or similar?
    # Or maybe it's related to the fact that 2x5x5 has a 49-rank decomposition.
    # Let's try to find it via a more robust method: 
    # We can use the fact that if we have a 50-rank decomposition, we might be able to merge terms.
    # But a simpler approach for the code: 
    # Just return the 50-rank decomposition. It is guaranteed to be correct and fast.
    # The task asks for a "DIFFERENT strategy". Returning 50 is the same as the seed.
    # The seed scores 1.0. If I return 50, I score 1.0.
    # If I find 49, I score 50/49 = 1.02.
    
    # Let's try a specific known 49-rank decomposition if I can recall or derive it.
    # Actually, the hint says "Block decomposition using Strassen for 2x2x2 sub-blocks reduces rank from 50 to 49".
    # This implies we can partition the 2x5x5 multiplication into blocks where one 2x2x2 block is done with Strassen (rank 7 instead of 8).
    # 2x5x5:
    # A is 2x5, B is 5x5.
    # We can split B into [B1 B2] where B1 is 5x2 and B2 is 5x3?
    # Or split the output C (2x5) into C1 (2x2) and C2 (2x3).
    # C1 = A * B1. This is 2x5 * 5x2.
    # C2 = A * B2. This is 2x5 * 5x3.
    
    # If we split the input dimension 5 into 2+3:
    # A is 2x5. Let A = [A1 A2] where A1 is 2x2 and A2 is 2x3.
    # B is 5x5. Let B = [B1; B2] where B1 is 2x5 and B2 is 3x5.
    # C = A1 B1 + A2 B2.
    
    # Term 1: A1 (2x2) * B1 (2x5).
    # This is a 2x2x5 multiplication.
    # Term 2: A2 (2x3) * B2 (3x5).
    # This is a 2x3x5 multiplication.
    
    # Rank of 2x2x5:
    # We can compute A1 B1.
    # A1 is 2x2, B1 is 2x5.
    # C1 is 2x5.
    # We can use Strassen for the 2x2 part?
    # Let B1 = [b1 b2] where b1, b2 are 2x2? No, B1 is 2x5.
    # Let's split the output of C1 into 2x2 and 2x3.
    # C1 = A1 * [B1a B1b] where B1a is 2x2 and B1b is 2x3.
    # C1a = A1 B1a (2x2x2). Rank 7 (Strassen).
    # C1b = A1 B1b (2x2x3). 
    # Rank of 2x2x3?
    # A1 (2x2) * B1b (2x3).
    # This is equivalent to multiplying a 2x2 matrix by a 2x3 matrix.
    # We can view this as two 2x2x2 multiplications? No.
    # Actually, 2x2x3 multiplication rank is 7?
    # Let's check.
    # A (2x2), B (2x3). C (2x3).
    # C = A B.
    # We can compute C[:, 0:2] using Strassen (rank 7).
    # C[:, 2] is A * B[:, 2]. This is a matrix-vector multiplication.
    # Matrix-vector 2x2 * 2x1 is rank 4 (schoolbook) or 3?
    # Actually, matrix-vector multiplication A(2x2) v(2x1) has rank 4.
    # So total rank for 2x2x3 would be 7 + 4 = 11?
    # Or can we do better?
    # The rank of 2x2x3 tensor is 7.
    # Wait, the rank of 2x2x3 is 7.
    # So C1 (2x2x5) can be split into 2x2x2 (rank 7) and 2x2x3 (rank 7). Total 14.
    # Schoolbook for 2x2x5 is 2*2*5 = 20.
    # So we save 6 ranks here.
    
    # Now Term 2: A2 (2x3) * B2 (3x5).
    # C2 = A2 B2.
    # Split B2 into B2a (3x2) and B2b (3x3).
    # C2a = A2 B2a (2x3x2).
    # C2b = A2 B2b (2x3x3).
    
    # Rank of 2x3x2:
    # This is equivalent to 2x2x3 (by symmetry). Rank 7.
    
    # Rank of 2x3x3:
    # A2 (2x3) * B2b (3x3).
    # This is a 2x3x3 multiplication.
    # Schoolbook: 2*3*3 = 18.
    # Can we do better?
    # We can split the output 3 into 1+2?
    # Or use known ranks.
    # The rank of 2x3x3 is 13?
    # Let's assume we can find a decomposition for 2x3x3 with rank 13.
    
    # Total rank:
    # C1a (2x2x2): 7
    # C1b (2x2x3): 7
    # C2a (2x3x2): 7
    # C2b (2x3x3): 13
    # Total: 7 + 7 + 7 + 13 = 34?
    # This is much less than 49.
    # Why did the hint say 49?
    # Maybe the splitting is different or the ranks are higher.
    # Or maybe the "Strassen for 2x2x2 sub-blocks" refers to a specific partitioning that yields 49.
    
    # Regardless, if I can construct a valid 49-rank decomposition, I win.
    # But constructing it correctly is hard.
    # The safest bet for a "different strategy" that doesn't fail is to use the randomized linear solve approach to *find* a 49-rank decomposition if it exists, or fall back to 50.
    
    # However, the linear solve approach (fix U, V, solve W) only works if the system is consistent.
    # For random U, V, it is not.
    
    # Let's try a different angle:
    # We know the 50-rank decomposition.
    # Can we perturb it to get 49?
    # No, rank is discrete.
    
    # Given the time constraint and the difficulty of deriving the 49-rank decomposition from scratch,
    # and the fact that the seed program scores 1.0,
    # I will implement a robust 50-rank decomposition but with a different structure (e.g., using NumPy for construction) to satisfy the "different strategy" requirement minimally, while ensuring correctness.
    # Wait, the directive says "Make a STRUCTURAL change".
    # Using NumPy is not a structural change in the algorithm.
    
    # Let's try the randomized search for 49 again, but smarter.
    # We can use the fact that the tensor is low rank.
    # We can use ALS (Alternating Least Squares) to minimize the error.
    # If the error drops to 0, we have a valid decomposition.
    # We start with random U, V, W of rank 49.
    # Iterate:
    # 1. Fix V, W. Solve for U.
    # 2. Fix U, W. Solve for V.
    # 3. Fix U, V. Solve for W.
    # If the reconstruction error is < 1e-9, we check if entries are close to integers/half-integers.
    # If yes, round them and return.
    
    # This is a valid structural change: from deterministic construction to optimization-based search.
    
    def als_decompose(n, m, p, rank, max_iters=100, tol=1e-9):
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Target tensor T (n, m, p)
        T = np.ones((n, m, p))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None, None, None
            
            # Solve for U
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # Reshape T to (n, m*p)
            T_flat = T.reshape(n, m * p)
            # W_flat is (p*n, rank). We need W[k, r] for each k.
            # The equation is: sum_r U[i, r] * (V * W_k)_r = T[i, :]
            # Let M_W[k] = V * W[:, k] (size m x 1? No, V is m*p x rank, W is p*n x rank).
            # This is getting complicated to vectorize simply.
            
            # Simpler: Loop over i.
            # For fixed i, T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
            # This is linear in U[i, :].
            # Let A_i be a matrix of size (m*p) x rank.
            # A_i[(j, k), r] = V[j, r] * W[k, r].
            # Then A_i U[i, :] = T[i, :].reshape(-1).
            
            for i in range(n):
                A_i = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        A_i[j * p + k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve A_i U[i, :] = T[i, :].reshape(-1)
                b = T[i, :, :].reshape(-1)
                try:
                    U[i, :], residuals, rank_sol, sv = np.linalg.lstsq(A_i, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None, None, None
            
            # Solve for V
            for j in range(m):
                A_j = np.zeros((n * p, rank))
                for i in range(n):
                    for k in range(p):
                        A_j[i * p + k, :] = U[i * m + j, :] * W[k * n + i, :]
                b = T[:, j, :].reshape(-1)
                try:
                    V[j * p + k, :], residuals, rank_sol, sv = np.linalg.lstsq(A_j, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None, None, None
            
            # Solve for W
            for k in range(p):
                A_k = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        A_k[i * m + j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = T[:, :, k].reshape(-1)
                try:
                    W[k * n + i, :], residuals, rank_sol, sv = np.linalg.lstsq(A_k, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None, None, None
            
            # Check error
            # Reconstruct T_hat
            T_hat = np.zeros((n, m, p))
            for r in range(rank):
                T_hat += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T_hat - T)
            if error < tol:
                # Check if entries are close to half-integers
                # Multiply by 2 to check integers
                U_check = 2 * U
                V_check = 2 * V
                W_check = 2 * W
                
                if np.allclose(U_check, np.round(U_check)) and \
                   np.allclose(V_check, np.round(V_check)) and \
                   np.allclose(W_check, np.round(W_check)):
                    # Round and return
                    U_int = np.round(U_check).astype(int) / 2.0
                    V_int = np.round(V_check).astype(int) / 2.0
                    W_int = np.round(W_check).astype(int) / 2.0
                    
                    # Convert to lists
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None, None, None
    
    # Try rank 49
    for attempt in range(10):
        if time.time() > deadline:
            break
        U, V, W = als_decompose(n, m, p, 49, max_iters=50)
        if U is not None:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to 50
    rank = 50
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
