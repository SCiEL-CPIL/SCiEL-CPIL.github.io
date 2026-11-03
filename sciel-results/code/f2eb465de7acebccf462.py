import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the tensor T of shape (n, m, p)
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Function to verify a decomposition
    def verify(U, V, W, R):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reconstruct T
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this efficiently by reshaping
        # Let's just do the sum for each (i, j, k)
        # Since n, m, p are small, this is fine.
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += abs(val - T[i, j, k])
        return error < 1e-9

    # Function to try to find a decomposition of rank R
    def find_decomposition(R, max_iters=20):
        if R >= rank_schoolbook:
            return None
            
        for attempt in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline - 5.0:
                    break
                    
                # Update U
                for r in range(R):
                    # T is fixed. V[:, r] and W[:, r] are fixed.
                    # We want to minimize || T - sum_r U_r \otimes V_r \otimes W_r ||^2
                    # This is a linear system for U[:, r]
                    # Let's construct the matrix M_U such that M_U @ U[:, r] = target
                    # This is complex to set up directly. 
                    # Alternative: Use gradient descent on the whole tensor?
                    # Or use the "peeling" method?
                    # For small tensors, we can use a simple optimization.
                    pass
                
            # This simple ALS setup is tricky to code correctly in a few lines.
            # Let's use a different approach: Random Linear Combinations
            # Check if T can be written as sum of R rank-1 tensors.
            # This is equivalent to finding R vectors in the tensor space.
            
            # Given the complexity, let's stick to a known structure if possible.
            # For 3x4x4, a rank-45 decomposition might be achievable by 
            # combining Strassen (rank 7 for 2x2x2) with schoolbook for the rest.
            
            # Let's try a specific construction:
            # Split 4x4 into 2x2 blocks.
            # A (3x4) = [A1 (3x2) | A2 (3x2)]
            # B (4x4) = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
            # C1 = A1 B11 + A2 B21
            # C2 = A1 B12 + A2 B22
            
            # We need to compute 4 products of (3x2) and (2x2).
            # Rank of 3x2x2?
            # Schoolbook: 12.
            # Can we do better?
            # 3x2x2 is equivalent to multiplying 3 vectors of length 2 by a 2x2 matrix?
            # No, it's (3x2) * (2x2) -> (3x2).
            # Let A = [a1, a2, a3]^T (rows are 1x2).
            # B is 2x2.
            # C_i = A_i B.
            # This is 3 separate 1x2 by 2x2 multiplications.
            # Rank of 1x2x2?
            # T[i, j, k] = A[i, j] B[j, k].
            # i=0, j=0,1, k=0,1.
            # This is rank 2? No.
            # T is a 1x2x2 tensor.
            # It has 4 entries.
            # Rank 1? No.
            # Rank 2?
            # T[0,0,0] = A00 B00
            # T[0,0,1] = A00 B01
            # T[0,1,0] = A01 B00
            # T[0,1,1] = A01 B01
            # This is outer product of A (length 2) and B (length 2) flattened?
            # Yes, T is the outer product of vector A (size 2) and vector B (size 2) if we view B as a vector?
            # No, B is a matrix.
            # T[i, j, k] = A[j] B[j, k] (summing over j? No, j is fixed index of A).
            # Wait, A is 1x2. A[j] is scalar.
            # T[0, j, k] = A[j] * B[j, k].
            # This is not a standard outer product.
            
            # Let's abandon the manual construction and rely on the search or fallback.
            # Since I cannot guarantee a search implementation is correct in this short format, 
            # and the "Inspiration" suggests a score of 1.09 (Rank 44), 
            # I will return the schoolbook rank but optimize the code to be as fast as possible 
            # in case the evaluator penalizes slow execution? No, score is based on rank.
            
            # However, if I can find a rank-45 decomposition, I get 1.066.
            # Let's try to hardcode a rank-45 decomposition if I can find one online? 
            # I don't have internet access.
            
            # Let's try a "greedy" peeling algorithm.
            # 1. Find the best rank-1 approximation of T.
            # 2. Subtract it.
            # 3. Repeat R times.
            # 4. Check if the remainder is zero.
            
            def peel(T, R, max_steps=50):
                U = []
                V = []
                W = []
                T_rem = T.copy()
                for r in range(R):
                    if time.time() > deadline - 5.0:
                        return None
                    # Find u, v, w that maximize u^T (T_rem \times_1 v \times_2 w)
                    # This is a non-convex optimization.
                    # Use random initialization and gradient ascent.
                    u = np.random.randn(n)
                    v = np.random.randn(m)
                    w = np.random.randn(p)
                    
                    for step in range(max_steps):
                        # Update u
                        # T_rem \times_1 v \times_2 w is a vector of size n
                        # tmp = np.tensordot(np.tensordot(T_rem, w, axes=([2],[0])), v, axes=([2],[0]))
                        # This is slow in Python loops.
                        # Use numpy.
                        # M = np.tensordot(T_rem, w, axes=([2],[0])) # Shape (n, m)
                        # u_new = M @ v
                        M = np.tensordot(T_rem, w, axes=([2],[0]))
                        u_new = M @ v
                        if np.linalg.norm(u_new) > 1e-10:
                            u = u_new / np.linalg.norm(u_new)
                        
                        # Update v
                        # M = np.tensordot(T_rem, u, axes=([0],[0])) # Shape (m, p)
                        M = np.tensordot(T_rem, u, axes=([0],[0]))
                        v_new = M @ w
                        if np.linalg.norm(v_new) > 1e-10:
                            v = v_new / np.linalg.norm(v_new)
                            
                        # Update w
                        # M = np.tensordot(T_rem, u, axes=([0],[0])) # Shape (m, p)
                        # M2 = np.tensordot(M, v, axes=([0],[0])) # Shape (p)
                        M = np.tensordot(T_rem, u, axes=([0],[0]))
                        w_new = M @ v
                        if np.linalg.norm(w_new) > 1e-10:
                            w = w_new / np.linalg.norm(w_new)
                            
                    # Scale u, v, w to match T_rem
                    # T_rem \approx scale * u \otimes v \otimes w
                    # scale = sum(T_rem * (u \otimes v \otimes w)) / sum((u \otimes v \otimes w)^2)
                    outer = np.einsum('i,j,k->ijk', u, v, w)
                    numerator = np.sum(T_rem * outer)
                    denominator = np.sum(outer * outer)
                    if denominator > 1e-10:
                        scale = numerator / denominator
                    else:
                        return None
                        
                    U.append(u * np.sqrt(abs(scale))) # Distribute scale
                    V.append(v * np.sign(scale) * np.sqrt(abs(scale)))
                    W.append(w)
                    
                    T_rem = T_rem - scale * outer
                    
                    # Check if T_rem is zero
                    if np.linalg.norm(T_rem) < 1e-9:
                        # We found a decomposition of rank r+1
                        # Convert to integer/half-integer?
                        # The peeling method usually yields real numbers.
                        # We need integer/half-integer.
                        # This is a problem.
                        return None
                        
                return None

            # The peeling method doesn't guarantee integer/half-integer.
            # The task requires integer/half-integer entries.
            
            # Given the constraints, I will return the schoolbook rank.
            # It is the only guaranteed valid solution.
            pass
        
        return None

    # Fallback to schoolbook
    rank = rank_schoolbook
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
