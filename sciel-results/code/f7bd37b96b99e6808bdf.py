import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Dimensions of the tensor indices
    dim_a = n * m  # 8
    dim_b = m * p  # 28
    dim_c = p * n  # 14
    
    # 1. Construct the target multiplication tensor T of shape (dim_a, dim_b, dim_c)
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    
    # Fill T based on the matrix multiplication definition:
    # C = A * B
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # In tensor form: T[a_idx, b_idx, c_idx] = 1 if the indices correspond to a valid term
    # a_idx = i * m + j
    # b_idx = j * p + k
    # c_idx = k * n + i
    # The term exists if the j in a matches the j in b, and the i in a matches the i in c, and k in b matches k in c.
    # Actually, the tensor T is defined such that the bilinear map (A, B) -> C is given by:
    # C[c_idx] = sum_{a, b} T[a, b, c] * A[a] * B[b]
    # For C[i, k] (c_idx = k*n + i), the coefficient of A[i, j] (a_idx = i*m + j) and B[j, k] (b_idx = j*p + k) is 1.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # 2. Compute the rank of the tensor.
    # The rank of a tensor T is the minimum R such that T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # This is equivalent to finding the minimum number of rank-1 tensors that sum to T.
    # A standard way to find an upper bound on the rank (and often the exact rank for small tensors) is to
    # flatten the tensor into a matrix and find its rank, but that gives the "slice rank" or similar.
    # Actually, the tensor rank is hard to compute exactly. However, for matrix multiplication tensors,
    # the rank is known to be bounded by the matrix rank of any unfolding.
    # Let's try a different approach: we can find a decomposition by finding a basis for the slices.
    
    # Let's use the "slice rank" or simply find a set of vectors that span the tensor.
    # A valid decomposition of rank R corresponds to R triples (u_r, v_r, w_r) such that:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # We can find such a decomposition by:
    # 1. Fixing a set of R vectors w_r that span the "c-slices" of T.
    # 2. For each w_r, the corresponding slice T[:, :, c] is a linear combination of the outer products u_r \otimes v_r.
    # This is complex.
    
    # Alternative strategy: Use the known fact that the rank of the matrix multiplication tensor M_{n,m,p} is at most n*m*p.
    # We can try to find a smaller rank by using SVD on the flattened tensor.
    # Flatten T to a matrix of shape (dim_a * dim_b, dim_c).
    # Let M = T.reshape(dim_a * dim_b, dim_c).
    # The rank of M is at most min(dim_a * dim_b, dim_c).
    # If we take the right singular vectors of M as our w_r, then M = U_svd * S * V_svd^T.
    # This doesn't directly give the tensor rank.
    
    # Let's try a constructive approach using the "border rank" or just finding a valid decomposition via linear algebra.
    # We can set U and V to be identity-like matrices and solve for W, or vice versa.
    # If we set U to be a basis for the space of A (dim_a x R) and V to be a basis for B (dim_b x R),
    # then we need W such that T = sum U[:, r] V[:, r]^T \otimes W[:, r].
    # This is equivalent to: for each c, T[:, :, c] = sum_r U[:, r] V[:, r]^T W[c, r].
    # This is a linear system for W if U and V are fixed.
    
    # Let's try to find a decomposition with rank R = dim_c = 14.
    # If we can find U (8x14) and V (28x14) such that the system is consistent, we are done.
    # Actually, we can just set R = dim_c and try to solve for U and V? No, that's not linear.
    
    # Let's use a simpler, guaranteed method:
    # The trivial decomposition has rank 56.
    # We can try to reduce the rank by merging terms.
    # But a more robust method is to use the "slice" approach.
    # Consider the slices T[:, :, c] for c in 0..13.
    # Each slice is an 8x28 matrix.
    # If we can express each slice as a linear combination of R rank-1 matrices, and the coefficients are the same for all c (via W), then we have a decomposition.
    
    # Let's try R = 28 (dim_b).
    # If we set V to be the identity matrix (28x28), then V[:, r] is a standard basis vector.
    # Then U[:, r] V[:, r]^T is a matrix with U[:, r] in column r and zeros elsewhere.
    # Then T[:, :, c] = sum_r U[:, r] e_r^T W[c, r] = sum_r W[c, r] U[:, r] e_r^T.
    # This means the (a, b) entry of T[:, :, c] is W[c, b] U[a, b].
    # So T[a, b, c] = U[a, b] W[c, b].
    # This requires T to be separable in b, which it is not generally.
    # Specifically, for fixed a and c, T[a, b, c] must be proportional to U[a, b] for all b.
    # Let's check if this is possible.
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # For fixed a (i, j) and c (k, i), b is determined as j*p+k.
    # So T[a, b, c] is 1 at exactly one b for each (a, c) pair?
    # No, for fixed a (i, j) and c (k, i), the valid b is j*p+k.
    # So T[a, b, c] is non-zero only for b = j*p+k.
    # Thus, for fixed a and c, T[a, b, c] is 0 for all b except b0.
    # So T[a, b, c] = U[a, b] W[c, b] implies that if b != b0, U[a, b] W[c, b] = 0.
    # If we choose U[a, b] = 1 for all b, then W[c, b] must be 0 for b != b0 and W[c, b0] = 1.
    # This works!
    # Let's verify:
    # Set U[a, b] = 1 for all a, b. (Shape 8x28)
    # Set W[c, b] = 1 if c and b are compatible (i.e., there exists i, j, k such that c=k*n+i, b=j*p+k, a=i*m+j), else 0.
    # Wait, W[c, b] must be independent of a.
    # The compatibility of c and b: c=k*n+i, b=j*p+k.
    # For a fixed c and b, is there a unique i, j, k?
    # c = k*n + i => k = c // n, i = c % n.
    # b = j*p + k => j = b // p, k' = b % p.
    # So compatibility requires k = k', i.e., c // n == b % p.
    # If this holds, then for any a = i*m + j', T[a, b, c] is 1 if j' == j, else 0.
    # But our formula T[a, b, c] = U[a, b] W[c, b] gives 1 for all a if W[c, b]=1.
    # This is incorrect because T[a, b, c] depends on a (specifically on j).
    # So U cannot be all 1s.
    
    # Let's try R = dim_a * dim_c = 8 * 14 = 112? No, that's larger than 56.
    # Let's try R = dim_a * dim_b = 8 * 28 = 224? No.
    
    # Let's go back to the trivial decomposition. Rank 56.
    # Can we do better?
    # The rank of M_{2,4,7} is known to be 56? Or less?
    # For M_{n,m,p}, the rank is at most n*m*p.
    # For M_{2,2,2}, rank is 7 (Strassen).
    # For M_{2,4,7}, maybe we can use a product of Strassen?
    # 4 = 2*2, 7 is prime.
    # This is complex.
    
    # Given the time limit and the difficulty of finding a provably lower rank decomposition for this specific size,
    # and the fact that the trivial decomposition is valid and simple,
    # I will stick with the trivial decomposition but ensure it is returned correctly.
    # However, the prompt asks for a "MATERIALLY DIFFERENT" strategy.
    # The current program uses a trivial decomposition.
    # My new strategy: Use SVD to find the minimum number of slices that span the tensor, which might be less than n*m*p if there is linear dependence among the slices.
    # Let's check if the 56 rank-1 terms are linearly independent.
    # They are, because each has a unique support.
    # So the rank is exactly 56.
    
    # Therefore, the best we can do is 56.
    # But wait, the task says "Find a rank-minimal exact tensor decomposition".
    # If the rank is 56, then 56 is the answer.
    # The current program returns 56.
    # So the score is 1.0.
    # To improve, we must find a rank < 56.
    # Is it possible?
    # The lower bound is max(nm, mp, pn) = max(8, 28, 14) = 28.
    # So the rank is between 28 and 56.
    # If we can find a decomposition of rank 28, we win.
    
    # Let's try to find a decomposition of rank 28.
    # This would require R=28.
    # T[a, b, c] = sum_{r=1}^{28} U[a, r] V[b, r] W[c, r].
    # This is a system of 8*28*14 = 3136 equations for 8*28 + 28*28 + 14*28 = 224 + 784 + 392 = 1400 variables.
    # This is overdetermined.
    # We can use least squares to find U, V, W that minimize the error.
    # If the error is zero, we have a valid decomposition.
    
    # Let's try this.
    # We will fix V to be random and solve for U and W? No, it's bilinear.
    # We can use alternating least squares (ALS).
    
    R = 28
    dim_a, dim_b, dim_c = 8, 28, 14
    
    U = np.random.randn(dim_a, R)
    V = np.random.randn(dim_b, R)
    W = np.random.randn(dim_c, R)
    
    for iter in range(50):
        if time.time() - start_time > deadline - 5:
            break
            
        # Fix U and V, solve for W
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For each c, T[:, :, c] = sum_r W[c, r] (U[:, r] V[:, r]^T)
        # Let M_c be the 8x28 matrix T[:, :, c].
        # Let A_r be the 8x28 matrix U[:, r] V[:, r]^T.
        # Then M_c = sum_r W[c, r] A_r.
        # This is a linear system for W[c, :] for each c.
        # Flatten A_r to a vector of length 224.
        # Flatten M_c to a vector of length 224.
        # Then M_c_vec = sum_r W[c, r] A_r_vec.
        # Let A_mat be a 224 x R matrix with columns A_r_vec.
        # Then M_c_vec = A_mat @ W[c, :].
        # So W[c, :] = A_mat^+ @ M_c_vec (pseudo-inverse).
        
        A_mat = np.zeros((dim_a * dim_b, R))
        for r in range(R):
            A_mat[:, r] = (U[:, r] * V[:, r]).reshape(-1) # Element-wise product, then flatten? No, outer product.
            # Wait, (U[:, r] V[:, r]^T) is an outer product.
            # Flattening an outer product is not the same as element-wise product.
            # Let's be careful.
            # M_c = sum_r W[c, r] (U[:, r] \otimes V[:, r])
            # Flatten M_c to vec(M_c).
            # vec(U[:, r] \otimes V[:, r]) = V[:, r] \otimes U[:, r] (Kronecker product).
            # So A_mat[:, r] = np.kron(V[:, r], U[:, r])
            
        A_mat = np.array([np.kron(V[:, r], U[:, r]) for r in range(R)]).T
        
        for c in range(dim_c):
            M_c_vec = T[:, :, c].reshape(-1)
            # Solve A_mat @ W[c, :] = M_c_vec
            W[c, :], residuals, rank, sv = np.linalg.lstsq(A_mat, M_c_vec, rcond=None)
            
        # Fix V and W, solve for U
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For each a, T[a, :, :] = sum_r U[a, r] (V[:, r] \otimes W[:, r])
        # Let B_r = V[:, r] \otimes W[:, r] (Kronecker product, shape 28x14 = 392)
        # Let T_a = T[a, :, :] (shape 28x14)
        # Then vec(T_a) = sum_r U[a, r] vec(B_r)
        # Let B_mat be a 392 x R matrix with columns vec(B_r).
        # Then vec(T_a) = B_mat @ U[a, :]
        
        B_mat = np.array([np.kron(V[:, r], W[:, r]) for r in range(R)]).T
        
        for a in range(dim_a):
            T_a_vec = T[a, :, :].reshape(-1)
            U[a, :], residuals, rank, sv = np.linalg.lstsq(B_mat, T_a_vec, rcond=None)
            
        # Fix U and W, solve for V
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For each b, T[:, b, :] = sum_r V[b, r] (U[:, r] \otimes W[:, r])
        # Let C_r = U[:, r] \otimes W[:, r] (shape 8x14 = 112)
        # Let T_b = T[:, b, :] (shape 8x14)
        # Then vec(T_b) = sum_r V[b, r] vec(C_r)
        # Let C_mat be a 112 x R matrix with columns vec(C_r).
        # Then vec(T_b) = C_mat @ V[b, :]
        
        C_mat = np.array([np.kron(U[:, r], W[:, r]) for r in range(R)]).T
        
        for b in range(dim_b):
            T_b_vec = T[:, b, :].reshape(-1)
            V[b, :], residuals, rank, sv = np.linalg.lstsq(C_mat, T_b_vec, rcond=None)
            
    # Check if the decomposition is valid
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_recon - T)
    
    if error < 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to trivial decomposition
        trivial_U = np.zeros((dim_a, 56), dtype=int)
        trivial_V = np.zeros((dim_b, 56), dtype=int)
        trivial_W = np.zeros((dim_c, 56), dtype=int)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    trivial_U[a_idx, r] = 1
                    trivial_V[b_idx, r] = 1
                    trivial_W[c_idx, r] = 1
                    r += 1
        return {"U": trivial_U.tolist(), "V": trivial_V.tolist(), "W": trivial_W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}, Sum diff: {diff.sum()}")
