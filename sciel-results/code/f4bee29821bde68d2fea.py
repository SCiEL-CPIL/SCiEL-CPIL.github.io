"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook decomposition (rank 36) as fallback
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
    
    # Verify a decomposition is valid
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # And for all other index combos it should be 0
        # The matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        # Actually the tensor is T[i,j,k] where the multiplication tensor has T[i,j,k] = delta
        # The tensor for matmul: T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
        # and the decomposition satisfies sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Wait, let me re-read the trace convention.
        # W[k*n+i, r] - so W is indexed by (k, i) where k is the column of B / row of C, i is row of A / row of C
        # U[i*m+j, r] - U indexed by (i, j) - row i of A, col j of A
        # V[j*p+k, r] - V indexed by (j, k) - row j of B, col k of B
        # So the tensor is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for the matmul tensor
        # The matmul tensor T[i,j,k] = 1 (it's just 1 for all valid i,j,k in the contraction)
        # Actually no. The matrix multiplication tensor T is defined such that
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1
        # So the rank-r decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Build the tensor T (all ones)
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Known: rank(3x3x3) = 23, rank(2x2x2) = 7
    # For 3x3x4, we can try various approaches
    
    # Approach 1: Use the fact that 3x3x4 can be seen as two 3x3x2 blocks
    # Or use structured low-rank decompositions
    
    # Let's try a direct search approach with random initialization
    # and gradient-free optimization (evolutionary or random restarts)
    
    def try_random_decomposition(rank, n_restarts=50):
        """Try to find a rank-r decomposition using random search."""
        nm = n * m  # 9
        mp = m * p  # 12
        pn = p * n  # 12
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale to make the tensor entries approximately correct
            # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # Simple scaling: divide by rank^(1/3)
            scale = rank ** (1/3)
            U = U / scale
            V = V / scale
            W = W / scale
            
            # Use simple iterative refinement
            # For each (i,j,k), we need the product to sum to 1
            # This is a non-convex problem, so use random restarts with small perturbations
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-10:
                    break
                
                # Perturb randomly
                alpha = 0.1 * (1 - it / 200)
                U += alpha * np.random.randn(nm, rank)
                V += alpha * np.random.randn(mp, rank)
                W += alpha * np.random.randn(pn, rank)
            
            if best_err < 1e-8:
                # Found a valid decomposition!
                # Try to clean up to integers/half-integers
                U_clean = np.round(best_U * 2) / 2
                V_clean = np.round(best_V * 2) / 2
                W_clean = np.round(best_W * 2) / 2
                
                if verify(U_clean.tolist(), V_clean.tolist(), W_clean.tolist(), tol=1e-4):
                    return U_clean.tolist(), V_clean.tolist(), W_clean.tolist()
                
                # Try without rounding
                if verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), tol=1e-6):
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Approach 2: Structured decomposition using known techniques
    # For 3x3x4, try to exploit the rectangular structure
    
    # Let's try a specific construction:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this using rank-r terms
    
    # Known bound: rank(3x3x4) <= 24 (by viewing as two 3x3x2)
    # Actually, let's try to construct rank 24 explicitly
    
    def try_rank_24():
        """Try to construct a rank-24 decomposition."""
        rank = 24
        nm = n * m  # 9
        mp = m * p  # 12
        pn = p * n  # 12
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Idea: for each j, use a 3x4 multiplication which has rank 12
        # But we can share structure across j values
        # 
        # Alternative: use the "border rank" technique
        # For now, let's try a simple structured approach
        
        # Split into two groups of j: {0,1} and {2}
        # Group 1 (j=0,1): 2 * 3 * 4 = 24 terms -> too many
        # 
        # Let's try: for each (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        # We need 12 output entries, each is a sum of 3 products
        # Total "naive" rank is 36
        # 
        # Try to find linear dependencies
        
        # Use a different approach: parameterize and solve
        # Let's use a structured ansatz
        
        # Ansatz: U[i*m+j, r] depends on (i,j,r), V[j*p+k, r] on (j,k,r), W[k*n+i, r] on (k,i,r)
        # Try: for r in [0, 3*n*p) = [0, 12), use one set
        #       for r in [12, 24), use another set
        
        # Actually, let me try a completely different structured approach
        # Use the fact that we can write:
        # C = A * B
        # = [A0 A1 A2] * [B0; B1; B2] where Aj is 3x4, Bi is 3x4
        # No wait, A is 3x3, B is 3x4
        # A = [a00 a01 a02; a10 a11 a12; a20 a21 a22]
        # B = [b00 b01 b02 b03; b10 b11 b12 b13; b20 b21 b22 b23]
        
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # Let's try to find a rank-24 decomposition by solving a system
        # This is complex, so let's use numerical optimization
        
        return None
    
    # Approach 3: Use numpy to try to find a low-rank decomposition via optimization
    def optimize_decomposition(rank, max_iters=500, n_restarts=10):
        """Optimize a rank-r decomposition using gradient descent."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        best_result = None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradients numerically (or analytically)
                # Loss = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
                
                # Analytical gradient:
                # dL/dU[i,j,r] = 2 * sum_{k} (sum_{r'} U[i,j,r']V[j,k,r']W[k,i,r'] - 1) * V[j,k,r] * W[k,i,r]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Precompute S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                S = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                
                err = np.sum((S - 1)**2)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        best_result = (U.copy(), V.copy(), W.copy())
                        break
                
                # Compute gradients
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = S[i,j,k] - 1.0
                            for r in range(rank):
                                grad_U[i*m+j, r] += 2 * diff * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += 2 * diff * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += 2 * diff * U[i*m+j, r] * V[j*p+k, r]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if it % 50 == 0 and it > 0:
                    lr *= 0.9
            
            if best_result is not None and best_err < 1e-10:
                break
        
        if best_result is not None:
            U, V, W = best_result
            # Try to clean up
            for factor in [2, 1, 0.5]:
                U_c = np.round(U * factor) / factor
                V_c = np.round(V * factor) / factor
                W_c = np.round(W * factor) / factor
                if verify(U_c.tolist(), V_c.tolist(), W_c.tolist(), tol=1e-4):
                    return U_c.tolist(), V_c.tolist(), W_c.tolist()
            if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from low to high
    # Known: rank(3x3x3) = 23, so rank(3x3x4) >= 23
    # Upper bound: 36 (schoolbook), possibly lower
    
    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 34]:
        if time.time() > deadline - 10:
            break
        
        result = optimize_decomposition(rank, max_iters=300, n_restarts=5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Also try random search for this rank
        result = try_random_decomposition(rank, n_restarts=20)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook as fallback
    return schoolbook()
# EVOLVE-BLOCK-END
