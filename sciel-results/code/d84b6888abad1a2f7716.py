"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for trial in range(10):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_np[:, r], A.flatten())
                v_val = np.dot(V_np[:, r], B.flatten())
                # W index is k*n + i, so we need to map back to C[i, k]
                w_vec = W_np[:, r]
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += u_val * v_val * w_vec[idx]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    def verify_exact(U, V, W):
        """Verify exactness using rational arithmetic."""
        from fractions import Fraction
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        rank = len(U_frac[0]) if U_frac else 0
        
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    def search_decomposition(target_rank, seed=0):
        """Search for a decomposition with given rank using structured random approach."""
        if time.time() - start_time > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Strategy: Use the structure that for each (i,k), the sum over j should give C[i,k]
        # The tensor T[i,j,k] = 1 for all valid (i,j,k)
        
        # Parameterize U, V, W with small integer values
        # Try different patterns
        
        # Approach 1: Block structure
        # For each block of j indices, create shared terms
        
        best = None
        
        # Try to find a decomposition by solving linear equations
        # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is a system of nonlinear equations. Let's try a greedy approach.
        
        # Initialize with random small integers
        U = rng.randint(-2, 3, size=(n * m, target_rank))
        V = rng.randint(-2, 3, size=(m * p, target_rank))
        W = rng.randint(-2, 3, size=(p * n, target_rank))
        
        # For each (i,j,k), we need the sum to equal 1
        # Let's try to fix W and solve for U, V or vice versa
        
        # Alternative: Use the fact that we can have zero entries
        # Create a sparse structure
        
        # Try pattern: For each (i,k), have a dedicated term
        # But allow sharing across j
        
        # Let's try a more systematic approach
        # For each j, create m*p terms, but allow cross-j sharing
        
        # Actually, let's try to use the structure of matrix multiplication
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # So for each (i,k), we need sum_j (something) = A[i,j]*B[j,k]
        
        # The tensor decomposition requires:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        
        # Let's try to find a solution by fixing some variables
        
        # For simplicity, let's try W[k*n+i, r] = 1 for all k,i and some r
        # Then we need sum_r U[i*m+j, r] * V[j*p+k, r] = 1
        
        # This is a bilinear form
        
        # Let's try a different approach: use known decompositions
        # For 3x3x3, we know rank can be 23 (Laderman)
        # For 3x3x5, we can combine
        
        # Laderman's algorithm for 3x3x3
        # Let's implement it and extend
        
        if target_rank >= 45:
            # Schoolbook works
            return schoolbook()
        
        # Try to find a lower rank decomposition
        # Use random search with verification
        
        for attempt in range(100):
            if time.time() - start_time > deadline:
                break
            
            # Random initialization with structure
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # For each (i,k), create some terms
            for i in range(n):
                for k in range(p):
                    # Add some random terms
                    for r in range(min(5, target_rank)):
                        if rng.random() < 0.3:
                            # Random j
                            j = rng.randint(0, m)
                            U[i * m + j, r] = rng.choice([1, -1, 2, -2])
                            V[j * p + k, r] = rng.choice([1, -1, 2, -2])
                            W[k * n + i, r] = rng.choice([1, -1, 2, -2])
            
            # Verify
            if verify_exact(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known bounds: 3x3x3 has rank 23 (Laderman), so 3x3x5 should be at most 23 + 3*3*2 = 41
    # But we can do better with shared structure
    
    # Try ranks from low to high
    for rank in range(25, 45):
        if time.time() - start_time > deadline:
            break
        
        result = search_decomposition(rank, seed=rank)
        if result is not None:
            if verify(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
                break  # Found a valid lower rank
    
    return best_result
# EVOLVE-BLOCK-END
