import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # safety margin before 150s limit

    # Known rank-46 decomposition using Strassen blocks
    rank_static = 46
    U_static = [[0] * rank_static for _ in range(n * m)]
    V_static = [[0] * rank_static for _ in range(m * p)]
    W_static = [[0] * rank_static for _ in range(p * n)]

    col = 0

    def add_strassen(ja, jb, ka, kb):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U_static[0*m+ja][col] = 1
        U_static[1*m+jb][col] = 1
        V_static[ja*p+ka][col] = 1
        V_static[jb*p+kb][col] = 1
        W_static[ka*n+0][col] = 1
        W_static[kb*n+1][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U_static[1*m+ja][col] = 1
        U_static[1*m+jb][col] = 1
        V_static[ja*p+ka][col] = 1
        W_static[ka*n+1][col] = 1
        W_static[kb*n+1][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U_static[0*m+ja][col] = 1
        V_static[ja*p+kb][col] = 1
        V_static[jb*p+kb][col] = -1
        W_static[kb*n+0][col] = 1
        W_static[kb*n+1][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U_static[1*m+jb][col] = 1
        V_static[jb*p+ka][col] = 1
        V_static[ja*p+ka][col] = -1
        W_static[ka*n+0][col] = 1
        W_static[ka*n+1][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U_static[0*m+ja][col] = 1
        U_static[0*m+jb][col] = 1
        V_static[jb*p+kb][col] = 1
        W_static[ka*n+0][col] = -1
        W_static[kb*n+0][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U_static[1*m+ja][col] = 1
        U_static[0*m+ja][col] = -1
        V_static[ja*p+ka][col] = 1
        V_static[ja*p+kb][col] = 1
        W_static[kb*n+1][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U_static[0*m+jb][col] = 1
        U_static[1*m+jb][col] = -1
        V_static[jb*p+ka][col] = 1
        V_static[jb*p+kb][col] = 1
        W_static[ka*n+0][col] = 1
        col += 1

    # 4 Strassen blocks
    add_strassen(0, 1, 0, 1)
    add_strassen(2, 3, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(2, 3, 2, 3)

    # Remaining: J in {0,1,2,3}, K=4
    for j in [0, 1, 2, 3]:
        k = 4
        U_static[0*m+j][col] = 1
        V_static[j*p+k][col] = 1
        W_static[k*n+0][col] = 1
        col += 1
        U_static[1*m+j][col] = 1
        V_static[j*p+k][col] = 1
        W_static[k*n+1][col] = 1
        col += 1

    # J=4, K in {0,1,2,3,4}
    for k in [0, 1, 2, 3, 4]:
        j = 4
        U_static[0*m+j][col] = 1
        V_static[j*p+k][col] = 1
        W_static[k*n+0][col] = 1
        col += 1
        U_static[1*m+j][col] = 1
        V_static[j*p+k][col] = 1
        W_static[k*n+1][col] = 1
        col += 1

    assert col == rank_static, f"Static decomposition rank mismatch: {col} != {rank_static}"

    # Verify the static decomposition
    def verify(U, V, W, rnk):
        # Check that sum_r U[i,j,r]*V[j,k,r]*W[k,l,r] = delta_{i,l} * A_{i,j} * B_{j,k}
        # More precisely, the tensor T[i,k,j] (with W indexed as k*n+i) must match
        # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] = 1 if... actually T[i,k,j] where W index is k*n+i
        # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 when (i,j,k) represents a valid multiplication term
        # i.e., equals 1 for all valid (i,j,k) since the tensor has all 1s
        # Wait - the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
        # No wait. Let me reconsider.
        # 
        # The matrix multiplication tensor for (n,m,p) has components indexed by (i,j,k)
        # where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 if the (i,k) entry of C = A*B comes from A[i,j]*B[j,k]
        # Actually T[i,j,k] = delta for the multiplication: C[i,k] += A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid indices.
        # 
        # The decomposition requires:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
        # for all i, j, k.
        
        err_max = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rnk):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    err = abs(s - 1.0)
                    if err > err_max:
                        err_max = err
        return err_max < 1e-9

    err_static = verify(U_static, V_static, W_static, rank_static)
    assert err_static < 1e-9, f"Static decomposition failed verification with err={err_static}"

    # Try to find a lower rank decomposition using tensor rank minimization
    # Strategy: Try rank 45, 44, ... by searching for valid decompositions
    
    def try_random_decomposition(rnk, n_iter=200):
        """Try to find a rank-rnk decomposition with random initialization and gradient-free search."""
        for it in range(n_iter):
            if time.time() > deadline:
                return None
            
            # Initialize with random half-integer entries in [-2, 2]
            U = [[0] * rnk for _ in range(n * m)]
            V = [[0] * rnk for _ in range(m * p)]
            W = [[0] * rnk for _ in range(p * n)]
            
            np_rng = np.random.default_rng(hash((rnk, it)) % (2**32))
            for i in range(n * m):
                for r in range(rnk):
                    U[i][r] = np_rng.integers(-2, 3) * 0.5
            for i in range(m * p):
                for r in range(rnk):
                    V[i][r] = np_rng.integers(-2, 3) * 0.5
            for i in range(p * n):
                for r in range(rnk):
                    W[i][r] = np_rng.integers(-2, 3) * 0.5
            
            # Compute error
            err = verify(U, V, W, rnk)
            if err < 1e-9:
                return (U, V, W)
        
        return None

    def try_structural_improvement(rnk):
        """Try to find a rank-rnk decomposition by modifying the known rank-46 structure."""
        # Convert to numpy for faster computation
        U_np = np.array(U_static, dtype=np.float64)[:, :rnk]
        V_np = np.array(V_static, dtype=np.float64)[:, :rnk]
        W_np = np.array(W_static, dtype=np.float64)[:, :rnk]
        
        # Try removing columns one at a time and checking if the remaining
        # columns can be adjusted to maintain the decomposition
        
        # Actually, let's try a different approach: search over small perturbations
        # of the known decomposition by combining columns
        
        # The rank-46 decomposition has structure. Try to see if any two columns
        # can be merged (i.e., if the tensor lies in a lower-dimensional subspace)
        
        # Compute the tensor T from the decomposition and check its rank
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For rank reduction, we need to find a linear combination of rank-1 terms
        # that still spans the tensor.
        
        # Try: can we express column r as a linear combination of other columns?
        # If U[:,r] = sum_{r' != r} a_{r'} U[:,r'], and similarly for V and W,
        # then we can eliminate that rank-1 term.
        
        # This is a complex optimization. Instead, try targeted search.
        
        # Try replacing the "remaining" terms (the 18 non-Strassen terms) with
        # a more efficient scheme.
        
        # The 18 remaining terms cover:
        # - 4 terms for (j in {0,1,2,3}, k=4, i=0)
        # - 4 terms for (j in {0,1,2,3}, k=4, i=1)  
        # - 5 terms for (j=4, k in {0,1,2,3,4}, i=0)
        # - 5 terms for (j=4, k in {0,1,2,3,4}, i=1)
        # Total: 18 terms, each rank 1.
        
        # Can we do better for j=4 or k=4?
        # For j=4, k in {0,1,2,3,4}: we have terms (i=0,k) and (i=1,k)
        # C[0,k] += A[0,4]*B[4,k] and C[1,k] += A[1,4]*B[4,k]
        # These are independent products, so we need 2*5=10 terms unless we can
        # share structure. But A[0,4] and A[1,4] are different entries, and 
        # B[4,k] varies with k.
        
        # Similarly for k=4, j in {0,1,2,3}: C[i,4] += A[i,j]*B[j,4]
        # These are also independent.
        
        # The Strassen blocks handle the 4x4 submatrix (j in {0,1,2,3}, k in {0,1,2,3})
        # with 4*7=28 terms instead of 4*4*2=32 terms (saves 4).
        
        # Total: 28 + 4*2 + 5*2 = 28+8+10 = 46. Correct.
        
        # To get rank 45, we'd need to save 1 more term.
        # Possible approaches:
        # 1. Find a rank-6 scheme for one of the 2x2 Strassen blocks
        #    (but 7 is minimal for 2x2x2 over reals by Strassen's lower bound for 3x3,
        #     and for 2x2x2 the minimum is 7 by a classic result)
        # 2. Find a way to merge one of the boundary terms with a Strassen term
        # 3. Use a non-strassen-based 4x4 scheme with rank < 28
        
        # For 2x2x2, rank 7 is proven minimal, so option 1 is out.
        # Option 2: Can we combine a boundary term with a Strassen term?
        # The boundary terms involve j=4 or k=4, while Strassen terms involve
        # j,k in {0,1,2,3}. They act on different entries, so direct merging
        # seems unlikely unless we use a clever linear combination.
        
        # Let's try a random search for rank 45
        result = try_random_decomposition(45, n_iter=500)
        if result:
            return result
        return None

    # Attempt to find lower rank decompositions
    for rnk in range(45, 40, -1):  # Try 45, 44, 43, 42, 41
        if time.time() > deadline - 10:
            break
        
        result = try_structural_improvement(rnk)
        if result:
            U, V, W = result
            # Convert to list format and verify once more
            err = verify(U, V, W, rnk)
            if err < 1e-9:
                return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline - 5:
            break
    
    # Return the static rank-46 decomposition
    return {"U": U_static, "V": V_static, "W": W_static}
