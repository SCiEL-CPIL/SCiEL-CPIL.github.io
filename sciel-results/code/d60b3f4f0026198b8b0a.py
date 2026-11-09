"""Matrix multiplication 3,3,7 - rank minimization with slice-and-fiber linear algebra."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 150s limit

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, num_samples=30):
        """Verify tensor decomposition correctness using random integer matrices."""
        rng = random.Random(42)
        rank = len(U[0]) if U else 0
        for _ in range(num_samples):
            A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            # True product
            C_true = [[0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j] * B[j][k]
                    C_true[i][k] = s
            # Product via factorization: C[i,k] = sum_{j,c} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            C_fact = [[0] * p for _ in range(n)]
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        uc = U[i * m + j][c]
                        if uc == 0:
                            continue
                        for k in range(p):
                            vc = V[j * p + k][c]
                            if vc == 0:
                                continue
                            wc = W[k * n + i][c]
                            if wc == 0:
                                continue
                            C_fact[i][k] += uc * vc * wc
            for i in range(n):
                for k in range(p):
                    if C_true[i][k] != C_fact[i][k]:
                        return False
        return True

    def try_slice_fiber_decomposition(max_rank=63):
        """
        Attempt to find a lower-rank decomposition using slice-and-fiber linear algebra.
        
        Strategy: For each fixed value of one index, the multiplication tensor restricts
        to a matrix. We seek the minimum number of rank-1 slices whose combinations
        reconstruct all these fiber matrices.
        
        For 3x3x7, we fix the output index k (0..6). For each k, the fiber is a 3x3 matrix
        where entry (i,j) = delta_{i,j} * 1 (since C[i,k] = sum_j A[i,j]*B[j,k], and the
        tensor coefficient is 1 when the indices match appropriately).
        
        Actually, the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        Wait - that's not right. Let me reconsider.
        
        The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) where
        i in [0,n), j in [0,m), k in [0,p). The decomposition requires:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        For schoolbook, each (i,j,k) gets its own rank-1 term.
        
        The slice-and-fiber approach: Fix k. Then for each k, we need:
        For all (i,j): T[i,j,k] = 1 = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        This is a system where for fixed k, the matrix M_k[i,j] = 1 must be expressed
        as a sum of rank-1 matrices formed by outer products of vectors from U, V_k, W_k.
        
        Let's try a structured approach: assume W has a specific form based on the
        asymmetry (p=7 is larger than n=m=3).
        """
        if time.time() > deadline - 10:
            return None
        
        # Try to find a decomposition with rank < 63
        # Use a greedy approach: start with schoolbook and try to merge terms
        
        # Alternative: use the fact that for fixed i, the vectors V[:,j*p+k] and W[k*n+i]
        # must satisfy certain linear dependencies.
        
        # Let's try a parametric family: assume W has entries that are linear combinations
        # of a small number of basis vectors.
        
        # For now, try random search over structured decompositions
        rng = np.random.RandomState(42)
        
        best_rank = 63
        best_decomp = None
        
        # Try ranks from 50 down to 40 (time-limited)
        for target_rank in range(50, 39, -1):
            if time.time() > deadline - 20:
                break
            
            # Try a few random structured decompositions
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                
                # Generate random integer/half-integer entries
                scale = 2  # Use integers or half-integers
                U_rand = (rng.randint(0, scale, size=(n*m, target_rank)) - scale//2) / (scale/2 if scale % 2 == 0 else 1)
                V_rand = (rng.randint(0, scale, size=(m*p, target_rank)) - scale//2) / (scale/2 if scale % 2 == 0 else 1)
                W_rand = (rng.randint(0, scale, size=(p*n, target_rank)) - scale//2) / (scale/2 if scale % 2 == 0 else 1)
                
                # Check if this decomposition is valid
                # T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1.0
                            actual = 0.0
                            for r in range(target_rank):
                                actual += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                            if abs(actual - expected) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list format
                    U_list = [[float(x) for x in row] for row in U_rand]
                    V_list = [[float(x) for x in row] for row in V_rand]
                    W_list = [[float(x) for x in row] for row in W_rand]
                    decomp = {"U": U_list, "V": V_list, "W": W_list}
                    
                    if verify(decomp["U"], decomp["V"], decomp["W"], num_samples=10):
                        best_rank = target_rank
                        best_decomp = decomp
                        break  # Found a valid one at this rank
        
        if best_decomp is not None and best_rank < 63:
            return best_decomp
        
        return None

    # First, try the slice-and-fiber approach to find a potentially lower rank
    result = try_slice_fiber_decomposition()
    
    if result is None:
        # Fall back to schoolbook
        result = schoolbook()
    
    # Final verification
    if not verify(result["U"], result["V"], result["W"], num_samples=10):
        # Should not happen, but return schoolbook as safe fallback
        result = schoolbook()
    
    return result
# EVOLVE-BLOCK-END
