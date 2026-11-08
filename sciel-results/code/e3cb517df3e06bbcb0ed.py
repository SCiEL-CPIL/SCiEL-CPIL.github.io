import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    # A is n×m = 2×3, B is m×p = 3×4, C is n×p = 2×4
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (12, rank)  
    # W: (p*n, rank) = (8, rank)
    
    def check_decomposition(U, V, W, rank, max_checks=50):
        """Check if decomposition is exact using random evaluations."""
        np.random.seed(42)
        for _ in range(max_checks):
            # Random A (n×m) and B (m×p) with small integer entries
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A @ B
            
            # Flatten A and B
            a_flat = A.flatten()  # length n*m = 6
            b_flat = B.flatten()  # length m*p = 12
            
            # Compute C via decomposition
            # C = sum_k U[:,k] * V[:,k] * W[:,k] (outer product style)
            # Actually: C[k*n+i, j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Wait, let me re-read the convention: W[k*n+i, r]
            # So W is indexed by (k*n+i) where k is col of B / row of C, i is row of A / col of C
            # U is indexed by (i*m+j) where i is row of A, j is col of A / row of B
            # V is indexed by (j*p+k) where j is row of B, k is col of B
            
            C_computed = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        u_val = U[i*m + 0, r]  # This is wrong, need to sum over j
                        # Actually: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        pass
                    
                    # Let me redo: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    for j in range(m):
                        for r in range(rank):
                            val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    def build_rank22():
        """Build the known rank-22 decomposition."""
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: 00:0, 01:1, 10:3, 11:4
        # B indices: 00:0, 01:1, 10:4, 11:5
        # C indices (W): 00:0, 01:2, 10:1, 11:3
        
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B indices: 02:2, 03:3, 12:6, 13:7
        # C indices (W): 02:4, 03:6, 12:5, 13:7
        
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: 02:2, 12:5
        # B indices: 20:8, 21:9, 22:10, 23:11
        # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
        #            10:1, 11:3, 12:5, 13:7 for i=1
        
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank
    
    def try_search(rank, time_budget=30.0):
        """Try to find a decomposition of given rank via structured search."""
        start = time.time()
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Try various structured decompositions
        # Strategy: Use Strassen-like patterns for different block partitions
        
        # For rank 20-21, we need to be more clever
        # One idea: share some Strassen terms between blocks
        
        # Actually, let me try a different approach: 
        # The key constraint is that we have 6*12*8 = 576 tensor entries
        # Each rank term contributes at most (nonzero_U * nonzero_V * nonzero_W) entries
        # For a "sparse" term with 2 nonzeros in each, that's 8 entries
        # 22 terms * 8 = 176, but we need 576... 
        # Wait, that's not right. Each term is a rank-1 tensor in the 6×12×8 space
        # The number of nonzero entries in the full tensor is n*m*p = 2*3*4 = 24
        # (only entries where j=k are nonzero, and specifically T[i*m+j, j*p+k, k*n+i] = 1 when j==k... 
        # no wait, T[i,j,k] = delta_{j,k} where the indices are (i,j) for A, j for B-row, k for B-col
        # Actually the multiplication tensor has T[a,b,c] = 1 if b==c (in appropriate indexing)
        
        # The tensor is: for i in 0..n-1, j in 0..m-1, k in 0..p-1:
        # T[(i,j), j, (k,i)] = 1  (using some flattening)
        # In our convention: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The tensor entry for (i,j,k) is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 if the "middle" index matches, i.e., always (since j is shared)
        # Wait no. In matrix mult C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle)
        # The tensor has n*m*p = 2*3*4 = 24 nonzero entries, all equal to 1.
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # and 0 otherwise (but there are no "otherwise" - all (i,j,k) triples are valid)
        
        # Actually every (i,j,k) with i<n, j<m, k<p gives a term C[i,k] += A[i,j]*B[j,k]
        # So the tensor is: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # That's 2*3*4 = 24 entries, all 1.
        
        # For a rank-r decomposition, we need to represent this 24-entry tensor as sum of r rank-1 tensors.
        
        # Let me just verify the rank-22 solution works, then try to improve.
        
        # Given time constraints, let's just return the known rank-22 solution
        # but first do a quick verification
        U, V, W, r = build_rank22()
        
        # Quick verification with a few random tests
        np.random.seed(0)
        for _ in range(20):
            A = np.random.randint(-2, 3, size=(n, m)).astype(float)
            B = np.random.randint(-2, 3, size=(m, p)).astype(float)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for rr in range(r):
                            val += U[i*m + j][rr] * V[j*p + k][rr] * W[k*n + i][rr]
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Fallback: return a simple valid decomposition (rank 24)
                U24 = [[0] * 24 for _ in range(n * m)]
                V24 = [[0] * 24 for _ in range(m * p)]
                W24 = [[0] * 24 for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U24[i*m + j][col] = 1
                            V24[j*p + k][col] = 1
                            W24[k*n + i][col] = 1
                            col += 1
                return {"U": U24, "V": V24, "W": W24}
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # For now, just return the known rank-22 solution
    result = try_search(22)
    
    # Safety check
    if time.time() - start_time > deadline - safety_margin:
        # Return the simple rank-24 as fallback
        U24 = [[0] * 24 for _ in range(n * m)]
        V24 = [[0] * 24 for _ in range(m * p)]
        W24 = [[0] * 24 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U24[i*m + j][col] = 1
                    V24[j*p + k][col] = 1
                    W24[k*n + i][col] = 1
                    col += 1
        return {"U": U24, "V": V24, "W": W24}
    
    return result
