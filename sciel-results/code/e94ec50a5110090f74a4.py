"""Matrix multiplication 3x5 by 5x6: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 90

    # Schoolbook decomposition as fallback
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

    # Build the matrix multiplication tensor T[i,j,k] = delta
    # The tensor has indices: row (i from A, j from B), col (k from B, i from C)
    # T_{(i,j),(j',k'),(k,i')} = 1 if j=j' and i=i', else 0
    
    # For rank-r decomposition, we need:
    # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T
    
    # The tensor entries are 1 when (i,j) pairs with (j,k) to give (k,i),
    # i.e., U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] sums to 1
    # for the "diagonal" terms and 0 for off-diagonal.
    
    # This is equivalent to: for each (i,j,k) with the constraint that 
    # the multiplication is A[i,j] * B[j,k] -> C[k,i] (trace convention W[k*n+i])
    # We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 
    # when this is a valid multiplication term, 0 otherwise.
    
    # Actually, the matrix multiplication tensor is:
    # T[(i,j), (j',k'), (k'',i'')] = 1 iff j=j' and i=i''
    # So for a given (i,j) in A, (j',k') in B, (k'',i'') in C:
    # T = 1 if j == j' and i == i''
    
    # Let's use a different approach: try to find a rank-r decomposition
    # by solving a system of equations.
    
    # The number of constraints is n*m*p = 90 (each (i,j,k) triple gives one equation)
    # For rank r, we have r*(n*m + m*p + p*n) = r*(15 + 30 + 18) = 63r variables
    # We need 63r >= 90, so r >= ceil(90/63) = 2 (trivially)
    # But the equations are bilinear, so we need more.
    
    # Let's try to construct a decomposition with rank < 90.
    # One known technique: use the fact that matrix multiplication can be
    # decomposed using fewer multiplications for small matrices.
    
    # For 3x5 * 5x6, let's try to find a decomposition using a randomized
    # approach with integer/half-integer entries.
    
    # Approach: Start with schoolbook and try to merge terms.
    # Or: Use a structured approach based on the Kronecker product structure.
    
    # Let's try a direct search: for small ranks, check if a decomposition exists.
    # We'll use a simple iterative method.
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # for all (i,j,k) where the multiplication is defined
        # Actually, the tensor is:
        # T[(i,j),(j',k'),(k'',i'')] = 1 if j==j' and i==i''
        # So for each i in [0,n), j in [0,m), k in [0,p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # And for all other combinations, it should equal 0.
        
        # Let's verify all 15*30*18 = 8100 tensor entries
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each (i,j) in A, (j',k') in B, (k'',i'') in C:
        # T = 1 if j == j' and i == i'', else 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (j == jp and i == ip) else 0.0
                                val = 0.0
                                for r in range(len(U[0])):
                                    val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                                if abs(val - expected) > 1e-9:
                                    return False
        return True
    
    def check_decomposition_fast(U, V, W):
        """Faster check using numpy broadcasting."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = len(U[0])
        
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We need: for each (i,j) in [0,n)x[0,m), (jp,k) in [0,m)x[0,p), (kp,ip) in [0,px[0,n):
        # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = 1 if j==jp and i==ip else 0
        
        # This is equivalent to checking that the tensor T decomposes correctly.
        # Let's use a different check: 
        # The product C = A*B has C[k, i] = sum_j A[i, j] * B[j, k]
        # In the trace convention W[k*n+i], C is stored as W[k*n+i]
        # So C[k, i] = sum_j A[i, j] * B[j, k]
        
        # For the decomposition: C[k, i] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need this to equal sum_j A[i, j] * B[j, k] for all A, B.
        
        # This means: for each (k, i), the sum over j of (sum_r U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
        # should equal sum_j A[i,j] * B[j,k]
        
        # Actually, the bilinear form is:
        # C[k, i] = sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        
        # For this to equal sum_j A[i,j] * B[j,k] for all A, B,
        # we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,j'} * delta_{i,i'}
        # for the appropriate indices.
        
        # Let's just verify a few random A, B matrices.
        random.seed(42)
        for _ in range(20):
            A = np.random.randint(-2, 3, size=(n, m))
            B = np.random.randint(-2, 3, size=(m, p))
            C_expected = A @ B
            C_decomp = np.zeros((p, n))
            for i in range(n):
                for j in range(m):
                    for r in range(r):
                        for k in range(p):
                            C_decomp[k, i] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if not np.allclose(C_decomp, C_expected, atol=1e-9):
                return False
        return True
    
    # Try to find a rank-r decomposition for r < 90.
    # Strategy: Use a structured approach.
    # For small matrices, there are known algorithms with fewer multiplications.
    
    # Let's try a simple approach: assume entries are in {-1, 0, 1, 0.5}
    # and search for a decomposition.
    
    best_result = schoolbook()
    best_rank = rank_sb
    
    # Try ranks from a reasonable lower bound up to 89
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But that's not a tight bound. The actual lower bound is higher.
    
    # For 3x5 * 5x6, let's try to find a decomposition with rank < 90.
    # We'll use a randomized search.
    
    def random_decomposition(rank, seed=0):
        """Generate a random decomposition with integer/half-integer entries."""
        random.seed(seed)
        np.random.seed(seed)
        values = [-1, -0.5, 0, 0.5, 1]
        U = np.random.choice(values, size=(n*m, rank))
        V = np.random.choice(values, size=(m*p, rank))
        W = np.random.choice(values, size=(p*n, rank))
        return U.tolist(), V.tolist(), W.tolist()
    
    # Search for a better decomposition
    for target_rank in range(50, 90):  # Try ranks 50-89
        if time.time() > deadline - 10:
            break
        
        # Try a few random decompositions for this rank
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            U, V, W = random_decomposition(target_rank, seed=random.randint(0, 100000))
            
            # Quick check: verify with a few random matrices
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            valid = True
            for _ in range(5):
                A = np.random.randint(-2, 3, size=(n, m))
                B = np.random.randint(-2, 3, size=(m, p))
                C_expected = A @ B
                C_decomp = np.zeros((p, n))
                for i in range(n):
                    for j in range(m):
                        for r in range(target_rank):
                            for k in range(p):
                                C_decomp[k, i] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if not np.allclose(C_decomp, C_expected, atol=1e-9):
                    valid = False
                    break
            
            if valid:
                # Full verification
                if check_decomposition_fast(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    # Try to find even better
                    break
    
    if time.time() > deadline - 5:
        return best_result
    
    # If no better decomposition found, return the schoolbook
    return best_result
# EVOLVE-BLOCK-END
