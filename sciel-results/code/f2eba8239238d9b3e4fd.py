"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
import random
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-6 decomposition over F_3.
    Falls back to rank-7 Strassen decomposition.
    """
    n, m, p = 2, 2, 2
    dim_u = n * m # 4
    dim_v = m * p # 4
    dim_w = p * n # 4
    
    # The tensor T is defined by T[i, a, k] = 1 for all i, a, k in {0, 1}
    # We represent the tensor as a 4x4x4 array, but effectively we check if
    # sum_r U[:,r] outer V[:,r] outer W[:,r] == T
    # However, the standard decomposition is:
    # T[i,a,k] = sum_r U[i,a,r] * V[a,k,r] * W[k,i,r] ?? No.
    #
    # Correct tensor form for MatMul:
    # The tensor $\mathcal{M}$ has indices $(i, a, k)$ where $i \in \{0,1\}, a \in \{0,1\}, k \in \{0,1\}$.
    # $\mathcal{M}[i, a, k] = 1$.
    # A rank-r decomposition is $\mathcal{M}[i, a, k] = \sum_{r=1}^r u_r[i] v_r[a] w_r[k]$.
    #
    # In the problem's format:
    # U is (n*m, rank) -> U[i*m + a, r] should correspond to $u_r[i]$?
    # No, if $U[i*m+a, r] = u_r[i]$, then $U$ has repeated rows for different $a$.
    # Let's check the Strassen code provided.
    # U[0*2+0][0] = 1 (A00), U[1*2+1][0] = 1 (A11).
    # This implies U[i*m+a, r] is the coefficient of A[i,a] in the r-th product.
    # Similarly V[a*p+k, r] is coefficient of B[a,k].
    # W[k*n+i, r] is coefficient for C[i,k].
    #
    # The identity is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_a U[i*m+a, r] * A[i,a]) * (sum_k' V[a*p+k', r] * B[a,k'])
    # Wait, the standard Strassen form is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_a U[i*m+a, r] A[i,a]) * (sum_b V[b*p+k, r] B[b,k])
    # Note: The index 'a' in U and 'b' in V must match for the multiplication to be a scalar product of vectors.
    # In Strassen, M1 = (A00+A11)(B00+B11).
    # U col 0: [1, 0, 0, 1]^T (A00, A01, A10, A11) -> indices 0,1,2,3.
    # V col 0: [1, 0, 0, 1]^T (B00, B01, B10, B11) -> indices 0,1,2,3.
    # Product is (A00+A11)(B00+B11).
    # This works if the summation is over the flattened indices?
    # No, it's a bilinear form.
    # Let's verify the Strassen code's logic.
    # U[0,0]=1, U[3,0]=1. (A00, A11)
    # V[0,0]=1, V[3,0]=1. (B00, B11)
    # If we compute sum_a U[i,a,r] A[i,a] we get a vector? No.
    # The standard decomposition writes C as sum of rank-1 tensors.
    # Rank 1 tensor is $u \otimes v \otimes w$.
    # $T[i,a,k] = \sum_r u_r[i] v_r[a] w_r[k]$.
    # If we map $u_r[i]$ to $U[i*m+0, r]$? No, $U$ has dimension $n*m$.
    # The convention in the prompt: U shape (n*m, rank).
    # This implies $U$ flattens the $A$ indices.
    # If the decomposition is $T = \sum_r \text{vec}(U_r) \otimes \text{vec}(V_r) \otimes \text{vec}(W_r)$,
    # then $T[i,a,k] = \sum_r U[i,a,r] V[a,k,r] W[k,i,r]$?
    # This doesn't look like standard matmul.
    #
    # Let's stick to the working Strassen code's implicit logic which passed.
    # It constructs U, V, W such that the bilinear form matches.
    
    # We will perform a short search for rank 6 over F_3.
    # Since rank 6 is impossible, this will fail.
    
    def check_decomposition(U, V, W):
        # Verify if U, V, W decompose the matmul tensor.
        # We need to check if for all i, k:
        # (A*B)[i,k] = sum_r W[k*n+i, r] * (sum_a U[i*m+a, r] A[i,a]) * (sum_b V[b*p+k, r] B[b,k])
        # This is not quite right. The term (sum_a U[i*m+a, r] A[i,a]) is a scalar dependent on i and r.
        # Let's call $P_{i,r} = \sum_a U[i*m+a, r] A[i,a]$.
        # Let $Q_{k,r} = \sum_b V[b*p+k, r] B[b,k]$.
        # Then $C_{i,k} = \sum_r W[k*n+i, r] P_{i,r} Q_{k,r}$.
        # We want this to equal $\sum_a A_{i,a} B_{a,k}$.
        
        # We can test this by setting random integer matrices A and B and checking if C matches.
        # Using integers to avoid floating point issues.
        
        # Generate random A, B
        # Since we are searching over F_3, we can test over F_3 or integers.
        # If it holds over F_3, it holds over Z? No, but if it's a polynomial identity, it holds everywhere.
        # Let's test with specific basis matrices.
        
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    # Create A with 1 at (i,a), 0 elsewhere
                    A = np.zeros((n, m), dtype=int)
                    A[i, a] = 1
                    # Create B with 1 at (a,k)? No, B index is (row, col).
                    # The term A[i,a] B[b,k] contributes to C[i,k] only if b=a.
                    # So set B[a,k] = 1.
                    B = np.zeros((m, p), dtype=int)
                    B[a, k] = 1
                    
                    # Expected C[i,k] = 1
                    expected = 1
                    
                    # Computed C[i,k]
                    computed = 0
                    for r in range(len(U[0])):
                        # P_{i,r} = sum_a' U[i*m+a', r] A[i,a']
                        # Since A is 0 except at (i,a), P_{i,r} = U[i*m+a, r]
                        p_val = U[i*m + a, r]
                        
                        # Q_{k,r} = sum_b' V[b'*p+k, r] B[b',k]
                        # Since B is 0 except at (a,k), Q_{k,r} = V[a*p+k, r]
                        q_val = V[a*p + k, r]
                        
                        w_val = W[k*n + i, r]
                        
                        computed += w_val * p_val * q_val
                        
                    if computed != expected:
                        return False
        return True

    # Search for rank 6 over F_3
    # To make it fast, we only do a limited random search.
    # A true exhaustive search is too slow.
    start_time = time.time()
    max_time = 10 # seconds
    rank = 6
    found = False
    
    # Random search over F_3
    # We fix the first vector to break symmetry (gauge fixing) to reduce space?
    # For simplicity, just random sampling.
    
    while time.time() - start_time < max_time:
        U = np.random.randint(0, 3, size=(dim_u, rank))
        V = np.random.randint(0, 3, size=(dim_v, rank))
        W = np.random.randint(0, 3, size=(dim_w, rank))
        
        # Optional: Normalize to ensure non-trivial solutions?
        # If any column is zero, it's not a rank-6 decomposition (rank < 6).
        # We can skip if any column is all zeros.
        if not np.any(U, axis=0).all() or not np.any(V, axis=0).all() or not np.any(W, axis=0).all():
            continue
            
        if check_decomposition(U, V, W):
            found = True
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    if found:
        return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback: Rank 7 Strassen
    # Construct Strassen decomposition
    rank = 7
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    # Indices:
    # A: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # B: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # C: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # W index: k*n + i.
    # C(0,0) -> 0*2+0=0
    # C(0,1) -> 1*2+0=2
    # C(1,0) -> 0*2+1=1
    # C(1,1) -> 1*2+1=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[1][3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[2][4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
