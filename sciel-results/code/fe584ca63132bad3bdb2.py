"""Matrix multiplication 3,3,6 - low rank tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # T[i,j,k] = 1 if j==k (C[i,k] = sum_j A[i,j]*B[j,k])
    # Tensor shape: (n, m, p) = (3, 3, 6)
    # Flattened index: i*9 + j*6 + k
    
    def time_check():
        return time.time() - start_time > DEADLINE
    
    def verify(U, V, W, rank):
        """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] = T"""
        # U: (n*m, rank) = (9, rank)
        # V: (m*p, rank) = (18, rank)
        # W: (p*n, rank) = (18, rank)
        # T_flat[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
        # T[a,b,c] = 1 iff j==k (i.e., (a%m)==(b//p))
        for c in range(rank):
            u = np.array(U[:, c]).reshape(n, m)
            v = np.array(V[:, c]).reshape(m, p)
            w = np.array(W[:, c]).reshape(p, n)
            # term = u[i,j] * v[j,k] * w[k,i]
            # We check: sum over c of u[i,j]*v[j,k]*w[k,i] should equal 1 if j==k else 0
            # Build the term tensor
            term = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # Expected: 1 if j==k else 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if j == k else 0.0
                        if abs(term[i,j,k] - expected) > 1e-9:
                            return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy broadcasting."""
        result = np.zeros((n, m, p))
        for c in range(rank):
            u = np.array(U[:, c]).reshape(n, m)
            v = np.array(V[:, c]).reshape(m, p)
            w = np.array(W[:, c]).reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # Check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    if abs(result[i,j,k] - expected) > 1e-9:
                        return False
        return True
    
    # Best known fallback: schoolbook rank 54
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_result = schoolbook()
    
    # Try to find lower rank decompositions using randomized approach
    # Key insight: T[i,j,k] = delta(j,k)
    # We can factor: T = sum over some structure
    # 
    # Approach: Try rank r < 54 with random initialization + algebraic refinement
    # Using the fact that the tensor has structure (Kronecker-like)
    
    # Try specific structured decompositions
    
    # Method 1: Use the fact that T[i,j,k] = delta(j,k)
    # This means for each (i,k), the "slice" over j is a unit vector
    # We can try: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i,j] * v_c[j,k] * w_c[k,i] contributes to delta(j,k)
    
    # A natural decomposition: for each j, T[i,j,j] = 1 for all i
    # So T = sum_j (e_i over i) ⊗ (e_j over j, e_j over k) ⊗ (e_j over k, e_i over i)
    # That's just schoolbook with rank = n*p... no wait.
    
    # Actually: T[i,j,k] = 1 if j==k
    # = sum_j delta(i,i') ... no
    # = sum_j (sum_i delta(i,i)) * delta(j,j) * delta(k,k) ... 
    # = sum_j (ones vector in i) ⊗ (e_j in j, e_j in k) ⊗ (e_j in k, ones in i)
    # Wait, that's not right either because the i dimension is free.
    
    # T[i,j,k] = delta(j,k) for all i
    # = sum_j [vector of 1s in i] ⊗ [e_j in j-position, e_j in k-position of B] ⊗ [e_j in k-position, 1s in i-position of C]
    # Hmm, let me think more carefully.
    # 
    # U has shape (n*m, r), indexed by (i,j)
    # V has shape (m*p, r), indexed by (j,k)  
    # W has shape (p*n, r), indexed by (k,i)
    # 
    # T[i,j,k] = sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r]
    # 
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if j==k else 0
    # 
    # Key observation: the RHS depends only on (j,k), not on i!
    # So for all i: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta(j,k)
    #
    # This is a strong constraint. Let's exploit it.
    #
    # If we set U[i,j,r] independent of i (i.e., U[i,j,r] = U[j,r] for all i),
    # and W[k,i,r] = W[k,r] for all i, then:
    # sum_r U[j,r] * V[j,k,r] * W[k,r] = delta(j,k) for all i
    # This reduces to: for each (j,k), sum_r U[j,r]*V[j,k,r]*W[k,r] = delta(j,k)
    # 
    # So we need matrices U (m x r), V (m x p x r), W (p x r) such that
    # for all j,k: sum_r U[j,r] * V[j,k,r] * W[k,r] = delta(j,k)
    #
    # This is equivalent to: the m x p matrix M_{jk} = sum_r U[j,r]*V[j,k,r]*W[k,r] = I_m (for j=k) or 0
    # Wait, j ranges over m=3, k ranges over p=6. So M is 3x6.
    # M[j,k] = delta(j,k) means M = [I_3 | 0_{3x3}]
    #
    # So we need: M_{jk} = sum_r U[j,r] * V[j,k,r] * W[k,r] = [I_3 0]
    #
    # This is a matrix factorization problem! We need to write the 3x6 matrix [I_3 0]
    # as a sum of rank-1 contributions through the V tensor.
    #
    # Simplest: r = 3, U = I_3, W = I_3 (for k<3), V[j,k,r] = delta(j,r)*delta(k,r)
    # Then M[j,k] = sum_r delta(j,r)*delta(j,k,r... ) hmm let me just do it directly.
    #
    # With r=3: U[j,r] = delta(j,r), W[k,r] = delta(k,r) for k<3, 0 for k>=3
    # V[j,k,r] needs: sum_r delta(j,r)*V[j,k,r]*delta(k,r) = V[j,k,j] (if j<3 and k=j) 
    # = delta(j,k) for k=j<3. And for k>=3, W[k,r]=0 so M[j,k]=0. 
    # So set V[j,k,r] = delta(j,r)*delta(k,r). Then M[j,k] = sum_r delta(j,r)*delta(j,r)*delta(k,r)
    # = delta(j,k) for k<3. For k>=3, W[k,r]=0. 
    # But we need M[j,k]=0 for j!=k, which is satisfied since delta(j,r)*delta(k,r) with r=j gives delta(j,k).
    #
    # So rank 3 works?! Let me verify.
    # U[i,j,r] = delta(j,r) for r in {0,1,2}, 0 otherwise. Shape (9, 3)
    # V[j,k,r] = delta(j,r)*delta(k,r). Shape (18, 3)
    # W[k,i,r] = delta(k,r) for r in {0,1,2}. Shape (18, 3)
    #
    # T[i,j,k] = sum_{r=0}^{2} delta(j,r) * delta(j,r)*delta(k,r) * delta(k,r)
    # = sum_r delta(j,r)^2 * delta(k,r)^2 = sum_r delta(j,r)*delta(k,r) = delta(j,k) (for k<3)
    # For k>=3: delta(k,r) = 0 for all r in {0,1,2}, so T=0. 
    # But we need T[i,j,k] = 1 if j==k. For k>=3, j can be at most 2, so j!=k always. T=0. Correct!
    # For k<3: T = delta(j,k). Correct!
    #
    # RANK 3! That's amazing. Let me double check.
    #
    # n=3, m=3, p=6
    # U: (9, 3), U[i*m+j, r] = delta(j, r) for r=0,1,2
    # V: (18, 3), V[j*p+k, r] = delta(j, r) * delta(k, r)
    # W: (18, 3), W[k*n+i, r] = delta(k, r)
    
    U3 = [[0]*3 for _ in range(n*m)]
    V3 = [[0]*3 for _ in range(m*p)]
    W3 = [[0]*3 for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for r in range(3):
                U3[i*m+j][r] = 1 if j == r else 0
    for j in range(m):
        for k in range(p):
            for r in range(3):
                V3[j*p+k][r] = 1 if (j == r and k == r) else 0
    for k in range(p):
        for i in range(n):
            for r in range(3):
                W3[k*n+i][r] = 1 if k == r else 0
    
    if verify(U3, V3, 3):
        return {"U": U3, "V": V3, "W": W3}
    
    # If rank 3 didn't verify (shouldn't happen), try rank 6
    U6 = [[0]*6 for _ in range(n*m)]
    V6 = [[0]*6 for _ in range(m*p)]
    W6 = [[0]*6 for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for r in range(6):
                U6[i*m+j][r] = 1 if j == (r % 3) else 0
    for j in range(m):
        for k in range(p):
            for r in range(6):
                V6[j*p+k][r] = 1 if (j == (r % 3) and k == r) else 0
    for k in range(p):
        for i in range(n):
            for r in range(6):
                W6[k*n+i][r] = 1 if k == r else 0
    
    if verify(U6, V6, 6):
        return {"U": U6, "V": V6, "W": W6}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {"U": U, "V": V, "W": W}
