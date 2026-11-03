"""Rewritten program: structured search for low-rank decomposition of 4x4*4x9 matrix multiplication tensor."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    
    The tensor T is defined by:
    For A (n x m) * B (m x p) = C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The decomposition uses U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k], which is always true.
    
    Wait - this interpretation gives an all-ones tensor which has rank 1. That can't be right.
    
    Let me reconsider. The correct tensor for matrix multiplication is:
    T[i, j, k] where the indices correspond to (row of A, col of A / row of B, col of B)
    and the "output" index is determined by the constraint that col of A = row of B.
    
    Actually, I think the standard formulation is:
    The multiplication tensor is a 3-tensor with entries indexed by (i, j, k) where
    i ranges over rows of A (0..n-1), j over columns of A / rows of B (0..m-1), 
    k over columns of B (0..p-1).
    
    The tensor value T[i,j,k] = 1 for all valid (i,j,k) because every A[i,j]*B[j,k] 
    contributes to C[i,k].
    
    But this gives rank 1! So I must be wrong.
    
    Let me look at this from the Strassen perspective. For 2x2*2x2:
    Schoolbook uses 8 multiplications. The tensor has 8 "terms" A[i,j]*B[j,k] for 
    i,j,k in {0,1}. The rank of this tensor is 7 (Strassen).
    
    So the tensor is NOT all-ones in the sense I was thinking. Let me reconsider.
    
    Actually, I think the issue is that the tensor is defined over the full index space
    (n*m) x (m*p) x (p*n), and T[a,b,c] = 1 if and only if the indices are "compatible" 
    in a specific way.
    
    For the schoolbook decomposition:
    For each (i,j,k) with i in [0,n), j in [0,m), k in [0,p):
      U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    This means the tensor entry T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k).
    
    So the tensor is NOT all ones. It's a specific pattern where T[a,b,c] = 1 if and only
    if there exist i,j,k such that a = i*m+j, b = j*p+k, c = k*n+i.
    
    In other words, T is a 0-1 tensor where T[a,b,c] = 1 iff the "middle indices" match:
    if a = i*m+j and b = j'*p+k, then we need j = j', and c = k*n+i.
    
    So T[a,b,c] = 1 iff:
      Let i = a // m, j = a % m
      Let j' = b // p, k = b % p  
      Let k' = c // n, i' = c % n
      Then T[a,b,c] = 1 iff j == j' AND i == i'
    
    This is the correct tensor! It's not all ones.
    
    Now I can search for low-rank decompositions of this tensor.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (10s safety margin)
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Build the target tensor as a sparse representation
    # T[a,b,c] = 1 iff (a%m == b//p) and (a//m == c%n)
    # where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        # Check all entries of the tensor
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                if j != j2:
                    # T[a,b,c] should be 0 for all c
                    for c in range(p * n):
                        s = sum(U[a][c] * V[b][c] * W[c] for c in range(rank))
                        if abs(s) > 1e-9:
                            return False
                else:
                    # T[a,b,c] = 1 iff c%n == i
                    for c in range(p * n):
                        i2 = c % n
                        target = 1.0 if i2 == i else 0.0
                        s = sum(U[a][c] * V[b][c] * W[c] for c in range(rank))
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using matrix operations."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # For each c, compute outer product contribution
        # This is expensive, so use a different approach
        # Check a random subset of entries
        np.random.seed(12345)
        n_samples = 2000
        for _ in range(n_samples):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            i2 = c % n
            
            if j != j2:
                target = 0.0
            else:
                target = 1.0 if i2 == i else 0.0
            
            s = np.sum(U[a] * V[b] * W[c])
            if abs(s - target) > 1e-8:
                return False
        return True
    
    def verify_full(U, V, W):
        """Full verification - slower but exact."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # For each (a,b), compute the vector over c
        # sum_c U[a,c]*V[b,c]*W[c] should equal T[a,b,:]
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                if j != j2:
                    # Should be all zeros
                    vec = U[a] * V[b] * W
                    if np.max(np.abs(vec)) > 1e-9:
                        return False
                else:
                    # Should be 1 where c%n == i, 0 otherwise
                    vec = U[a] * V[b] * W
                    for c in range(p * n):
                        i2 = c % n
                        target = 1.0 if i2 == i else 0.0
                        if abs(vec[c] - target) > 1e-9:
                            return False
        return True
    
    # Try to find a low-rank decomposition using structured search
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something
    # For 4x4*4x9: n*m=16, m*p=36, p*n=36
    # Schoolbook rank = 144
    
    # Let's try a gradient-free optimization approach
    # Start with random initialization and use coordinate descent
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Generate schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try structured decompositions
    # Key insight: we can potentially use Kronecker products or block structure
    
    # Approach 1: Try rank values and use random + local search
    for target_rank in [64, 80, 96, 108, 120, 132]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + attempt * 7)
            U = np.random.uniform(-0.5, 0.5, (n * m, target_rank))
            V = np.random.uniform(-0.5, 0.5, (m * p, target_rank))
            W = np.random.uniform(-0.5, 0.5, (p * n, target_rank))
            
            # Use a simple iterative refinement
            # The constraint is: for each (a,b,c), sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
            # This is a system of trilinear equations
            
            # Try a few iterations of fixing two matrices and solving for the third
            for it in range(50):
                if time.time() > deadline - 8:
                    break
                
                # Fix U and V, solve for W
                # For each c: sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c] for all (a,b)
                # This is a linear system in W[c,:] for each c
                # But it's coupled across c through the same k index
                
                # Actually, for fixed U and V, the equation is:
                # For each (a,b,c): W[c, :] dot (U[a,:] * V[b,:]) = T[a,b,c]
                # where U[a,:]*V[b,:] is element-wise product
                
                # For each c, we need W[c,:] to satisfy:
                # sum_k W[c,k] * U[a,k] * V[b,k] = T[a,b,c] for all (a,b)
                
                # This is a linear system: M_c @ W[c,:] = t_c
                # where M_c[a,b] = sum_k U[a,k]*V[b,k] ... no that's not right.
                
                # Actually M_c is not well-defined this way. Let me reconsider.
                
                # The equation for fixed U,V is:
                # For each triple (a,b,c): sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
                
                # Group by c: for each c, we have a linear system in W[c,:]:
                # sum_k W[c,k] * (U[a,k]*V[b,k]) = T[a,b,c] for all (a,b)
                
                # So for each c, define A_c[a,b] = sum_k ... no, A_c doesn't depend on c.
                # The coefficient of W[c,k] is U[a,k]*V[b,k], which doesn't depend on c!
                
                # So for each c, the system is:
                # For all (a,b): sum_k (U[a,k]*V[b,k]) * W[c,k] = T[a,b,c]
                
                # This is: B @ W[c,:] = t_c where B[a,b,k] = U[a,k]*V[b,k]
                # Reshape: for each (a,b), the equation is W[c,:] dot (U[a,:]*V[b,:]) = T[a,b,c]
                
                # So we have a system with n*m*m*p equations and target_rank unknowns (for W[c,:])
                # This is overdetermined if n*m*m*p > target_rank, which it is (16*4*9=576 > target_rank)
                
                # Use least squares: W[c,:] = pinv(B_c) @ t_c where B_c is the matrix
                # with rows = U[a,:]*V[b,:] for all (a,b), and t_c[a,b] = T[a,b,c]
                
                # Build B: shape (n*m*m*p, target_rank)
                B = np.zeros((n * m * m * p, target_rank))
                idx = 0
                for a in range(n * m):
                    for b in range(m * p):
                        B[idx] = U[a] * V[b]
                        idx += 1
                
                # For each c, solve B @ W[c,:] = t_c
                for c in range(p * n):
                    t_c = np.zeros(n * m * m * p)
                    idx2 = 0
                    for a in range(n * m):
                        i = a // m
                        j = a % m
                        for b in range(m * p):
                            j2 = b // p
                            if j == j2:
                                i2 = c % n
                                if i2 == i:
                                    t_c[idx2] = 1.0
                            idx2 += 1
                    
                    # Solve least squares
                    try:
                        W[c] = np.linalg.lstsq(B, t_c, rcond=None)[0]
                    except:
                        W[c] = np.random.uniform(-0.1, 0.1, target_rank)
                
                # Check error
                err = 0
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            s = np.dot(U[a] * V[b], W[c])
                            i = a // m
                            j = a % m
                            j2 = b // p
                            i2 = c % n
                            target = 1.0 if (j == j2 and i2 == i) else 0.0
                            err += (s - target) ** 2
                err = np.sqrt(err)
                
                if err < 1e-6:
                    # Verify fully
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if verify_full(U_list, V_list, W_list):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
            if found:
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
