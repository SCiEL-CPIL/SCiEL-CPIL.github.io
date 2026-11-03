"""Matrix multiplication 3x3 x 3x8 rank minimization via structured search."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 72
    # We want to find a lower rank exact decomposition.
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U, V, W such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Start with schoolbook (rank 72) and try to merge terms.
    # A merge is possible if we can replace a set of schoolbook terms with fewer terms.
    
    # For 3x3x3, Strassen reduces rank from 27 to 21 (saves 6 per 3x3x3 block).
    # Here we have 3x3x8. We can view this as 3 blocks of 3x3x3 plus one 3x3x2.
    # Actually, it's 8 columns in k. Let's think of it as two 3x3x4 blocks or similar.
    
    # Let's try a direct approach: generate candidate rank-1 tensors and solve.
    # But that's too slow for rank < 72.
    
    # Better approach: Use the known Strassen-like decomposition for 3x3x3 blocks.
    # For each 3x3x3 sub-block (i in 0..2, j in 0..2, k in 0..2), we can use rank 21 instead of 27.
    # We have 8 k-values. Let's partition k into groups.
    
    # Actually, let's try a simpler structural approach:
    # The multiplication (3x3) * (3x8) -> (3x8)
    # This is equivalent to 8 separate (3x3)*(3x1) multiplications, but they share the first matrix.
    # Wait, no. A is 3x3, B is 3x8. C = A*B is 3x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor has entries T[i,j,k] = 1.
    # U has shape (3*3, rank) = (9, rank)
    # V has shape (3*8, rank) = (24, rank)
    # W has shape (8*3, rank) = (24, rank)
    
    # Let's try to find a decomposition with rank < 72.
    # One idea: use the fact that for each k, the multiplication is (3x3)*(3x1).
    # For a single (3x3)*(3x1) -> (3x1), the rank is at least 9 (schoolbook).
    # But since A is shared across all k, we can exploit this.
    
    # Actually, the tensor is T[i,j,k] = 1 for all i,j,k.
    # This is a product tensor: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # The rank of this product tensor is n*m*p = 72.
    # But we can do better by combining terms.
    
    # Let's try a heuristic search: start with schoolbook and try to remove terms
    # while maintaining exactness.
    
    # For now, let's implement a basic Strassen-like approach for the 3x3x3 blocks.
    # Partition k into two groups: k=0,1,2 and k=3,4,5,6,7.
    # For the first 3 k-values, use a 3x3x3 decomposition.
    # For the remaining 5 k-values, use schoolbook.
    
    # Actually, let's just try to find any rank < 72 by random search.
    # Generate random sparse U, V, W and check if they satisfy the equations.
    
    # The equations are: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 72 equations in rank variables (3*rank + 3*rank + 8*rank = 14*rank unknowns).
    
    # Let's try rank = 68 first (saving 4).
    
    # Actually, let's use a more systematic approach.
    # We know that for 3x3x3, the minimum rank is 21 (Strassen).
    # For 3x3x8, we can think of it as 8 copies of 3x3x1, but with shared A.
    
    # Let me try a different approach: use the known result that the rank of
    # (n x m) * (m x p) is at most n*m*p - min(n,m,p) + 1 or similar bounds.
    
    # For now, let's implement a simple local search:
    # 1. Start with schoolbook rank 72.
    # 2. Try to remove one term at a time and solve for the remaining.
    
    # This is complex. Let's just return schoolbook for now and improve iteratively.
    
    # Actually, let's try a specific construction.
    # For each pair (i,j), we need to compute sum_k A[i,j]*B[j,k] for all k.
    # This is just A[i,j] times the j-th row of B.
    # So C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] = 1.
    # We can write T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # But we can combine terms: e_i ⊗ (sum_k e_k) ⊗ e_k is not valid since it's not rank 1.
    
    # Let's try: for each i,j, define u_{i,j} = e_i ⊗ e_j, v_{i,j} = sum_k e_k (in the B space), w_{i,j} = ?
    # No, that doesn't work because W depends on k and i.
    
    # Let me think differently. 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as: for each i, C[i,:] = sum_j A[i,j] * B[j,:]
    # This is a linear combination of the rows of B, with coefficients from row i of A.
    
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We need this to equal 1 for all i,j,k.
    
    # Let's try a construction where we use rank = n*p + m*p + n*m - something.
    
    # Actually, the simplest improvement is to use the fact that for fixed i,
    # the terms involving different j can be combined.
    
    # Let's just implement a random search for rank < 72.
    
    import random
    random.seed(42)
    
    best_rank = n * m * p  # 72
    best_solution = None
    
    # Try ranks from 71 down
    for rank in range(71, 60, -1):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try a few random initializations
        for attempt in range(100):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Initialize with schoolbook-like structure
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # For each c, assign a random (i,j,k)
            for c in range(rank):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
            
            # Check if this satisfies the equations
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_solution = (U.copy(), V.copy(), W.copy())
                break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists of lists with integer/half-integer entries
        U_list = [[float(U[i, c]) for c in range(best_rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(best_rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
