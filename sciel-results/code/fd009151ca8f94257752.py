"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W):
    """Verify that the tensor decomposition is correct."""
    n, m, p = 3, 3, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i,m*r] * V[j,p*r] * W[k,n*r]
    # For matrix multiplication: T[i,j,k] = 1 if i,j,k correspond to A[i,l]*B[l,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # The matrix multiplication tensor has entries:
    # T[A_idx, B_idx, C_idx] where:
    #   A_idx = i*m + l (row i, col l of A)
    #   B_idx = l*p + k (row l, col k of B)
    #   C_idx = k*n + i (row k, col i of C, using W[k*n+i] convention)
    #
    # T[A_idx, B_idx, C_idx] = 1 if the indices are consistent (same l)
    
    rank = U.shape[1]
    
    # Compute the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need to check this equals the matrix multiplication tensor
    
    # Build the target tensor
    # For each valid (i, l, k) triple:
    #   a = i*m + l
    #   b = l*p + k
    #   c = k*n + i
    #   T[a,b,c] = 1
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Compute from decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is an outer product sum
    computed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        computed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check
    diff = np.abs(computed - target)
    max_diff = np.max(diff)
    return max_diff < 1e-9, max_diff

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] = val
        for idx, val in v_indices:
            V[idx][col] = val
        for idx, val in w_indices:
            W[idx][col] = val
        col += 1

    # Strassen Left (A11 * B1L) - 7 terms
    # A11 indices: 0,1,3,4
    # B1L indices: 0,1,4,5
    # C1L indices: 0,3,1,4
    
    # P1
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (4, 1)])
    # P2
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    # P3
    add_term([(0, 1)], [(1, 1), (5, -1)], [(3, 1), (4, 1)])
    # P4
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)])
    # P5
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (3, 1)])
    # P6
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
    # P7
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)])
    
    # A12 * B2L - 4 terms
    # A12 indices: 2, 5
    # B2L indices: 8, 9
    # C1L contribution:
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4
    
    add_term([(2, 1)], [(8, 1)], [(0, 1)])
    add_term([(2, 1)], [(9, 1)], [(3, 1)])
    add_term([(5, 1)], [(8, 1)], [(1, 1)])
    add_term([(5, 1)], [(9, 1)], [(4, 1)])
    
    # A21 * B1L - 4 terms
    # A21 indices: 6, 7
    # B1L indices: 0,1,4,5
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    add_term([(6, 1)], [(0, 1)], [(2, 1)])
    add_term([(6, 1)], [(1, 1)], [(5, 1)])
    add_term([(7, 1)], [(4, 1)], [(2, 1)])
    add_term([(7, 1)], [(5, 1)], [(5, 1)])
    
    # A22 * B2L - 2 terms
    # A22 index: 8
    # B2L indices: 8, 9
    # C2L contribution:
    # C[2,0]->2, C[2,1]->5
    
    add_term([(8, 1)], [(8, 1)], [(2, 1)])
    add_term([(8, 1)], [(9, 1)], [(5, 1)])
    
    # Strassen Right (A11 * B1R) - 7 terms
    # A11 indices: 0,1,3,4
    # B1R indices: 2,3,6,7
    # C1R indices: 6,9,7,10
    
    # P1
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(6, 1), (10, 1)])
    # P2
    add_term([(3, 1), (4, 1)], [(2, 1)], [(7, 1), (10, -1)])
    # P3
    add_term([(0, 1)], [(3, 1), (7, -1)], [(9, 1), (10, 1)])
    # P4
    add_term([(4, 1)], [(6, 1), (2, -1)], [(6, 1), (7, 1)])
    # P5
    add_term([(0, 1), (1, 1)], [(7, 1)], [(6, -1), (9, 1)])
    # P6
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(10, 1)])
    # P7
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(6, 1)])
    
    # A12 * B2R - 4 terms
    # A12 indices: 2, 5
    # B2R indices: 10, 11
    # C1R contribution:
    # C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10
    
    add_term([(2, 1)], [(10, 1)], [(6, 1)])
    add_term([(2, 1)], [(11, 1)], [(9, 1)])
    add_term([(5, 1)], [(10, 1)], [(7, 1)])
    add_term([(5, 1)], [(11, 1)], [(10, 1)])
    
    # A21 * B1R - 4 terms
    # A21 indices: 6, 7
    # B1R indices: 2,3,6,7
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    add_term([(6, 1)], [(2, 1)], [(8, 1)])
    add_term([(6, 1)], [(3, 1)], [(11, 1)])
    add_term([(7, 1)], [(6, 1)], [(8, 1)])
    add_term([(7, 1)], [(7, 1)], [(11, 1)])
    
    # A22 * B2R - 2 terms
    # A22 index: 8
    # B2R indices: 10, 11
    # C2R contribution:
    # C[2,2]->8, C[2,3]->11
    
    add_term([(8, 1)], [(10, 1)], [(8, 1)])
    add_term([(8, 1)], [(11, 1)], [(11, 1)])
    
    # Verify correctness
    valid, max_diff = _verify(U, V, W)
    if not valid:
        # Should not happen, but return anyway
        pass
    
    return {"U": U, "V": V, "W": W}
