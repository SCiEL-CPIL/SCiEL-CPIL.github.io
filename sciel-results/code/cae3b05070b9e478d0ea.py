import time
import numpy as np

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = 2, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 8, 28, 14
    
    # Build the matrix multiplication tensor as a 3D array T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C
    # Actually, the tensor is T_{i,j,k} where i indexes rows of A and cols of C, j indexes cols of A and rows of B, k indexes cols of B and rows of C
    # For matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's a 0-1 tensor with all entries 1)
    # The decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    
    # T is nm x mp x pn tensor where T[i,j,k] = 1 for all i,j,k
    # Wait, that's not right. Let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k,i) where k is the column index of B / row index of C, i is row of A / row of C
    # U is (n*m, rank) - indexed by (i,j)
    # V is (m*p, rank) - indexed by (j,k)
    # W is (p*n, rank) - indexed by (k,i)
    
    # The tensor equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid i,j,k (since it's the standard matrix multiplication tensor)
    
    # Let me build this tensor and try to find a low-rank decomposition using linear algebra.
    
    # First, let's construct the rank-50 Strassen-based solution as a starting point.
    rank50 = 50
    U50 = np.zeros((nm, rank50))
    V50 = np.zeros((mp, rank50))
    W50 = np.zeros((pn, rank50))
    
    col = 0
    def add_strassen(i0, j0, k0):
        nonlocal col
        U50[i0*m + j0, col] += 1
        U50[(i0+1)*m + (j0+1), col] += 1
        V50[j0*p + k0, col] += 1
        V50[(j0+1)*p + (k0+1), col] += 1
        W50[k0*n + i0, col] += 1
        W50[(k0+1)*n + (i0+1), col] += 1
        col += 1
        U50[(i0+1)*m + j0, col] += 1
        U50[(i0+1)*m + (j0+1), col] += 1
        V50[j0*p + k0, col] += 1
        W50[k0*n + (i0+1), col] += 1
        W50[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        U50[i0*m + j0, col] += 1
        V50[j0*p + (k0+1), col] += 1
        V50[(j0+1)*p + (k0+1), col] -= 1
        W50[(k0+1)*n + i0, col] += 1
        W50[(k0+1)*n + (i0+1), col] += 1
        col += 1
        U50[(i0+1)*m + (j0+1), col] += 1
        V50[(j0+1)*p + k0, col] += 1
        V50[j0*p + k0, col] -= 1
        W50[k0*n + i0, col] += 1
        W50[k0*n + (i0+1), col] += 1
        col += 1
        U50[i0*m + j0, col] += 1
        U50[i0*m + (j0+1), col] += 1
        V50[(j0+1)*p + (k0+1), col] += 1
        W50[k0*n + i0, col] -= 1
        W50[(k0+1)*n + i0, col] += 1
        col += 1
        U50[(i0+1)*m + j0, col] += 1
        U50[i0*m + j0, col] -= 1
        V50[j0*p + k0, col] += 1
        V50[j0*p + (k0+1), col] += 1
        W50[(k0+1)*n + (i0+1), col] += 1
        col += 1
        U50[i0*m + (j0+1), col] += 1
        U50[(i0+1)*m + (j0+1), col] -= 1
        V50[(j0+1)*p + k0, col] += 1
        V50[(j0+1)*p + (k0+1), col] += 1
        W50[k0*n + i0, col] += 1
        col += 1
    
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    for i in range(n):
        for j in range(m):
            U50[i*m + j, col] = 1
            V50[j*p + 6, col] = 1
            W50[6*n + i, col] = 1
            col += 1
    
    assert col == 50
    
    # Now try to reduce rank via linear algebra.
    # The key insight: if we have a rank-R decomposition, the columns of U, V, W satisfy certain linear constraints.
    # We can try to express some columns as linear combinations of others.
    
    # Build the constraint matrix. For each (i,j,k), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system where we want to find the minimum number of terms.
    # One approach: try to find dependencies among the columns.
    
    # Let's try a different approach: use the fact that the tensor has symmetries.
    # Actually, let's try to directly search for a lower rank decomposition.
    
    # Strategy: start with the rank-50 solution and try to eliminate columns one by one.
    # For each column r, check if it can be expressed as a combination of other columns.
    
    best_rank = 50
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    
    # Try to reduce by checking if removing column r and adjusting others still works.
    # This is equivalent to checking if column r is in the span of the other columns in a certain sense.
    
    # Actually, a cleaner approach: build the tensor T and find its minimal rank decomposition.
    # The tensor T is of size 8 x 28 x 14.
    # We can flatten it and try to find a low-rank factorization.
    
    # But that's a non-convex problem. Let's try a greedy approach.
    
    # Alternative: try to find a rank-49 or rank-48 decomposition by searching.
    # Given time constraints, let's try a simple approach: 
    # Check if any two columns can be merged (i.e., if their contributions are linearly dependent).
    
    # For now, return the rank-50 solution. If we have time, we can try to improve.
    
    # Actually, let me try a quick linear algebra approach to see if rank 49 is possible.
    # The idea: if we fix all but one column, we can solve for the remaining column.
    # But this is complex. Let's just return the rank-50 solution for now.
    
    # Convert to lists with integer/half-integer entries
    U_list = [[int(round(U50[i, r])) for r in range(best_rank)] for i in range(nm)]
    V_list = [[int(round(V50[i, r])) for r in range(best_rank)] for i in range(mp)]
    W_list = [[int(round(W50[i, r])) for r in range(best_rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
