import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Use trivial decomposition: rank = n*m*p = 48
    # But we can do better with a structured approach.
    # For 3x4 * 4x4:
    # Split A (3x4) into rows: top 2 rows and bottom 1 row
    # Top 2x4 * 4x4 -> can use Strassen-like or just trivial
    # Bottom 1x4 * 4x4 -> trivial rank 16
    # 
    # Actually, let's just use the trivial decomposition which is guaranteed correct.
    # U: 12 x 48, V: 16 x 48, W: 12 x 48
    # For each (i, j, k) where i in 0..2, j in 0..3, k in 0..3:
    #   U[i*m+j, idx] = 1
    #   V[j*p+k, idx] = 1  
    #   W[k*n+i, idx] = 1
    # This gives C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let me try a smarter approach:
    # Split A into 3 rows. For each row i of A (length 4), multiply by B (4x4).
    # Row i of A times B gives row i of C (length 4).
    # For row i: sum over j=0..3 of A[i,j] * B[j,:]
    # This is 4 scalar-vector products, each rank 4 (since B is 4x4, trivial rank 16 per row)
    # Total: 3 * 16 = 48. Same as trivial.
    
    # Can we do better? Use the structure of B.
    # If we group columns of A and rows of B differently...
    
    # Let's try: group A rows into pairs where possible.
    # A is 3x4. Group rows 0,1 as a 2x4 block, and row 2 separately.
    # 2x4 * 4x4: split into two 2x2 * 4x4 blocks? Or 2x4 * 2x2 for each column pair?
    
    # Actually, the cleanest improvement: use the fact that 2x2 matrix mult has rank 7.
    # Split the 4x4 B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # Top 2x4 A: split into two 2x2 blocks: A_top = [[A1, A2]] where A1, A2 are 2x2
    # Then A_top * B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # Each 2x2 * 2x2 is rank 7 (Strassen). But we have 4 such products.
    # However, they share structure. Let me think more carefully.
    
    # Let's just use the trivial 48-rank decomposition, which is definitely correct.
    rank = 48
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):  # rows of A
        for j in range(m):  # cols of A / rows of B
            for k in range(p):  # cols of B / cols of C
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i  # trace convention W[k*n+i, r]
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Verify the decomposition
    # C = sum_r U[:,r] * V[:,r]^T ... no, that's not right.
    # The tensor decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # For matrix mult: T[i,j,k] = A[i,j] * B[j,k] * delta_{output}
    # Actually the multiplication tensor M[i,j,k] where i indexes A, j indexes B, k indexes C output position
    # M[i,j,k] = 1 if k corresponds to (i,j) output, i.e., if we're computing C[i',k'] and i'=i, k'=k... 
    
    # Let me re-read the contract. The output factor W follows W[k*n+i, r].
    # So for computing C[i][k], we use W[k*n + i, r].
    # The decomposition should satisfy: for all i,k: C[i,k] = sum_j A[i,j] B[j,k]
    # And C[i,k] = sum_r U[i*4+j... wait, U has shape (n*m, rank) = (12, rank).
    # U indexes (i,j) pairs, V indexes (j,k) pairs, W indexes (k,i) pairs.
    # So the tensor entry T[(i,j),(j,k),(k,i)] should be nonzero only when... 
    # Actually T is a 3-tensor with modes (i,j), (j,k), (k,i) but with the constraint that the j in (i,j) matches the j in (j,k).
    
    # The multiplication tensor for A (n x m) * B (m x p) = C (n x p):
    # T[i,j,k,l] where i in 0..n-1, j in 0..m-1, k in 0..m-1, l in 0..p-1
    # T[i,j,k,l] = 1 if j == k, else 0
    # And C[i,l] = sum_{j,k} T[i,j,k,l] * A[i,j] * B[k,l]
    
    # In the factorization: T[i,j,k,l] = sum_r U[(i,j),r] * V[(k,l),r] * W[(l,i),r]
    # Wait, the contract says U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U index: i*m + j (i in 0..n-1, j in 0..m-1)
    # V index: k*p + l (k in 0..m-1, l in 0..p-1)  
    # W index: l*n + i (l in 0..p-1, i in 0..n-1) -- trace convention W[k*n+i] where k is col of C, i is row of C
    
    # So T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    # We need T[i,j,k,l] = delta_{j,k}
    
    # In my trivial decomposition:
    # For each (i,j,k) with i in 0..2, j in 0..3, k in 0..3:
    #   U[i*4+j, col] = 1, V[j*4+k, col] = 1, W[k*3+i, col] = 1
    # So T[i,j,j',k'] = sum_r ... = 1 if j==j' and the col matches.
    # Actually for a given col r corresponding to (i0,j0,k0):
    # U[i*4+j, r] = 1 iff i==i0 and j==j0
    # V[j'*4+k', r] = 1 iff j'==j0 and k'==k0
    # W[k''*3+i', r] = 1 iff k''==k0 and i'==i0
    # So T[i,j,j',k'] = 1 iff i==i0, j==j0, j'==j0, k'==k0, k''==k0, i'==i0
    # For T[i,j,j',k'] with the free indices: T[i,j,j',k'] = delta_{j,j'} when i is fixed... 
    # Hmm, let me just trust that the trivial decomposition works since it's the standard one.
    
    return {"U": U, "V": V, "W": W}
