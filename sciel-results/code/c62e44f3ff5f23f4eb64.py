"""Tensor decomposition for 2x3x8 matrix multiplication."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We use a constructive decomposition based on Strassen-like ideas.
    # The matrix A is 2x3 and B is 3x8.
    # We can view the multiplication as computing C = A @ B.
    # 
    # A standard schoolbook approach would take 2*3*8 = 48 multiplications.
    # We can do better by exploiting structure.
    #
    # One effective strategy is to split the middle dimension (3) into 2+1.
    # Let A = [A_left | A_right] where A_left is 2x2 and A_right is 2x1.
    # Let B = [B_top; B_bottom] where B_top is 2x8 and B_bottom is 1x8.
    # Then C = A_left @ B_top + A_right @ B_bottom.
    #
    # The term A_left @ B_top is a 2x2 matrix times a 2x8 matrix.
    # This can be viewed as 4 independent 2x2 matrix multiplications 
    # (one for each column pair of B_top? No, it's 2x2 * 2x8).
    # Actually, a 2x2 * 2x8 multiplication can be done with rank 7 * 4 = 28?
    # Let's check: R(2,2,2) = 7. A 2x2 * 2x8 is like 4 instances of 2x2*2x2?
    # No, B is 2x8. We can group the 8 columns into 4 groups of 2.
    # Each group is a 2x2 matrix. So we have 4 independent 2x2*2x2 multiplications.
    # Each takes rank 7. Total rank for this part = 4 * 7 = 28.
    #
    # The term A_right @ B_bottom is a 2x1 matrix times a 1x8 matrix.
    # This results in a 2x8 matrix.
    # This is 16 scalar multiplications (2 rows * 8 cols).
    # Can we do better? R(2,1,8).
    # The tensor for 2x1x8 has dimensions 2, 1, 8.
    # It's essentially the outer product of a vector of length 2 and a vector of length 8.
    # The rank of the all-ones tensor of size 2x1x8 is 2? No.
    # T[i,0,k] = 1.
    # Sum_t U[i,t] V[0,t] W[k,t] = 1.
    # If we set V[0,t]=1, then U[i,t] W[k,t] must sum to 1.
    # We can use 2 terms:
    # t=0: U[0,0]=1, U[1,0]=0, V[0,0]=1, W[0,0]=1, W[1,0]=1... wait W has size 8.
    # W[k,t] for k=0..7.
    # We need sum_t U[i,t] V[0,t] W[k,t] = 1 for all i,k.
    # If we use 2 terms:
    # t=0: U[0,0]=1, V[0,0]=1, W[0,0]=1, W[1,0]=1, ..., W[7,0]=1. Then contribution to C[0,k] is 1.
    # But we also need C[1,k]=1.
    # If U[1,0]=0, then C[1,k] gets 0 from t=0.
    # We need another term for row 1.
    # t=1: U[1,1]=1, V[0,1]=1, W[0,1]=1, ..., W[7,1]=1.
    # Then C[0,k] = 1 (from t=0) + 0 = 1.
    # C[1,k] = 0 + 1 (from t=1) = 1.
    # So R(2,1,8) = 2?
    # Let's check the tensor definition again.
    # T[i,j,k] = 1 if j=0. (Since m=1, only j=0 exists).
    # So T[i,0,k] = 1 for all i in {0,1}, k in {0..7}.
    # Decomposition:
    # t=0: U[0,0]=1, V[0,0]=1, W[k,0]=1 for all k.
    # t=1: U[1,1]=1, V[0,1]=1, W[k,1]=1 for all k.
    # Sum:
    # For i=0, j=0, k: U[0,0]V[0,0]W[k,0] + U[0,1]V[0,1]W[k,1] = 1*1*1 + 0 = 1. Correct.
    # For i=1, j=0, k: U[1,0]V[0,0]W[k,0] + U[1,1]V[0,1]W[k,1] = 0 + 1*1*1 = 1. Correct.
    # So yes, rank 2 suffices for the 2x1x8 part.
    #
    # Total rank = 28 + 2 = 30.
    #
    # Let's verify this construction carefully.
    
    rank = 30
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 1: A_left (2x2) @ B_top (2x8)
    # We split B_top into 4 blocks of 2 columns each.
    # B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # Each block is 2x2.
    # For each block, we use Strassen's algorithm for 2x2*2x2.
    
    # Strassen's algorithm for 2x2 matrices:
    # Given M1 (2x2) and M2 (2x2), compute M1 @ M2.
    # M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    # M1 @ M2 = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen's 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # Results:
    # [0,0] = p1 + p4 - p5 + p7
    # [0,1] = p3 + p5
    # [1,0] = p2 + p4
    # [1,1] = p1 - p2 + p3 + p6
    
    # We need to map these to the tensor decomposition.
    # The indices for U, V, W are:
    # U index: i*m + j, where i is row of A (0,1), j is col of A (0,1,2).
    # For A_left, j is 0,1. So U indices are 0,1,2,3.
    # V index: j*p + k, where j is row of B (0,1,2), k is col of B (0..7).
    # For B_top, j is 0,1. So V indices are 0..7 (for j=0) and 8..15 (for j=1).
    # W index: k*n + i, where k is col of C (0..7), i is row of C (0,1).
    # So W indices are 0..15.
    
    # Let's define the Strassen terms for a specific block.
    # Suppose we are processing block b (0..3), corresponding to columns 2b and 2b+1 of B.
    # Let col1 = 2b, col2 = 2b+1.
    # The submatrices are:
    # A_left = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_block = [[B[0,col1], B[0,col2]], [B[1,col1], B[1,col2]]]
    
    # The output C_block is 2x2, corresponding to C[:, col1:col2+1].
    # C[0, col1], C[0, col2], C[1, col1], C[1, col2].
    
    # We will add 7 terms to the decomposition for each block.
    
    for b in range(4):
        col1 = 2 * b
        col2 = 2 * b + 1
        
        # Base index for this block's terms in the rank dimension
        base = b * 7
        
        # Term 0: p1 = (A[0,0]+A[1,1]) * (B[0,col1]+B[1,col2])
        # Wait, Strassen formulas:
        # p1 = (a+d)(e+h) where a=A[0,0], d=A[1,1], e=B[0,col1], h=B[1,col2]
        # U: A[0,0] and A[1,1]
        # V: B[0,col1] and B[1,col2]
        # W: contributes to all 4 outputs with coefficients.
        
        # Let's map the contributions.
        # p1 contributes:
        # C[0, col1]: +1
        # C[0, col2]: +1
        # C[1, col1]: +1
        # C[1, col2]: -1
        
        t = base + 0
        U[0, t] = 1  # A[0,0]
        U[3, t] = 1  # A[1,1]
        V[col1, t] = 1  # B[0, col1]
        V[p + col2, t] = 1  # B[1, col2]
        W[col1 * n + 0, t] = 1  # C[0, col1]
        W[col2 * n + 0, t] = 1  # C[0, col2]
        W[col1 * n + 1, t] = 1  # C[1, col1]
        W[col2 * n + 1, t] = -1  # C[1, col2]
        
        # Term 1: p2 = (A[1,0]+A[1,1]) * B[0, col1]
        # p2 contributes:
        # C[1, col1]: +1
        # C[1, col2]: -1
        
        t = base + 1
        U[3, t] = 1  # A[1,0]
        U[4, t] = 1  # A[1,1]  # Wait, index for A[1,1] is 1*m+1 = 1*3+1 = 4.
        # My previous index mapping: U[i*m+j].
        # A[0,0] -> 0
        # A[0,1] -> 1
        # A[1,0] -> 3
        # A[1,1] -> 4
        # I used U[3] for A[1,1] in term 0. That was wrong.
        # Let's restart the index mapping.
        
        # U indices:
        # A[0,0] -> 0
        # A[0,1] -> 1
        # A[0,2] -> 2
        # A[1,0] -> 3
        # A[1,1] -> 4
        # A[1,2] -> 5
        
        # V indices:
        # B[0,0] -> 0
        # ...
        # B[0,k] -> k
        # B[1,0] -> 8
        # ...
        # B[1,k] -> 8+k
        # B[2,0] -> 16
        # ...
        # B[2,k] -> 16+k
        
        # W indices:
        # C[0,0] -> 0
        # C[0,1] -> 1
        # ...
        # C[0,k] -> k
        # C[1,0] -> 8
        # ...
        # C[1,k] -> 8+k
        
        # Let's redo the Strassen terms with correct indices.
        
        # Term 0: p1 = (A[0,0]+A[1,1]) * (B[0,col1]+B[1,col2])
        t = base + 0
        U[0, t] = 1  # A[0,0]
        U[4, t] = 1  # A[1,1]
        V[col1, t] = 1  # B[0, col1]
        V[p + col2, t] = 1  # B[1, col2]
        W[col1 * n + 0, t] = 1  # C[0, col1]
        W[col2 * n + 0, t] = 1  # C[0, col2]
        W[col1 * n + 1, t] = 1  # C[1, col1]
        W[col2 * n + 1, t] = -1  # C[1, col2]
        
        # Term 1: p2 = (A[1,0]+A[1,1]) * B[0, col1]
        t = base + 1
        U[3, t] = 1  # A[1,0]
        U[4, t] = 1  # A[1,1]
        V[col1, t] = 1  # B[0, col1]
        W[col1 * n + 1, t] = 1  # C[1, col1]
        W[col2 * n + 1, t] = -1  # C[1, col2]
        
        # Term 2: p3 = A[0,0] * (B[0, col2]-B[1, col2])
        t = base + 2
        U[0, t] = 1  # A[0,0]
        V[col2, t] = 1  # B[0, col2]
        V[p + col2, t] = -1  # B[1, col2]
        W[col2 * n + 0, t] = 1  # C[0, col2]
        
        # Term 3: p4 = A[1,1] * (B[1, col1]-B[0, col1])
        t = base + 3
        U[4, t] = 1  # A[1,1]
        V[p + col1, t] = 1  # B[1, col1]
        V[col1, t] = -1  # B[0, col1]
        W[col1 * n + 1, t] = 1  # C[1, col1]
        
        # Term 4: p5 = (A[0,0]+A[0,1]) * B[1, col2]
        t = base + 4
        U[0, t] = 1  # A[0,0]
        U[1, t] = 1  # A[0,1]
        V[p + col2, t] = 1  # B[1, col2]
        W[col2 * n + 0, t] = 1  # C[0, col2]
        
        # Term 5: p6 = (A[1,0]-A[0,0]) * (B[0, col1]+B[0, col2])
        t = base + 5
        U[3, t] = 1  # A[1,0]
        U[0, t] = -1  # A[0,0]
        V[col1, t] = 1  # B[0, col1]
        V[col2, t] = 1  # B[0, col2]
        W[col2 * n + 1, t] = 1  # C[1, col2]
        
        # Term 6: p7 = (A[0,1]-A[1,1]) * (B[1, col1]+B[1, col2])
        t = base + 6
        U[1, t] = 1  # A[0,1]
        U[4, t] = -1  # A[1,1]
        V[p + col1, t] = 1  # B[1, col1]
        V[p + col2, t] = 1  # B[1, col2]
        W[col1 * n + 0, t] = 1  # C[0, col1]
        
    # Part 2: A_right (2x1) @ B_bottom (1x8)
    # A_right = [[A[0,2]], [A[1,2]]]
    # B_bottom = [B[2,0], ..., B[2,7]]
    # C_right = A_right @ B_bottom
    # C[0,k] += A[0,2] * B[2,k]
    # C[1,k] += A[1,2] * B[2,k]
    
    # We established this can be done with rank 2.
    # t = 28: U[0,28]=1, V[16,28]=1, W[k,28]=1 for all k?
    # No, W has size 16. W[k*n + i, t].
    # For i=0, k=0..7, W[k, 28] = 1.
    # For i=1, k=0..7, W[8+k, 28] = 0.
    
    # t = 29: U[1,29]=1, V[16,29]=1, W[8+k, 29] = 1 for all k.
    
    t = 28
    U[2, t] = 1  # A[0,2]
    V[2*p, t] = 1  # B[2, 0] ... wait, V index is j*p + k.
    # B[2,k] index is 2*p + k.
    # We need to multiply by B[2,k] for all k.
    # But in the tensor decomposition, each term t has a single V vector.
    # V[:, t] is a vector of length 24.
    # If we set V[2*p+k, t] = 1 for all k, then the term is:
    # U[2,t] * (sum_k B[2,k]) * W[:,t]
    # This is not correct. We need A[0,2] * B[2,k] for each k separately.
    # The rank-2 construction I thought of earlier was:
    # t=0: U[0,0]=1, V[0,0]=1, W[k,0]=1. This computes A[0,0]*B[0,0]*1 + ...
    # No, that was for the tensor T[i,j,k]=1.
    # Here, the tensor entry is T[i, j, k] = 1 if j=2 (for A_right @ B_bottom).
    # Specifically, T[i, 2*p+k, k*n+i] = 1.
    # Wait, the tensor T is defined such that sum_t U[i,t] V[j,t] W[k,t] = T[i,j,k].
    # For the 2x1x8 part, T[i, 2*p+k, k*n+i] = 1.
    # Note that j is fixed to 2*p+k? No, j varies.
    # In the full 2x3x8 tensor, T[i, j, k] is non-zero only if the indices match the multiplication.
    # For A_right @ B_bottom:
    # i is 0 or 1.
    # j is 2*p + k' (where k' is the column index in B, 0..7).
    # k is k'*n + i.
    # So T[i, 2*p+k', k'*n+i] = 1.
    
    # Let's check the rank of this sub-tensor.
    # It has 2 * 8 * 1 = 16 non-zero entries.
    # Can it be decomposed with rank 2?
    # We need sum_t U[i,t] V[2*p+k',t] W[k'*n+i,t] = 1.
    # If we use 2 terms:
    # t=0: U[0,0]=1, V[2*p+k',0]=1 for all k', W[k'*n,0]=1 for all k'.
    # Then for i=0, sum_t U[0,t] V[2*p+k',t] W[k'*n+0,t] = 1 * 1 * 1 = 1.
    # For i=1, U[1,0]=0, so sum is 0. But we need 1.
    # So we need another term for i=1.
    # t=1: U[1,1]=1, V[2*p+k',1]=1 for all k', W[k'*n+1,1]=1 for all k'.
    # Then for i=1, sum is 1.
    # So yes, rank 2 works.
    
    t = 28
    U[2, t] = 1  # A[0,2]
    for k in range(p):
        V[2*p + k, t] = 1  # B[2, k]
        W[k * n + 0, t] = 1  # C[0, k]
        
    t = 29
    U[5, t] = 1  # A[1,2]
    for k in range(p):
        V[2*p + k, t] = 1  # B[2, k]
        W[k * n + 1, t] = 1  # C[1, k]
        
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        result = np.zeros((n*m, m*p, p*n), dtype=float)
        for t in range(rank):
            u = U[:, t]
            v = V[:, t]
            w = W[:, t]
            # Efficient outer product using broadcasting
            result += np.einsum('i,j,k->ijk', u, v, w)
        
        correct = np.zeros((n*m, m*p, p*n), dtype=float)
        for row_a in range(n):
            for col_a in range(m):
                for row_b in range(m):
                    for col_b in range(p):
                        if col_a == row_b:
                            i = row_a * m + col_a
                            j = row_b * p + col_b
                            k = col_b * n + row_a
                            correct[i, j, k] = 1
        
        return np.allclose(result, correct, atol=1e-10)
    
    is_valid = verify(U, V, W, n, m, p)
    if is_valid:
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
