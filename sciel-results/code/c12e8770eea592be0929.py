import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Trivial solution (rank 90)
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial), dtype=int)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=int)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    # Strassen factors for 2x2x2 (rank 7)
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)

    # m1 = (a00+a11)(b00+b11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # m2 = (a10+a11)b00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1
    # m3 = a00(b01-b11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1
    # m4 = a11(b10-b00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[1, 3] = 1
    # m5 = (a00+a01)b11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1
    # m6 = (a10-a00)(b00+b01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    # m7 = (a01-a11)(b10+b11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Verify Strassen locally
    Ur_s = U_s.reshape(2, 2, 7)
    Vr_s = V_s.reshape(2, 2, 7)
    Wr_s = W_s.reshape(2, 2, 7)
    T_s = np.einsum('ijr,jkr,kir->ijk', Ur_s, Vr_s, Wr_s)
    if not np.allclose(T_s, np.ones((2, 2, 2)), atol=1e-9):
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}

    # Block decomposition using Strassen
    # Cover i in {0,1} with 2x2x2 Strassen blocks
    # j in {0,1} and {2,3}, k in {0,1}, {2,3}, {4,5}
    # That's 2 * 3 = 6 Strassen blocks, each rank 7 = 42 rank
    
    # Remainder 1: i in {0,1}, j=4, k in {0..5}
    # For each k, we need C[i,k] += A[i,4]*B[4,k]
    # We can use rank 1 for each k? No, we need to sum over i.
    # Actually, for fixed j=4, C[i,k] += A[i,4]*B[4,k]
    # This is an outer product of A[:,4] and B[4,:], which is rank 1 in the (i,k) space.
    # But in our tensor decomposition, we add rank 1 terms.
    # U: i* m + j -> for i in {0,1}, j=4: rows are 0*5+4=4, 1*5+4=9
    # V: j* p + k -> for j=4, k in 0..5: rows are 4*6+k = 24+k
    # W: k* n + i -> for k in 0..5, i in {0,1}: rows are k*3+i
    
    # Let's think: C[i,k] += A[i,4]*B[4,k]
    # We can add rank 1 terms for each k? No, that would be rank 6.
    # Or rank 1 total? U[i*5+4, r] = A[i,4], V[4*6+k, r] = B[4,k], W[k*3+i, r] = 1?
    # Then sum_r U[i*5+4,r] V[4*6+k,r] W[k*3+i,r] = U[i*5+4,r] V[4*6+k,r]
    # If we set r as a single index, U[i*5+4, r] = A[i,4], V[4*6+k, r] = B[4,k], W[k*3+i, r] = 1
    # But U, V, W must be fixed matrices, not depending on A, B.
    # So we need to represent the multiplication of the column A[:,4] by row B[4,:].
    # This is a rank-1 matrix in the (i,k) space.
    # In the tensor decomposition, we add one rank-1 term:
    # U[i*5+4, r] = 1 for i in {0,1}, V[4*6+k, r] = 1 for k in 0..5, W[k*3+i, r] = 1
    # Then the contribution is sum_i' sum_k' ... wait no.
    # The contribution to C[i,k] is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For this to equal A[i,4]*B[4,k], we need the sum over r to give the product.
    # If we have one rank-1 term, U[i*5+4, r] * V[4*6+k, r] * W[k*3+i, r] = A[i,4]*B[4,k]
    # This is only possible if U[i*5+4, r] depends on A, which it doesn't.
    # So we need a decomposition that works for all A, B.
    # The standard way: for each k, add a rank-1 term where U[i*5+4, r_k] = 1, V[4*6+k, r_k] = 1, W[k*3+i, r_k] = 1 for i in {0,1}
    # But that would be rank 6 (one for each k).
    # Alternatively, we can use the fact that the tensor for "multiply column by row" has rank equal to the dimension of the outer product, which is... 
    # Actually, the tensor T[i,k] = x[i]*y[k] is rank 1. But here x and y are variable.
    # In the matrix multiplication tensor, the sub-tensor for fixed j is T[i,j,k] = A[i,j]*B[j,k]
    # This sub-tensor has rank 1 in the (i,k) space for each j, but in the full tensor it contributes rank equal to the number of (i,k) pairs? No.
    # Let's go back to basics. The trivial decomposition uses one rank-1 term for each (i,j,k) triple.
    # For fixed j=4, there are 2*6=12 triples (i,k) pairs.
    # Can we do better? The sub-tensor is sum over i,k of A[i,4]B[4,k] e_i^I e_k^K
    # This is an outer product of the vector (A[0,4], A[1,4]) and (B[4,0], ..., B[4,5])
    # As a tensor in I x K space, it's rank 1. But in the full decomposition, we're summing over r.
    # So we can represent this with rank 1? No, because U, V, W are fixed.
    # We need: sum_r U[i*5+4, r] V[4*6+k, r] W[k*3+i, r] = 1 (for all i, k)
    # Wait no, we need it to be A[i,4]*B[4,k]. But A and B are not in the equation.
    # I'm confusing myself. The tensor decomposition is for the multiplication map.
    # The map is: (A, B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor T is defined by T[i,j,k] = 1.
    # We want T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # So for j=4, we need T[i,4,k] = 1 = sum_r U[i*5+4, r] V[4*6+k, r] W[k*3+i, r]
    # This is a constraint on U, V, W for the entries where j=4.
    # The sub-matrix of the tensor for j=4 is a 2x6 matrix of ones.
    # We need to decompose this 2x6 matrix of ones as a sum of rank-1 matrices (from the perspective of the (i,k) indices), where each rank-1 matrix is of the form u_i * v_k * w_{k,i}
    # But w depends on both k and i, so it's not a standard matrix decomposition.
    # Let's just use the trivial decomposition for the remainder parts to be safe, or find a better decomposition.
    
    # Let's use a different approach: just use Strassen for the 2x5x6 part (i in {0,1})
    # 2x5x6: We can split j into {0,1}, {2,3}, {4}
    # For j in {0,1}: 2x2x6. Split k into {0,1}, {2,3}, {4,5}. Each is 2x2x2, use Strassen. Rank 3*7=21
    # For j in {2,3}: same, rank 21
    # For j=4: 2x1x6. This is C[i,k] += A[i,4]*B[4,k]
    # For this, we need T[i,4,k] = 1 for i in {0,1}, k in 0..5
    # We can use rank 6: for each k, one rank-1 term
    # U[i*5+4, r_k] = 1, V[4*6+k, r_k] = 1, W[k*3+i, r_k] = 1 for i in {0,1}
    # But wait, if W[k*3+i, r_k] = 1 for both i=0 and i=1, then the term contributes to both C[0,k] and C[1,k].
    # Let's check: sum_r U[i*5+4, r] V[4*6+k, r] W[k*3+i, r]
    # For r = r_k (corresponding to column k): U[i*5+4, r_k] = 1, V[4*6+k, r_k] = 1, W[k*3+i, r_k] = 1
    # So the contribution is 1*1*1 = 1. Good.
    # But we need to make sure that for other r, the contribution is 0.
    # So for r != r_k, at least one of U[i*5+4, r], V[4*6+k, r], W[k*3+i, r] must be 0.
    # This is doable. Rank for this part: 6.
    
    # Total for i in {0,1}: 21 + 21 + 6 = 48
    
    # For i=2: 1x5x6. C[2,k] = sum_j A[2,j]*B[j,k]
    # We need T[2,j,k] = 1 for j in 0..4, k in 0..5
    # This is a 5x6 matrix of ones.
    # We can use rank 6: for each k, one rank-1 term
    # U[2*5+j, r_k] = 1, V[j*6+k, r_k] = 1, W[k*3+2, r_k] = 1
    # Rank 6.
    
    # Total rank: 48 + 6 = 54. Still not great.
    
    # Let me try to be smarter about the remainder.
    # Actually, let's just use the trivial decomposition for the parts not covered by Strassen.
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}: covered by 6 Strassen blocks, rank 42
    # i in {0,1}, j=4, k in {0,1,2,3,4,5}: 12 terms in trivial, can we do better?
    # i=2, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}: 30 terms in trivial
    
    # Let's just use trivial for the remainders for now and see if we can improve.
    # Rank = 42 + 12 + 30 = 84. Same as before.
    
    # Let's try to use a better decomposition for the 2x1x6 and 1x5x6 parts.
    # For 2x1x6: T[i,4,k] = 1. We can use rank 2? 
    # If we use rank 2, we need sum_{r=1}^2 U[i*5+4, r] V[4*6+k, r] W[k*3+i, r] = 1
    # Let U[0*5+4, r] = u_i_r, U[1*5+4, r] = u'_i_r (for i=0,1)
    # This seems hard to get below rank 6 with the constraint that W depends on (k,i).
    
    # Let's just go with rank 84 for now and verify it works.
    
    rank_block = 84
    U_block = np.zeros((n * m, rank_block), dtype=int)
    V_block = np.zeros((m * p, rank_block), dtype=int)
    W_block = np.zeros((p * n, rank_block), dtype=int)

    def add_block_2x2x2(i_off, j_off, k_off, r_start):
        for r in range(7):
            for ii in range(2):
                for jj in range(2):
                    val = U_s[ii*2+jj, r]
                    if val != 0:
                        U_block[(i_off+ii)*m + (j_off+jj), r_start+r] = val
            for jj in range(2):
                for kk in range(2):
                    val = V_s[jj*2+kk, r]
                    if val != 0:
                        V_block[(j_off+jj)*p + (k_off+kk), r_start+r] = val
            for kk in range(2):
                for ii in range(2):
                    val = W_s[kk*2+ii, r]
                    if val != 0:
                        W_block[(k_off+kk)*n + (i_off+ii), r_start+r] = val

    # 6 Strassen blocks covering i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    add_block_2x2x2(0, 0, 0, 0)
    add_block_2x2x2(0, 2, 0, 7)
    add_block_2x2x2(0, 0, 2, 14)
    add_block_2x2x2(0, 2, 2, 21)
    add_block_2x2x2(0, 0, 4, 28)
    add_block_2x2x2(0, 2, 4, 35)

    # Remainder 1: i in {0,1}, j=4, k in {0..5}, trivial rank 12
    col = 42
    for ii in range(2):
        for kk in range(6):
            U_block[ii*m + 4, col] = 1
            V_block[4*p + kk, col] = 1
            W_block[kk*n + ii, col] = 1
            col += 1

    # Remainder 2: i=2, j in {0..4}, k in {0..5}, trivial rank 30
    for jj in range(5):
        for kk in range(6):
            U_block[2*m + jj, col] = 1
            V_block[jj*p + kk, col] = 1
            W_block[kk*n + 2, col] = 1
            col += 1

    # Verify full decomposition
    Ur = U_block.reshape(n, m, rank_block)
    Vr = V_block.reshape(m, p, rank_block)
    Wr = W_block.reshape(p, n, rank_block)
    T_check = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)

    if np.allclose(T_check, np.ones((n, m, p)), atol=1e-9):
        return {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
    else:
        # Fallback to trivial
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}

# EVOLVE-BLOCK-END
