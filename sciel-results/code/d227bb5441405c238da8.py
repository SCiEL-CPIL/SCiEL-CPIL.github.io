import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # rank = 36
    rank = 36
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)

    # We decompose A (2x4) into A1 (2x2) and A2 (2x2)
    # B (4x5) into B1 (2x5) and B2 (2x5)
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is a 2x2 * 2x5 multiplication.
    # We further decompose B_i (2x5) into B_i1 (2x2) and B_i2 (2x3)
    # A_i * B_i = A_i * B_i1 + A_i * B_i2
    # A_i * B_i1 is 2x2 * 2x2 -> rank 7 (Strassen)
    # A_i * B_i2 is 2x2 * 2x3 -> rank 12 (Strassen for first 2 cols, schoolbook for last col? 
    # Wait. 2x2 * 2x3. 
    # Let B_i2 = [b1, b2, b3] (columns).
    # A * b1 is 2x1. A * b2 is 2x1. A * b3 is 2x1.
    # A * b1: rank 4? No, A is 2x2, b1 is 2x1. A*b1 = A * [u; v] = u*col1(A) + v*col2(A).
    # This is bilinear in A and b1.
    # Let's look at rank of M_{2,2,3}.
    # M_{2,2,3} = M_{2,2,2} \oplus M_{2,2,1}.
    # Rank(M_{2,2,2}) = 7.
    # Rank(M_{2,2,1}) = 4. (A 2x2 times a 2x1 vector. c1 = a11 b1 + a12 b2, c2 = a21 b1 + a22 b2. Rank 4).
    # So Rank(M_{2,2,3}) = 11?
    # Let's check. 
    # M_{2,2,3} tensor T_{i,j,k}. i in {1,2}, j in {1,2}, k in {1,2,3}.
    # T = sum_{j} e_i^A \otimes e_j^B \otimes (E_j^C)
    # Actually, let's just construct it directly.
    
    # Block 1: A1 (rows 0,1 of A, cols 0,1 of A) -> indices i in {0,1}, j in {0,1}
    # B1 (rows 0,1 of B, all cols) -> indices j in {0,1}, k in {0,1,2,3,4}
    # Product C11 = A1 * B1.
    # C11 contributes to C[i, k].
    
    # Block 2: A2 (rows 0,1 of A, cols 2,3 of A) -> indices i in {0,1}, j in {2,3}
    # B2 (rows 2,3 of B, all cols) -> indices j in {2,3}, k in {0,1,2,3,4}
    # Product C22 = A2 * B2.
    
    # We need to define the rank-1 terms for A1*B1 (2x2 * 2x5) and A2*B2 (2x2 * 2x5).
    # Let's construct a generic 2x2 * 2x5 decomposition.
    # Let A = [[a11, a12], [a21, a22]]
    # Let B = [[b11, b12, b13, b14, b15], [b21, b22, b23, b24, b25]]
    # C = A * B
    # c11 = a11 b11 + a12 b21
    # c12 = a11 b12 + a12 b22
    # c13 = a11 b13 + a12 b23
    # c14 = a11 b14 + a12 b24
    # c15 = a11 b15 + a12 b25
    # c21 = a21 b11 + a22 b21
    # c22 = a21 b12 + a22 b22
    # c23 = a21 b13 + a22 b23
    # c24 = a21 b14 + a22 b24
    # c25 = a21 b15 + a22 b25
    
    # We can split B into B_left (2x2) and B_right (2x3).
    # C_left = A * B_left (2x2) -> Rank 7 (Strassen)
    # C_right = A * B_right (2x3)
    # Let's check rank of M_{2,2,3}.
    # T_{2,2,3} = sum_{j=1}^2 sum_{k=1}^3 e_i \otimes e_j \otimes E_{i, j, k} ? No.
    # T_{i, j, k} = 1 if we multiply A[i,j] by B[j,k].
    # For M_{2,2,3}:
    # i in {0,1}, j in {0,1}, k in {0,1,2}.
    # C[0,0] = a00 b00 + a01 b10
    # C[0,1] = a00 b01 + a01 b11
    # C[0,2] = a00 b02 + a01 b12
    # C[1,0] = a10 b00 + a11 b10
    # C[1,1] = a10 b01 + a11 b11
    # C[1,2] = a10 b02 + a11 b12
    
    # Is rank(M_{2,2,3}) = 11?
    # Let's try to find a decomposition.
    # We know rank(M_{2,2,2}) = 7.
    # M_{2,2,3} = M_{2,2,2} \oplus M_{2,2,1}.
    # M_{2,2,1} rank is 4.
    # So rank(M_{2,2,3}) = 11.
    # Total rank for 2x2 * 2x5 is 7 + 11 = 18.
    # Two such blocks: 18 + 18 = 36.
    
    # Let's construct the rank-1 tensors for M_{2,2,2} (Strassen).
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    # U: (2x2) -> (2x1) vector of coeffs for A.
    # V: (2x2) -> (2x1) vector of coeffs for B.
    # W: (2x2) -> (2x1) vector of coeffs for C.
    # Wait, the interface is U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # For a sub-block, we map the local indices to global indices.
    
    # Strassen terms for 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1 - m4 + m5 - m7
    # c01 = m3 + m7
    # c10 = m2 + m4
    # c11 = m1 - m3 + m2 - m6
    
    # Let's define a helper to add a Strassen block.
    def add_strassen_block(u_offset, v_offset, w_offset, base_rank):
        # u_offset: (i, j) -> global U index
        # v_offset: (j, k) -> global V index
        # w_offset: (k, i) -> global W index
        # base_rank: starting column index in U, V, W
        
        # Term 1: (a00+a11) * (b00+b11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00=1, c11=1
        U[u_offset(0,0), base_rank] = 1
        U[u_offset(1,1), base_rank] = 1
        V[v_offset(0,0), base_rank] = 1
        V[v_offset(1,1), base_rank] = 1
        W[w_offset(0,0), base_rank] = 1
        W[w_offset(1,1), base_rank] = 1
        
        # Term 2: (a10+a11) * b00
        U[u_offset(1,0), base_rank+1] = 1
        U[u_offset(1,1), base_rank+1] = 1
        V[v_offset(0,0), base_rank+1] = 1
        W[w_offset(0,1), base_rank+1] = 1  # c10
        # c10 = m2 + m4. So W for m2 is c10?
        # Wait. c10 = m2 + m4.
        # c11 = m1 - m3 + m2 - m6.
        # Let's re-verify Strassen output mapping.
        # c00 = m1 - m4 + m5 - m7
        # c01 = m3 + m7
        # c10 = m2 + m4
        # c11 = m1 - m3 + m2 - m6
        
        # So W for term 2 (m2) should contribute to c10 and c11.
        W[w_offset(0,1), base_rank+1] = 1  # c10? No, w_offset(k, i). c is (i, k).
        # Global W index is k*n + i.
        # So w_offset(k, i) should return k*n + i.
        # Let's check the task: W[k*n+i, r].
        # So W index for C[i, k] is k*n + i.
        # In add_strassen_block, w_offset(k, i) should return the index for C[i, k].
        
        # Let's restart the Strassen term definitions with correct W mapping.
        # C[i, k] is at W[k*n + i].
        
        # Term 1 (m1):
        # W: c00=1, c11=1.
        # c00 is i=0, k=0 -> W[0*n + 0]
        # c11 is i=1, k=1 -> W[1*n + 1]
        W[w_offset(0,0), base_rank] = 1
        W[w_offset(1,1), base_rank] = 1
        
        # Term 2 (m2):
        # c10 (i=1, k=0) -> W[0*n + 1]
        # c11 (i=1, k=1) -> W[1*n + 1]
        W[w_offset(0,1), base_rank+1] = 1
        W[w_offset(1,1), base_rank+1] = 1
        
        # Term 3 (m3):
        # a00 * (b01-b11)
        U[u_offset(0,0), base_rank+2] = 1
        V[v_offset(0,1), base_rank+2] = 1
        V[v_offset(1,1), base_rank+2] = -1
        # c01 (i=0, k=1) -> W[1*n + 0]
        # c11 (i=1, k=1) -> W[1*n + 1]
        W[w_offset(1,0), base_rank+2] = 1
        W[w_offset(1,1), base_rank+2] = -1
        
        # Term 4 (m4):
        # a11 * (b10-b00)
        U[u_offset(1,1), base_rank+3] = 1
        V[v_offset(1,0), base_rank+3] = 1
        V[v_offset(0,0), base_rank+3] = -1
        # c00 (i=0, k=0) -> W[0*n + 0]
        # c10 (i=1, k=0) -> W[0*n + 1]
        W[w_offset(0,0), base_rank+3] = -1
        W[w_offset(0,1), base_rank+3] = 1
        
        # Term 5 (m5):
        # (a00+a01) * b11
        U[u_offset(0,0), base_rank+4] = 1
        U[u_offset(0,1), base_rank+4] = 1
        V[v_offset(1,1), base_rank+4] = 1
        # c00 (i=0, k=0) -> W[0*n + 0]
        # c01 (i=0, k=1) -> W[1*n + 0]
        W[w_offset(0,0), base_rank+4] = 1
        W[w_offset(1,0), base_rank+4] = 1
        
        # Term 6 (m6):
        # (a10-a00) * (b00+b01)
        U[u_offset(1,0), base_rank+5] = 1
        U[u_offset(0,0), base_rank+5] = -1
        V[v_offset(0,0), base_rank+5] = 1
        V[v_offset(0,1), base_rank+5] = 1
        # c11 (i=1, k=1) -> W[1*n + 1]
        W[w_offset(1,1), base_rank+5] = -1
        
        # Term 7 (m7):
        # (a01-a11) * (b10+b11)
        U[u_offset(0,1), base_rank+6] = 1
        U[u_offset(1,1), base_rank+6] = -1
        V[v_offset(1,0), base_rank+6] = 1
        V[v_offset(1,1), base_rank+6] = 1
        # c00 (i=0, k=0) -> W[0*n + 0]
        # c01 (i=0, k=1) -> W[1*n + 0]
        W[w_offset(0,0), base_rank+6] = -1
        W[w_offset(1,0), base_rank+6] = 1
        
        return base_rank + 7

    # Now for M_{2,2,1} (A 2x2, B 2x1)
    # C = A * b
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # Rank 4.
    def add_rank4_block(u_offset, v_offset, w_offset, base_rank):
        # Term 1: a00 * b0
        U[u_offset(0,0), base_rank] = 1
        V[v_offset(0,0), base_rank] = 1
        W[w_offset(0,0), base_rank] = 1 # c0
        # Term 2: a01 * b1
        U[u_offset(0,1), base_rank+1] = 1
        V[v_offset(1,0), base_rank+1] = 1
        W[w_offset(0,0), base_rank+1] = 1 # c0
        # Term 3: a10 * b0
        U[u_offset(1,0), base_rank+2] = 1
        V[v_offset(0,0), base_rank+2] = 1
        W[w_offset(1,0), base_rank+2] = 1 # c1
        # Term 4: a11 * b1
        U[u_offset(1,1), base_rank+3] = 1
        V[v_offset(1,0), base_rank+3] = 1
        W[w_offset(1,0), base_rank+3] = 1 # c1
        return base_rank + 4

    # Now we assemble the 2x2 * 2x5 block.
    # B is 2x5. Split into B_left (2x2, k=0,1) and B_right (2x3, k=2,3,4).
    # For B_left, we use Strassen (rank 7).
    # For B_right, we need rank 11.
    # M_{2,2,3} = M_{2,2,2} \oplus M_{2,2,1}.
    # So we can use Strassen for the first 2 columns of B_right (k=2,3) and Rank4 for the last 1 column (k=4).
    # Wait, Strassen is for 2x2 * 2x2.
    # So for B_right (2x3), we can split into B_right1 (2x2, k=2,3) and B_right2 (2x1, k=4).
    # A * B_right1 is 2x2 * 2x2 -> Rank 7.
    # A * B_right2 is 2x2 * 2x1 -> Rank 4.
    # Total 11.
    
    def add_2x2x5_block(u_offset, v_offset, w_offset, base_rank):
        # u_offset(i, j) -> global U index
        # v_offset(j, k) -> global V index
        # w_offset(k, i) -> global W index
        
        # B_left: k=0,1
        def v_left(j, k):
            return v_offset(j, k)
        def w_left(k, i):
            return w_offset(k, i)
        # Note: w_offset expects global k. Here k is local 0,1.
        # We need to map local k to global k.
        # In the caller, v_offset and w_offset will handle the global mapping.
        # So here, j and k are local indices.
        
        r = add_strassen_block(u_offset, v_left, w_left, base_rank)
        
        # B_right1: k=2,3 (local k=0,1 in this sub-block)
        def v_right1(j, k_local):
            return v_offset(j, k_local + 2)
        def w_right1(k_local, i):
            return w_offset(k_local + 2, i)
        
        r = add_strassen_block(u_offset, v_right1, w_right1, r)
        
        # B_right2: k=4 (local k=0 in this sub-block)
        def v_right2(j, k_local):
            return v_offset(j, k_local + 4)
        def w_right2(k_local, i):
            return w_offset(k_local + 4, i)
        
        r = add_rank4_block(u_offset, v_right2, w_right2, r)
        
        return r

    # Block 1: A1 (i=0,1; j=0,1) * B1 (j=0,1; k=0,1,2,3,4)
    def u1(i, j):
        return i * m + j  # Global A index
    def v1(j, k):
        return j * p + k  # Global B index
    def w1(k, i):
        return k * n + i  # Global C index
        
    r = add_2x2x5_block(u1, v1, w1, 0)
    
    # Block 2: A2 (i=0,1; j=2,3) * B2 (j=2,3; k=0,1,2,3,4)
    def u2(i, j_local):
        return i * m + (j_local + 2)
    def v2(j_local, k):
        return (j_local + 2) * p + k
    def w2(k, i):
        return k * n + i
        
    r = add_2x2x5_block(u2, v2, w2, r)
    
    # r should be 36.
    assert r == rank, f"Rank mismatch: {r} vs {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
