import time
import sys

n, m, p = 4, 4, 9
schoolbook_rank = n * m * p

START_TIME = time.time()
TIME_LIMIT = 115  # safety margin

def elapsed():
    return time.time() - START_TIME

def _time_ok():
    return elapsed() < TIME_LIMIT

def _schoolbook_decomp():
    """
    Return the schoolbook decomposition of rank n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {'U': U, 'V': V, 'W': W}

def _verify(U, V, W):
    """
    Verify that the decomposition is exact.
    For each term t, we have U[:,t] * V[:,t] * W[:,t]
    The tensor T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    We check that T[i,j,k] = 1 if i,j,k are valid indices, else 0.
    Actually, the tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all i,j,k.
    Wait, the tensor is T_{i,j,k} where i is row of A, j is col of A / row of B, k is col of B.
    The product C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T has entries T[i,j,k] = 1 for all i,j,k.
    So we need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 for all i,j,k.
    """
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # We can do this efficiently by iterating over terms
    T = [[[0] * p for _ in range(m)] for _ in range(n)]
    
    for t in range(rank):
        u_col = [U[idx][t] for idx in range(n * m)]
        v_col = [V[idx][t] for idx in range(m * p)]
        w_col = [W[idx][t] for idx in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    T[i][j][k] += u_val * v_val * w_val
    
    # Check if T[i,j,k] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i][j][k] != 1:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    U, V, W are 4x7, 4x7, 4x7.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen's algorithm
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map indices
    # A is 2x2: A[i,j] with i,j in {0,1}
    # B is 2x2: B[j,k] with j,k in {0,1}
    # C is 2x2: C[i,k] with i,k in {0,1}
    
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    # Term 0: M1
    # U: A11, A22 -> U[0,0]=1, U[3,0]=1
    # V: B11, B22 -> V[0,0]=1, V[3,0]=1
    # W: C11, C22 -> W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2
    # U: A21, A22 -> U[2,1]=1, U[3,1]=1
    # V: B11 -> V[0,1]=1
    # W: C21, C22 -> W[1,1]=1, W[3,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # Term 2: M3
    # U: A11 -> U[0,2]=1
    # V: B21, B22 -> V[2,2]=1, V[3,2]=-1
    # W: C12, C22 -> W[1,2]=1, W[3,2]=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: M4
    # U: A22 -> U[3,3]=1
    # V: B11, B21 -> V[0,3]=1, V[2,3]=-1
    # W: C11, C21 -> W[0,3]=1, W[1,3]=1
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: M5
    # U: A11, A12 -> U[0,4]=1, U[1,4]=1
    # V: B22 -> V[3,4]=1
    # W: C11, C12 -> W[0,4]=-1, W[1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: M6
    # U: A21, A11 -> U[2,5]=1, U[0,5]=-1
    # V: B11, B12 -> V[0,5]=1, V[1,5]=1
    # W: C22 -> W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7
    # U: A12, A22 -> U[1,6]=1, U[3,6]=-1
    # V: B21, B22 -> V[2,6]=1, V[3,6]=1
    # W: C11 -> W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_decomps(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions for M_{n,m,p1} and M_{n,m,p2} to get M_{n,m,p1+p2}.
    The U matrices are the same shape (n*m x rank1, n*m x rank2).
    The V matrices are (m*p1 x rank1, m*p2 x rank2).
    The W matrices are (p1*n x rank1, p2*n x rank2).
    
    For the combined problem M_{n,m,p1+p2}:
    U_combined = [U1 | U2]  (n*m x (rank1+rank2))
    V_combined = [V1; V2]  ((m*p1 + m*p2) x (rank1+rank2))
    W_combined = [W1; W2]  ((p1*n + p2*n) x (rank1+rank2))
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    
    nm = len(U1)
    mp1 = len(V1)
    mp2 = len(V2)
    pn1 = len(W1)
    pn2 = len(W2)
    
    mp = mp1 + mp2
    pn = pn1 + pn2
    
    U_comb = [[0] * r for _ in range(nm)]
    V_comb = [[0] * r for _ in range(mp)]
    W_comb = [[0] * r for _ in range(pn)]
    
    for i in range(nm):
        for t in range(r1):
            U_comb[i][t] = U1[i][t]
        for t in range(r2):
            U_comb[i][r1 + t] = U2[i][t]
    
    for j in range(mp1):
        for t in range(r1):
            V_comb[j][t] = V1[j][t]
    for j in range(mp2):
        for t in range(r2):
            V_comb[mp1 + j][r1 + t] = V2[j][t]
    
    for k in range(pn1):
        for t in range(r1):
            W_comb[k][t] = W1[k][t]
    for k in range(pn2):
        for t in range(r2):
            W_comb[pn1 + k][r1 + t] = W2[k][t]
    
    return U_comb, V_comb, W_comb

def _decomp_4x4x4():
    """
    Construct a rank-49 decomposition for M_{4,4,4} using Strassen.
    We view 4x4 matrices as 2x2 blocks of 2x2 matrices.
    The outer multiplication is M_{2,2,2} of blocks, each block multiplication is M_{2,2,2}.
    So we take the Kronecker product of the two Strassen decompositions.
    """
    U1, V1, W1 = _strassen_2x2x2()
    
    # The outer Strassen has rank 7, each term involves block multiplications.
    # For each term t in the outer Strassen, we need to multiply blocks.
    # The block multiplication is also M_{2,2,2} with rank 7.
    # So the total rank is 7 * 7 = 49.
    
    # Let's construct it properly.
    # Outer indices: I, J, K in {0,1} for blocks.
    # Inner indices: i, j, k in {0,1} for elements within blocks.
    # Global indices: I*2+i, J*2+j, K*2+k
    
    # U_outer[I*2+J, t_outer]
    # V_outer[J*2+K, t_outer]
    # W_outer[K*2+I, t_outer]
    
    # For each t_outer, we have a block multiplication.
    # The block multiplication has its own Strassen decomposition with terms t_inner.
    # U_inner[i*2+j, t_inner]
    # V_inner[j*2+k, t_inner]
    # W_inner[k*2+i, t_inner]
    
    # The combined term is (t_outer, t_inner).
    # U_combined[(I*2+i)*2 + (J*2+j), t] = U_outer[I*2+J, t_outer] * U_inner[i*2+j, t_inner]
    # V_combined[(J*2+j)*2 + (K*2+k), t] = V_outer[J*2+K, t_outer] * V_inner[j*2+k, t_inner]
    # W_combined[(K*2+k)*2 + (I*2+i), t] = W_outer[K*2+I, t_outer] * W_inner[k*2+i, t_inner]
    
    r_outer = 7
    r_inner = 7
    r = r_outer * r_inner
    
    U_comb = [[0] * r for _ in range(16)]
    V_comb = [[0] * r for _ in range(16)]
    W_comb = [[0] * r for _ in range(16)]
    
    for t_outer in range(r_outer):
        for t_inner in range(r_inner):
            t = t_outer * r_inner + t_inner
            for I in range(2):
                for J in range(2):
                    u_outer = U1[I*2+J][t_outer]
                    if u_outer == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            u_inner = U1[i*2+j][t_inner]
                            if u_inner == 0:
                                continue
                            u_val = u_outer * u_inner
                            if u_val != 0:
                                U_comb[(I*2+i)*2 + (J*2+j)][t] = u_val
            for J in range(2):
                for K in range(2):
                    v_outer = V1[J*2+K][t_outer]
                    if v_outer == 0:
                        continue
                    for j in range(2):
                        for k in range(2):
                            v_inner = V1[j*2+k][t_inner]
                            if v_inner == 0:
                                continue
                            v_val = v_outer * v_inner
                            if v_val != 0:
                                V_comb[(J*2+j)*2 + (K*2+k)][t] = v_val
            for K in range(2):
                for I in range(2):
                    w_outer = W1[K*2+I][t_outer]
                    if w_outer == 0:
                        continue
                    for k in range(2):
                        for i in range(2):
                            w_inner = W1[k*2+i][t_inner]
                            if w_inner == 0:
                                continue
                            w_val = w_outer * w_inner
                            if w_val != 0:
                                W_comb[(K*2+k)*2 + (I*2+i)][t] = w_val
    
    return U_comb, V_comb, W_comb

def _decomp_4x4x5():
    """
    Construct a rank-65 decomposition for M_{4,4,5}.
    We can use schoolbook for the first 4 columns (rank 64) and add 1 more column.
    Actually, let's just use schoolbook for M_{4,4,5} which has rank 80.
    But we want to improve.
    
    Alternatively, we can use the fact that M_{4,4,5} can be decomposed using
    a mix of Strassen and schoolbook.
    
    For now, let's use a simple approach:
    M_{4,4,5} = M_{4,4,4} + M_{4,4,1}
    M_{4,4,4} has rank 49 (from Strassen).
    M_{4,4,1} has rank 16 (schoolbook).
    Total rank = 49 + 16 = 65.
    """
    U4, V4, W4 = _decomp_4x4x4()
    
    # M_{4,4,1} schoolbook
    r1 = 16
    U1 = [[0] * r1 for _ in range(16)]
    V1 = [[0] * r1 for _ in range(4)]
    W1 = [[0] * r1 for _ in range(4)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i*4+j][col] = 1
            V1[j*1+0][col] = 1
            W1[0*4+i][col] = 1
            col += 1
    
    # Combine
    U_comb, V_comb, W_comb = _combine_decomps(U4, V4, W4, U1, V1, W1)
    
    return U_comb, V_comb, W_comb

def construct():
    """
    Construct a tensor decomposition for matrix multiplication (4,4,9).
    Returns {'U', 'V', 'W'} with integer entries.
    """
    # First, get the schoolbook decomposition as a fallback
    baseline = _schoolbook_decomp()
    
    # Try to construct a lower-rank decomposition
    # M_{4,4,9} = M_{4,4,4} + M_{4,4,5}
    # M_{4,4,4} has rank 49 (Strassen)
    # M_{4,4,5} has rank 65 (Strassen + schoolbook)
    # Total rank = 49 + 65 = 114
    
    try:
        if not _time_ok():
            return baseline
        
        U4, V4, W4 = _decomp_4x4x4()
        
        if not _time_ok():
            return baseline
        
        U5, V5, W5 = _decomp_4x4x5()
        
        if not _time_ok():
            return baseline
        
        U_comb, V_comb, W_comb = _combine_decomps(U4, V4, W4, U5, V5, W5)
        
        # Verify
        if _verify(U_comb, V_comb, W_comb):
            return {'U': U_comb, 'V': V_comb, 'W': W_comb}
        else:
            # Try an alternative: M_{4,4,9} = M_{4,4,3} + M_{4,4,3} + M_{4,4,3}
            # But for now, fall back to schoolbook
            return baseline
    except Exception as e:
        print(f"Error in improved construction: {e}", file=sys.stderr)
        return baseline

# Ensure the function is defined
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
