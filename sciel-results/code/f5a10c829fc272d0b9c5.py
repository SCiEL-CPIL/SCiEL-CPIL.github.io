import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(result, target, atol=tol)

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Standard Strassen rank-7 for 2x2x2. W uses trace convention W[k*2+i]."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # U: 0=A00, 1=A01, 2=A10, 3=A11
    # V: 0=B00, 1=B01, 2=B10, 3=B11
    # W: 0=C00, 1=C01, 2=C10, 3=C11 (W[k*2+i])
    
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _build_rank_28():
    """
    Rank-28 for 2x3 x 3x5.
    
    Split middle dimension j={0,1,2} into j1={0,1} and j2={2}.
    Split output k={0,1,2,3,4} into k1={0,1}, k2={2,3}, k3={4}.
    
    Part A: A[:,{0,1}] * B[{0,1},:] (2x2 x 2x5)
      For k in {0,1}: 2x2x2 Strassen, rank 7
      For k in {2,3}: 2x2x2 Strassen, rank 7
      For k in {4}: 2x2x1, rank 2 (two separate products)
      Subtotal: 7+7+2 = 16
    
    Part B: A[:,{2}] * B[{2},:] (2x1 x 1x5)
      rank 5 (five separate products)
      Subtotal: 5
    
    Total: 16+5 = 21. Not 28.
    
    Let me try a different split to get exactly 28.
    Maybe use schoolbook for some parts.
    
    Alternative: 
    Part A: 2x2 x 2x5, use schoolbook for all: 2*2*5 = 20
    Part B: 2x1 x 1x5, schoolbook: 2*1*5 = 10
    Total: 30. That's the full schoolbook.
    
    To get 28, I need to save 2 from schoolbook.
    Using Strassen on one 2x2x2 block saves 1 (from 4 to 7? No, 4->7 increases).
    Wait, schoolbook for 2x2x2 is 4, Strassen is 7. Strassen is WORSE for 2x2x2!
    
    I had it backwards. Strassen helps for larger matrices, not 2x2x2.
    For 2x2x2, schoolbook rank 4 is optimal (actually rank is 7 for 2x2x2? No).
    
    Actually, the tensor rank of 2x2x2 matrix multiplication IS 7 (Strassen proved).
    Schoolbook gives 4 multiplications but that's 4 rank-1 terms? No!
    Schoolbook for 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k].
    That's 4 terms: (i=0,j=0,k=0), (i=0,j=1,k=0), (i=1,j=0,k=0), etc.
    Wait, 2*2*2 = 8 terms for schoolbook.
    Strassen reduces to 7.
    
    So for 2x2x2: schoolbook rank 8, Strassen rank 7. Saves 1.
    
    Now for our problem 2x3x5:
    Schoolbook rank = 2*3*5 = 30.
    
    If I split into sub-problems and use Strassen where applicable:
    Split j={0,1} and {2}, k={0,1} and {2,3} and {4}:
    
    Sub-problem 1: A[:,{0,1}] * B[{0,1},{0,1}] -> 2x2x2, Strassen rank 7 (saves 1 from 8)
    Sub-problem 2: A[:,{0,1}] * B[{0,1},{2,3}] -> 2x2x2, Strassen rank 7 (saves 1 from 8)
    Sub-problem 3: A[:,{0,1}] * B[{0,1},{4}] -> 2x2x1, schoolbook rank 4
    Sub-problem 4: A[:,{2}] * B[{2},{0,1}] -> 2x1x2, schoolbook rank 4
    Sub-problem 5: A[:,{2}] * B[{2},{2,3}] -> 2x1x2, schoolbook rank 4
    Sub-problem 6: A[:,{2}] * B[{2},{4}] -> 2x1x1, schoolbook rank 2
    
    Total: 7+7+4+4+4+2 = 28! 
    
    Now I need to construct this properly.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 6 rows: A[0,0],A[0,1],A[0,2],A[1,0],A[1,1],A[1,2]
    V = [[0]*rank for _ in range(m*p)]  # 15 rows: B[0,0]..B[0,4], B[1,0]..B[1,4], B[2,0]..B[2,4]
    W = [[0]*rank for _ in range(p*n)]  # 10 rows: C[0,0],C[0,1],C[1,0],C[1,1],...,C[4,0],C[4,1]
    
    col = 0
    
    # Sub-problem 1: A[:,{0,1}] * B[{0,1},{0,1}] -> C[{0,1},{0,1}]
    # 2x2x2 Strassen, rank 7
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    sU, sV, sW = _strassen_2x2x2()
    # Map: sU[0]->U[0], sU[1]->U[1], sU[2]->U[3], sU[3]->U[4]
    # sV[0]->V[0], sV[1]->V[1], sV[2]->V[5], sV[3]->V[6]
    # sW[0]->W[0], sW[1]->W[1], sW[2]->W[2], sW[3]->W[3]
    u_map = [0, 1, 3, 4]
    v_map = [0, 1, 5, 6]
    w_map = [0, 1, 2, 3]
    for r in range(7):
        for i in range(4):
            U[u_map[i]][col+r] += sU[i][r]
        for i in range(4):
            V[v_map[i]][col+r] += sV[i][r]
        for i in range(4):
            W[w_map[i]][col+r] += sW[i][r]
    col += 7  # col=7
    
    # Sub-problem 2: A[:,{0,1}] * B[{0,1},{2,3}] -> C[{2,3},{0,1}]
    # 2x2x2 Strassen, rank 7
    # A indices: same as above: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    # C indices: C[2,0]=4, C[2,1]=5, C[3,0]=6, C[3,1]=7
    u_map = [0, 1, 3, 4]
    v_map = [2, 3, 7, 8]
    w_map = [4, 5, 6, 7]
    for r in range(7):
        for i in range(4):
            U[u_map[i]][col+r] += sU[i][r]
        for i in range(4):
            V[v_map[i]][col+r] += sV[i][r]
        for i in range(4):
            W[w_map[i]][col+r] += sW[i][r]
    col += 7  # col=14
    
    # Sub-problem 3: A[:,{0,1}] * B[{0,1},{4}] -> C[{4},{0,1}]
    # 2x2x1, schoolbook rank 4
    # C[4,0] = A[0,0]*B[0,4] + A[1,0]*B[1,4]
    # C[4,1] = A[0,1]*B[0,4] + A[1,1]*B[1,4]
    # Terms:
    # T1: A[0,0]*B[0,4] -> C[4,0]: U[0], V[4], W[8]
    # T2: A[1,0]*B[1,4] -> C[4,0]: U[3], V[9], W[8]
    # T3: A[0,1]*B[0,4] -> C[4,1]: U[1], V[4], W[9]
    # T4: A[1,1]*B[1,4] -> C[4,1]: U[4], V[9], W[9]
    
    U[0][col]=1; V[4][col]=1; W[8][col]=1; col+=1
    U[3][col]=1; V[9][col]=1; W[8][col]=1; col+=1
    U[1][col]=1; V[4][col]=1; W[9][col]=1; col+=1
    U[4][col]=1; V[9][col]=1; W[9][col]=1; col+=1
    # col=18
    
    # Sub-problem 4: A[:,{2}] * B[{2},{0,1}] -> C[{0,1},{2}]
    # Wait, I need to be careful with indices.
    # A[:,{2}] means columns 2 of A: A[0,2]=2, A[1,2]=5
    # B[{2},{0,1}] means row 2 of B, columns 0,1: B[2,0]=10, B[2,1]=11
    # C[{0,1},{2}] means rows 0,1 of C, column 2: C[0,2]=2, C[1,2]=3
    # Wait, C indices: C[k*n+i] where k is row, i is column.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[2,0]=4, C[2,1]=5, ...
    # So C[0,2] would be... but C is 5x2 (p x n = 5x2).
    # C[k,i] where k in 0..4, i in 0..1.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[2,0]=4, C[2,1]=5, C[3,0]=6, C[3,1]=7, C[4,0]=8, C[4,1]=9
    
    # Sub-problem 4: A[:,{2}] * B[{2},{0,1}] -> C[{0,1},{2}]
    # But C is only 5x2, so column 2 doesn't exist! I made an error.
    
    # Let me reconsider. The output C is p x n = 5 x 2.
    # C[k,i] = sum_j B[j,k] * A[i,j] ... no.
    # C[i,k] = sum_j A[i,j] * B[j,k] where i in 0..1, k in 0..4.
    # W index is k*n+i = k*2+i.
    # So W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1], W[4]=C[2,0], W[5]=C[2,1], ...
    
    # The tensor entry T[a,b,c] = 1 where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # and j=j', i=i', k=k'.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    # For sub-problem 4: A[:,{2}] * B[{2},{0,1}]
    # This contributes to C[i,k] for i in {0,1}, k in {0,1}
    # C[i,k] += A[i,2] * B[2,k]
    # Terms:
    # T1: A[0,2]*B[2,0] -> C[0,0]: U[2], V[10], W[0]
    # T2: A[1,2]*B[2,0] -> C[1,0]: U[5], V[10], W[2]
    # T3: A[0,2]*B[2,1] -> C[0,1]: U[2], V[11], W[1]
    # T4: A[1,2]*B[2,1] -> C[1,1]: U[5], V[11], W[3]
    
    U[2][col]=1; V[10][col]=1; W[0][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[2][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[3][col]=1; col+=1
    # col=22
    
    # Sub-problem 5: A[:,{2}] * B[{2},{2,3}] -> C[{2,3},{0,1}]
    # C[i,k] += A[i,2] * B[2,k] for k in {2,3}
    # Terms:
    # T1: A[0,2]*B[2,2] -> C[2,0]: U[2], V[12], W[4]
    # T2: A[1,2]*B[2,2] -> C[3,0]: U[5], V[12], W[6]
    # T3: A[0,2]*B[2,3] -> C[2,1]: U[2], V[13], W[5]
    # T4: A[1,2]*B[2,3] -> C[3,1]: U[5], V[13], W[7]
    
    U[2][col]=1; V[12][col]=1; W[4][col]=1; col+=1
    U[5][col]=1; V[12][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[13][col]=1; W[5][col]=1; col+=1
    U[5][col]=1; V[13][col]=1; W[7][col]=1; col+=1
    # col=26
    
    # Sub-problem 6: A[:,{2}] * B[{2},{4}] -> C[{4},{0,1}]
    # C[i,4] += A[i,2] * B[2,4]
    # Terms:
    # T1: A[0,2]*B[2,4] -> C[4,0]: U[2], V[14], W[8]
    # T2: A[1,2]*B[2,4] -> C[4,1]: U[5], V[14], W[9]
    
    U[2][col]=1; V[14][col]=1; W[8][col]=1; col+=1
    U[5][col]=1; V[14][col]=1; W[9][col]=1; col+=1
    # col=28
    
    if col != rank:
        return None
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    result = _build_rank_28()
    if result is not None:
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    return _schoolbook(n, m, p)
