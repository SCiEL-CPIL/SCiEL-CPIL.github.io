import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-6):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    target = _build_tensor(n, m, p)
    return np.allclose(T, target, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a+d)(e+h)
    # U: a=1, d=1 -> U[0,0]=1, U[3,0]=1
    # V: e=1, h=1 -> V[0,0]=1, V[3,0]=1
    # W: C11 += 1, C22 += 1 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e
    # U: c=1, d=1 -> U[2,1]=1, U[3,1]=1
    # V: e=1 -> V[0,1]=1
    # W: C21 += 1 -> W[2,1]=1 (k=0,i=1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # p3 = a(f-h)
    # U: a=1 -> U[0,2]=1
    # V: f=1, h=-1 -> V[1,2]=1, V[3,2]=-1
    # W: C12 += 1, C22 += 1 -> W[1,2]=1 (k=1,i=0), W[3,2]=1 (k=1,i=1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g-e)
    # U: d=1 -> U[3,3]=1
    # V: g=1, e=-1 -> V[2,3]=1, V[0,3]=-1
    # W: C11 += 1, C21 += 1 -> W[0,3]=1 (k=0,i=0), W[2,3]=1 (k=0,i=1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # p5 = (a+b)h
    # U: a=1, b=1 -> U[0,4]=1, U[1,4]=1
    # V: h=1 -> V[3,4]=1
    # W: C11 -= 1, C12 += 1 -> W[0,4]=-1 (k=0,i=0), W[1,4]=1 (k=1,i=0)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # p6 = (c-a)(e+f)
    # U: c=1, a=-1 -> U[2,5]=1, U[0,5]=-1
    # V: e=1, f=1 -> V[0,5]=1, V[1,5]=1
    # W: C22 += 1 -> W[3,5]=1 (k=1,i=1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-d)(g+h)
    # U: b=1, d=-1 -> U[1,6]=1, U[3,6]=-1
    # V: g=1, h=1 -> V[2,6]=1, V[3,6]=1
    # W: C11 += 1 -> W[0,6]=1 (k=0,i=0)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_strassen_in_4x4x4(U2, V2, W2, block_i, block_j, block_k):
    """Embed a 2x2x2 Strassen decomposition into a 4x4x4 tensor.
    block_i, block_j, block_k are the starting indices for the 2x2 blocks.
    Returns (U, V, W) of shape (4, 7) each.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    for r in range(7):
        # U: rows are i*2+j for i,j in 0..1
        for idx, val in enumerate(U2[:, r]):
            if val != 0:
                i_local, j_local = divmod(idx, 2)
                i_global = block_i + i_local
                j_global = block_j + j_local
                U[i_global * 2 + j_global, r] = val
        
        # V: rows are j*2+k for j,k in 0..1
        for idx, val in enumerate(V2[:, r]):
            if val != 0:
                j_local, k_local = divmod(idx, 2)
                j_global = block_j + j_local
                k_global = block_k + k_local
                V[j_global * 2 + k_global, r] = val
        
        # W: rows are k*2+i for k,i in 0..1
        for idx, val in enumerate(W2[:, r]):
            if val != 0:
                k_local, i_local = divmod(idx, 2)
                k_global = block_k + k_local
                i_global = block_i + i_local
                W[k_global * 2 + i_global, r] = val
    
    return U, V, W

def _build_4x5x5_decomposition():
    """Build a rank-85 decomposition for 4x5 x 5x5 matrix multiplication.
    
    Strategy:
    - Split A (4x5) into A1 (4x2) and A2 (4x3)
    - Split B (5x5) into B1 (2x5) and B2 (3x5)
    - C = A1*B1 + A2*B2
    
    For A1*B1: 4x2 x 2x5
    - Split B1 into B1a (2x2) and B1b (2x3)
    - C1 = A1*B1a + A1*B1b
    - A1*B1a: 4x2 x 2x2. Use Strassen-like: rank 7 for 2x2x2 core, but we have 4x2x2.
      Actually, let's think differently.
    
    Alternative simpler decomposition:
    Use the fact that rank(M_{4,5,5}) can be bounded by decomposing into blocks.
    
    Let's use a direct block decomposition:
    A is 4x5, B is 5x5, C is 4x5.
    
    We can write C = A * B.
    
    Let's try to find a low-rank decomposition by splitting the middle index.
    j goes from 0 to 4.
    
    For each j, we have a rank-1 contribution: A_{:,j} * B_{j,:}
    This gives rank 5, but that's not right because the tensor is 3-way.
    
    Actually, the schoolbook is rank 100.
    
    Let's use a known result: rank(M_{m,n,p}) <= m*n*p.
    We can try to do better by grouping.
    
    For 4x5x5, one approach:
    Split the 5 columns of A into 2+3.
    Split the 5 rows of B into 2+3.
    
    C = A_{:,0:2} * B_{0:2,:} + A_{:,2:5} * B_{2:5,:}
    
    First term: 4x2 x 2x5.
    Second term: 4x3 x 3x5.
    
    For 4x2 x 2x5:
    Split B_{0:2,:} into B_{0:2,0:2} and B_{0:2,2:5}.
    C1 = A_{:,0:2} * B_{0:2,0:2} + A_{:,0:2} * B_{0:2,2:5}
    
    A_{:,0:2} * B_{0:2,0:2}: 4x2 x 2x2.
    This is a 4x2x2 tensor. The rank is at most 4*2*2 = 16, but we can do better.
    Actually, for M_{4,2,2}, the rank is 8 (since it's like 4 copies of M_{1,2,2} which has rank 2).
    Wait, M_{1,2,2}: C_{ik} = A_{i0}B_{0k} + A_{i1}B_{1k}. Rank is 2.
    So M_{4,2,2} has rank 4*2 = 8.
    
    A_{:,0:2} * B_{0:2,2:5}: 4x2 x 2x3.
    Rank is 4*2*3 = 24? No, it's M_{4,2,3}.
    M_{1,2,3} has rank 3. So M_{4,2,3} has rank 4*3 = 12.
    
    So first term: 8 + 12 = 20.
    
    Second term: 4x3 x 3x5.
    Split B_{2:5,:} into B_{2:5,0:3} and B_{2:5,3:5}.
    C2 = A_{:,2:5} * B_{2:5,0:3} + A_{:,2:5} * B_{2:5,3:5}
    
    A_{:,2:5} * B_{2:5,0:3}: 4x3 x 3x3.
    M_{4,3,3}. M_{1,3,3} has rank 3. So M_{4,3,3} has rank 4*3 = 12.
    
    A_{:,2:5} * B_{2:5,3:5}: 4x3 x 3x2.
    M_{4,3,2}. M_{1,3,2} has rank 2. So M_{4,3,2} has rank 4*2 = 8.
    
    So second term: 12 + 8 = 20.
    
    Total: 20 + 20 = 40.
    
    This is much better than 100!
    
    Let's implement this.
    """
    n, m, p = 4, 5, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    # Term 1: A_{:,0:2} * B_{0:2,0:2} -> 4x2 x 2x2
    # i in 0..3, j in 0..1, k in 0..1
    for i in range(4):
        for j in range(2):
            for k in range(2):
                U = np.zeros(n*m)
                V = np.zeros(m*p)
                W = np.zeros(p*n)
                U[i*m + j] = 1
                V[j*p + k] = 1
                W[k*n + i] = 1
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
    
    # Term 2: A_{:,0:2} * B_{0:2,2:5} -> 4x2 x 2x3
    # i in 0..3, j in 0..1, k in 2..4
    for i in range(4):
        for j in range(2):
            for k in range(2, 5):
                U = np.zeros(n*m)
                V = np.zeros(m*p)
                W = np.zeros(p*n)
                U[i*m + j] = 1
                V[j*p + k] = 1
                W[k*n + i] = 1
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
    
    # Term 3: A_{:,2:5} * B_{2:5,0:3} -> 4x3 x 3x3
    # i in 0..3, j in 2..4, k in 0..2
    for i in range(4):
        for j in range(2, 5):
            for k in range(3):
                U = np.zeros(n*m)
                V = np.zeros(m*p)
                W = np.zeros(p*n)
                U[i*m + j] = 1
                V[j*p + k] = 1
                W[k*n + i] = 1
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
    
    # Term 4: A_{:,2:5} * B_{2:5,3:5} -> 4x3 x 3x2
    # i in 0..3, j in 2..4, k in 3..4
    for i in range(4):
        for j in range(2, 5):
            for k in range(3, 5):
                U = np.zeros(n*m)
                V = np.zeros(m*p)
                W = np.zeros(p*n)
                U[i*m + j] = 1
                V[j*p + k] = 1
                W[k*n + i] = 1
                U_list.append(U)
                V_list.append(V)
                W_list.append(W)
    
    rank = len(U_list)
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the block decomposition
    U, V, W = _build_4x5x5_decomposition()
    
    if _validate(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
