"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _validate(U, V, W, n=2, m=3, p=4):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_embedding():
    """Construct a decomposition by embedding 2x3 and 3x4 into 4x4 and using Strassen.
    
    Strategy:
    - Embed A(2x3) into A'(4x4): A'[i,j] = A[i,j] if i<2,j<3 else 0
    - Embed B(3x4) into B'(4x4): B'[j,k] = B[j,k] if j<3,k<4 else 0  
    - C'(4x4) = A'*B', and C(2x4) is top-left 2x4 of C'
    
    Use Strassen's 7-term decomposition for 4x4 multiplication.
    The 4x4 multiplication tensor T4[i,j,k]=1 for all i,j,k in 0..3
    decomposes as sum of 7 rank-1 tensors.
    
    For our sub-tensor T234[i,j,k]=1 for i<2,j<3,k<4, we restrict the
    Strassen decomposition to these indices.
    """
    n, m, p = 2, 3, 4
    
    # Strassen 2x2 decomposition (for 2x2 * 2x2 = 2x2)
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1  # W[k*2+i]: C00=k0i0=W[0], C11=k1i1=W[3]
    
    # M2 = (A10+A11)B00 -> C10+=M2
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1  # C10=k1i0=W[2]
    
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1  # C01=k0i1=W[1], C11=k1i1=W[3]
    
    # M4 = A11(B10-B00) -> C10+=M4, C00+=M4
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[2,3]=1; W2[0,3]=1  # C10=k1i0=W[2], C00=k0i0=W[0]
    
    # M5 = (A00+A01)B11 -> C01+=M5, C11-=M5
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[1,4]=1; W2[3,4]=-1  # C01=k0i1=W[1], C11=k1i1=W[3]
    
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1  # C11=k1i1=W[3]
    
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1  # C00=k0i0=W[0]
    
    # Now, 4x4 = 2x2 blocks. A4 = [[A00,A01],[A10,A11]] where each is 2x2
    # A4[i*2+a, j*2+b] = A[i,j] for i,j in 0,1 and a,b in 0,1
    # Similarly for B4 and C4.
    # 
    # For 4x4 multiplication using Strassen on 2x2 blocks:
    # A4 = [[A00, A01], [A10, A11]] (2x2 block matrix, each block 2x2)
    # B4 = [[B00, B01], [B10, B11]]
    # C4 = A4 * B4
    # 
    # C4[00] = A00*B00 + A01*B10
    # C4[01] = A00*B01 + A01*B11
    # C4[10] = A10*B00 + A11*B10
    # C4[11] = A10*B01 + A11*B11
    # 
    # Each of these is a 2x2*2x2=2x2 multiplication, using 7 Strassen terms.
    # Total: 4*7 = 28 terms. That's worse than schoolbook (24).
    # 
    # But we can be smarter: since A' has zeros in rows 2,3 and cols 3,
    # and B' has zeros in row 3, we can eliminate some terms.
    # 
    # Let me think differently. Direct construction for 2x3 * 3x4.
    # 
    # The output C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1; j=0,1,2; k=0,1,2,3
    # 
    # I'll try to find a low-rank decomposition by exploiting structure.
    # 
    # Actually, let me just do a more efficient search with better initialization.
    pass

def _efficient_search(rank, n=2, m=3, p=4, max_time=20):
    """More efficient search for rank-rank decomposition."""
    start = time.time()
    import random
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try multiple random restarts
    for attempt in range(50):
        if time.time() - start > max_time:
            break
        
        seed = attempt * 104729 + rank * 7
        random.seed(seed)
        
        # Initialize with sparse random values
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for c in range(rank):
            # Each column has a few non-zero entries
            u_row = random.randint(0, n*m-1)
            v_row = random.randint(0, m*p-1)
            w_row = random.randint(0, p*n-1)
            U[u_row, c] = random.choice(values)
            V[v_row, c] = random.choice(values)
            W[w_row, c] = random.choice(values)
            # Sometimes add more non-zeros
            if random.random() < 0.5:
                U[random.randint(0, n*m-1), c] = random.choice(values)
            if random.random() < 0.5:
                V[random.randint(0, m*p-1), c] = random.choice(values)
            if random.random() < 0.5:
                W[random.randint(0, p*n-1), c] = random.choice(values)
        
        # Local search with steepest descent
        for it in range(300):
            if time.time() - start > max_time:
                break
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err += (s - 1.0)**2
            
            if err < 1e-12:
                if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U.tolist(), V.tolist(), W.tolist()
                break
            
            # Find best single-element change
            best_err = err
            best_change = None
            for mat_name, mat, rows in [('U', U, n*m), ('V', V, m*p), ('W', W, p*n)]:
                for row in range(rows):
                    for col in range(rank):
                        old_val = mat[row, col]
                        for new_val in values:
                            if abs(new_val - old_val) < 1e-9:
                                continue
                            mat[row, col] = new_val
                            e = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                                        e += (s - 1.0)**2
                            if e < best_err - 1e-15:
                                best_err = e
                                best_change = (mat_name, row, col, new_val)
                            mat[row, col] = old_val
            
            if best_change is None:
                break
            
            mat_name, row, col, new_val = best_change
            if mat_name == 'U':
                U[row, col] = new_val
            elif mat_name == 'V':
                V[row, col] = new_val
            else:
                W[row, col] = new_val
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try ranks in order of promise
    ranks_to_try = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        time_per_rank = min(20, max(8, 35 - rank))
        
        result = _efficient_search(rank, n, m, p, max_time=time_per_rank)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END

