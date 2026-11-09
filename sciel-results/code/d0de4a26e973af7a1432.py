"""Matrix multiplication 2x5x6 tensor decomposition - block-structured approach."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 5, 6

    # Strassen decomposition for 2x2x2 (rank 7)
    # C = A * B where A, B, C are 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen's 7 products:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6

    def strassen_2x2x2():
        """Return U (4x7), V (4x7), W (4x7) for 2x2x2 matrix multiplication."""
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7  -> j*2+k: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  -> k*2+i: (0,0)=0,(0,1)=2,(1,0)=1,(1,1)=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1 = (a+d)(e+h): U[0]=1, U[3]=1; V[0]=1, V[3]=1; W[0]=1, W[3]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2 = (c+d)e: U[2]=1, U[3]=1; V[0]=1; W[2]=1, W[3]=-1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # P3 = a(f-h): U[0]=1; V[1]=1, V[3]=-1; W[1]=1, W[3]=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # P4 = d(g-e): U[3]=1; V[2]=1, V[0]=-1; W[0]=1, W[2]=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # P5 = (a+b)h: U[0]=1, U[1]=1; V[3]=1; W[1]=-1, W[3]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = -1; W[3, 4] = 1
        
        # P6 = (c-a)(e+f): U[2]=1, U[0]=-1; V[0]=1, V[1]=1; W[3]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P7 = (b-d)(g+h): U[1]=1, U[3]=-1; V[2]=1, V[3]=1; W[0]=-1, W[3]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1
        
        return U, V, W

    # Build 2x3x3 decomposition (rank 11)
    # C[0:2, 0:3] = A[0:2, 0:3] * B[0:3, 0:3]
    # Split into: first 2 columns use Strassen (rank 7), 3rd column adds 4 terms
    # 3rd column: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #             C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # These are 4 rank-1 terms.
    
    def build_2x3x3():
        """Return U (6x11), V (9x11), W (6x11) for 2x3x3 matrix multiplication."""
        # A is 2x3: indices i*3+j: (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5
        # B is 3x3: indices j*3+k: (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5,(2,0)=6,(2,1)=7,(2,2)=8
        # C is 2x3: indices k*2+i: (0,0)=0,(0,1)=2,(1,0)=1,(1,1)=3,(2,0)=4,(2,1)=5
        
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((6, 11))
        
        # First 2 columns: use Strassen for 2x2x2
        # A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
        sU, sV, sW = strassen_2x2x2()
        # Map Strassen indices to 2x3x3 indices
        # A: i*2+j -> i*3+j (same for j<2)
        # B: j*2+k -> j*3+k (same for k<2)  
        # C: k*2+i -> k*2+i (same)
        U[:, 0:7] = sU
        V[:, 0:7] = sV
        W[:, 0:7] = sW
        
        # 3rd column terms:
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Term 7: A[0,0]*B[0,2] contributes to C[0,2]
        U[0, 7] = 1   # A[0,0]
        V[2, 7] = 1   # B[0,2]
        W[5, 7] = 1   # C[0,2] = k*2+i = 2*2+1 = 5
        
        # Term 8: A[0,1]*B[1,2] contributes to C[0,2]
        U[1, 8] = 1   # A[0,1]
        V[5, 8] = 1   # B[1,2]
        W[5, 8] = 1   # C[0,2]
        
        # Term 9: A[1,0]*B[0,2] contributes to C[1,2]
        U[3, 9] = 1   # A[1,0]
        V[2, 9] = 1   # B[0,2]
        W[4, 9] = 1   # C[1,2] = k*2+i = 2*2+0 = 4
        
        # Term 10: A[1,1]*B[1,2] contributes to C[1,2]
        U[4, 10] = 1  # A[1,1]
        V[5, 10] = 1  # B[1,2]
        W[4, 10] = 1  # C[1,2]
        
        return U, V, W

    # Build 2x2x3 decomposition (rank 11)
    # C[0:2, 0:3] = A[0:2, 0:2] * B[0:2, 0:3]
    # First 2 columns: Strassen (rank 7), 3rd column: 4 terms
    
    def build_2x2x3():
        """Return U (4x11), V (6x11), W (6x11) for 2x2x3 matrix multiplication."""
        # A is 2x2: indices i*2+j: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3
        # B is 2x3: indices j*3+k: (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5
        # C is 2x3: indices k*2+i: (0,0)=0,(0,1)=2,(1,0)=1,(1,1)=3,(2,0)=4,(2,1)=5
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # First 2 columns: Strassen
        sU, sV, sW = strassen_2x2x2()
        U[:, 0:7] = sU
        # B: j*2+k -> j*3+k for k<2: (0,0)=0,(0,1)=1,(1,0)=3,(1,1)=4
        V[0, 0:7] = sV[0, 0:7]
        V[1, 0:7] = sV[1, 0:7]
        V[3, 0:7] = sV[2, 0:7]
        V[4, 0:7] = sV[3, 0:7]
        W[:, 0:7] = sW
        
        # 3rd column: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        #             C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Term 7: A[0,0]*B[0,2] -> C[0,2]
        U[0, 7] = 1
        V[2, 7] = 1
        W[5, 7] = 1
        
        # Term 8: A[0,1]*B[1,2] -> C[0,2]
        U[1, 8] = 1
        V[5, 8] = 1
        W[5, 8] = 1
        
        # Term 9: A[1,0]*B[0,2] -> C[1,2]
        U[2, 9] = 1
        V[2, 9] = 1
        W[4, 9] = 1
        
        # Term 10: A[1,1]*B[1,2] -> C[1,2]
        U[3, 10] = 1
        V[5, 10] = 1
        W[4, 10] = 1
        
        return U, V, W

    # Assemble 2x5x6 decomposition
    # Split A columns: [0:3] and [3:5]
    # Split B rows: [0:3] and [3:5]
    # Split C columns: [0:3] and [3:6]
    
    # Block 1: A[0:2, 0:3] * B[0:3, 0:3] -> C[0:2, 0:3]  (2x3x3, rank 11)
    # Block 2: A[0:2, 0:3] * B[0:3, 3:6] -> C[0:2, 3:6]  (2x3x3, rank 11)
    # Block 3: A[0:2, 3:5] * B[3:5, 0:3] -> C[0:2, 0:3]  (2x2x3, rank 11)
    # Block 4: A[0:2, 3:5] * B[3:5, 3:6] -> C[0:2, 3:6]  (2x2x3, rank 11)
    
    total_rank = 44
    
    U_full = np.zeros((n * m, total_rank))  # 10 x 44
    V_full = np.zeros((m * p, total_rank))  # 30 x 44
    W_full = np.zeros((p * n, total_rank))  # 12 x 44
    
    col_offset = 0
    
    # Block 1: A cols 0-2, B rows 0-2, C cols 0-2
    b1U, b1V, b1W = build_2x3x3()
    # U: A[i, j] for i in 0-1, j in 0-2 -> index i*5+j
    for i in range(2):
        for j in range(3):
            U_full[i * 5 + j, col_offset:col_offset+11] = b1U[i * 3 + j, :]
    # V: B[j, k] for j in 0-2, k in 0-2 -> index j*6+k
    for j in range(3):
        for k in range(3):
            V_full[j * 6 + k, col_offset:col_offset+11] = b1V[j * 3 + k, :]
    # W: C[i, k] for i in 0-1, k in 0-2 -> index k*2+i
    for i in range(2):
        for k in range(3):
            W_full[k * 2 + i, col_offset:col_offset+11] = b1W[k * 2 + i, :]
    col_offset += 11
    
    # Block 2: A cols 0-2, B rows 0-2, C cols 3-5
    b2U, b2V, b2W = build_2x3x3()
    for i in range(2):
        for j in range(3):
            U_full[i * 5 + j, col_offset:col_offset+11] = b2U[i * 3 + j, :]
    for j in range(3):
        for k in range(3):
            V_full[j * 6 + (k + 3), col_offset:col_offset+11] = b2V[j * 3 + k, :]
    for i in range(2):
        for k in range(3):
            W_full[(k + 3) * 2 + i, col_offset:col_offset+11] = b2W[k * 2 + i, :]
    col_offset += 11
    
    # Block 3: A cols 3-4, B rows 3-4, C cols 0-2
    b3U, b3V, b3W = build_2x2x3()
    for i in range(2):
        for j in range(2):
            U_full[i * 5 + (j + 3), col_offset:col_offset+11] = b3U[i * 2 + j, :]
    for j in range(2):
        for k in range(3):
            V_full[(j + 3) * 6 + k, col_offset:col_offset+11] = b3V[j * 3 + k, :]
    for i in range(2):
        for k in range(3):
            W_full[k * 2 + i, col_offset:col_offset+11] = b3W[k * 2 + i, :]
    col_offset += 11
    
    # Block 4: A cols 3-4, B rows 3-4, C cols 3-5
    b4U, b4V, b4W = build_2x2x3()
    for i in range(2):
        for j in range(2):
            U_full[i * 5 + (j + 3), col_offset:col_offset+11] = b4U[i * 2 + j, :]
    for j in range(2):
        for k in range(3):
            V_full[(j + 3) * 6 + (k + 3), col_offset:col_offset+11] = b4V[j * 3 + k, :]
    for i in range(2):
        for k in range(3):
            W_full[(k + 3) * 2 + i, col_offset:col_offset+11] = b4W[k * 2 + i, :]
    col_offset += 11
    
    # Verify the decomposition
    tensor = np.zeros((n, m, p))
    for r in range(total_rank):
        u_col = U_full[:, r]
        v_col = V_full[:, r]
        w_col = W_full[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    
    if not np.allclose(tensor, 1.0, atol=1e-10):
        # Fallback to trivial decomposition
        U_triv = np.zeros((n * m, n * m * p))
        V_triv = np.zeros((m * p, n * m * p))
        W_triv = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_triv[i * m + j, col] = 1.0
                    V_triv[j * p + k, col] = 1.0
                    W_triv[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
