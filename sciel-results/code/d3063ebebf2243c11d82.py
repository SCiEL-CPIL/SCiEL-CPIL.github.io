import time
import numpy as np

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify(U, V, W, n=2, m=2, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """
    Constructs a rank-18 decomposition:
    - Strassen block for cols 0,1 (rank 7)
    - Strassen block for cols 2,3 (rank 7)
    - Schoolbook for col 4 (rank 4)
    """
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a Strassen block for columns col0, col1
    def add_strassen(col0, col1, offset):
        # M1 = (A00 + A11)(B00 + B11)
        U[0, offset] = 1; U[3, offset] = 1
        V[0*p+col0, offset] = 1; V[1*p+col1, offset] = 1
        W[col0*n+0, offset] = 1; W[col1*n+1, offset] = 1
        
        # M2 = (A10 + A11)B00
        U[2, offset+1] = 1; U[3, offset+1] = 1
        V[0*p+col0, offset+1] = 1
        W[col0*n+1, offset+1] = 1; W[col1*n+1, offset+1] = -1
        
        # M3 = A00(B11 - B00)
        U[0, offset+2] = 1
        V[0*p+col1, offset+2] = 1; V[1*p+col1, offset+2] = -1
        W[col1*n+0, offset+2] = 1; W[col1*n+1, offset+2] = 1
        
        # M4 = (A01 - A00)B10
        U[3, offset+3] = 1
        V[1*p+col0, offset+3] = 1; V[0*p+col0, offset+3] = -1
        W[col0*n+0, offset+3] = 1; W[col0*n+1, offset+3] = 1
        
        # M5 = (A00 + A01)B11
        U[0, offset+4] = 1; U[1, offset+4] = 1
        V[1*p+col1, offset+4] = 1
        W[col0*n+0, offset+4] = -1; W[col1*n+0, offset+4] = 1
        
        # M6 = (A10 - A00)(B01 + B11)
        U[2, offset+5] = 1; U[0, offset+5] = -1
        V[0*p+col0, offset+5] = 1; V[0*p+col1, offset+5] = 1
        W[col1*n+1, offset+5] = 1
        
        # M7 = (A11 - A01)(B10 + B00) -- Wait, standard Strassen M7 is (A11 - A01)(B10 + B00)?
        # Let's check standard Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b11)
        # M4 = a22(b12-b22)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # My indices: i is row, j is col.
        # A[i,j]. B[j,k]. C[i,k].
        # M1: (A00+A11)(B00+B11) -> U[0],U[3]; V[0,0],V[1,1]; W[0,0],W[1,1]
        # M2: (A10+A11)B00 -> U[2],U[3]; V[0,0]; W[1,0],W[1,1]
        # M3: A00(B11-B00) -> U[0]; V[0,1],V[1,1]; W[1,0],W[1,1]
        # M4: A11(B01-B11) -> U[3]; V[1,0],V[0,0]? No.
        # Let's stick to the code in the prompt which was verified.
        
    # Instead of reimplementing, I will use the exact logic from the prompt's _get_strassen_block_np
    # which was verified to work.
    
    def get_strassen_block(col0, col1):
        U_b = np.zeros((n*m, 7))
        V_b = np.zeros((m*p, 7))
        W_b = np.zeros((p*n, 7))
        
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0*p+col0, 0] = 1; V_b[1*p+col1, 0] = 1
        W_b[col0*n+0, 0] = 1; W_b[col1*n+1, 0] = 1
        
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0*p+col0, 1] = 1
        W_b[col0*n+1, 1] = 1; W_b[col1*n+1, 1] = -1
        
        U_b[0, 2] = 1
        V_b[0*p+col1, 2] = 1; V_b[1*p+col1, 2] = -1
        W_b[col1*n+0, 2] = 1; W_b[col1*n+1, 2] = 1
        
        U_b[3, 3] = 1
        V_b[1*p+col0, 3] = 1; V_b[0*p+col0, 3] = -1
        W_b[col0*n+0, 3] = 1; W_b[col0*n+1, 3] = 1
        
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[1*p+col1, 4] = 1
        W_b[col0*n+0, 4] = -1; W_b[col1*n+0, 4] = 1
        
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0*p+col0, 5] = 1; V_b[0*p+col1, 5] = 1
        W_b[col1*n+1, 5] = 1
        
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[1*p+col0, 6] = 1; V_b[1*p+col1, 6] = 1
        W_b[col0*n+0, 6] = 1
        
        return U_b, V_b, W_b

    # Block 1: cols 0, 1
    U_b1, V_b1, W_b1 = get_strassen_block(0, 1)
    U[:, 0:7] = U_b1
    V[:, 0:7] = V_b1
    W[:, 0:7] = W_b1
    
    # Block 2: cols 2, 3
    U_b2, V_b2, W_b2 = get_strassen_block(2, 3)
    U[:, 7:14] = U_b2
    V[:, 7:14] = V_b2
    W[:, 7:14] = W_b2
    
    # Block 3: col 4 (Schoolbook)
    U[0, 14] = 1; V[0*p+4, 14] = 1; W[4*n+0, 14] = 1
    U[1, 15] = 1; V[1*p+4, 15] = 1; W[4*n+0, 15] = 1
    U[2, 16] = 1; V[0*p+4, 16] = 1; W[4*n+1, 16] = 1
    U[3, 17] = 1; V[1*p+4, 17] = 1; W[4*n+1, 17] = 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    if not _verify(U_list, V_list, W_list, n, m, p):
        raise ValueError("Rank-18 construction failed verification")
        
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    return _construct_rank18(n, m, p)
