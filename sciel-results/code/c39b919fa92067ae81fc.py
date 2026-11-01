import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses 3 Strassen blocks (rank 7 each) for columns 0-5, and schoolbook (rank 4) for column 6.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen algorithm for 2x2 * 2x2:
    # A = [[a,b],[c,d]] indexed as A[0]=a, A[1]=b, A[2]=c, A[3]=d
    # B = [[e,f],[g,h]] indexed as B[0]=e, B[1]=f, B[2]=g, B[3]=h (for 2x2 block)
    # But our B is 2x7, so for columns k,k+1: B[0,k]=e, B[1,k]=f, B[0,k+1]=g, B[1,k+1]=h
    # C = [[C00,C01],[C10,C11]] indexed as C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11
    # Wait, the output convention is W[k*n+i, r], so C[i,k] is at index k*n+i
    # For 2x2 block at columns k,k+1:
    # C[i,k] at index k*2+i, C[i,k+1] at index (k+1)*2+i
    
    def add_strassen_block(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        # A indices: 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
        # B for this block: B[0,k]=B[7*0+k], B[1,k]=B[7*1+k], B[0,k+1]=B[7*0+k+1], B[1,k+1]=B[7*1+k+1]
        # C: C[0,k] at W[2*k+0], C[1,k] at W[2*k+1], C[0,k+1] at W[2*(k+1)+0], C[1,k+1] at W[2*(k+1)+1]
        
        # M1 = (a + d) * (B0k + B1,k+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        
        # M2 = (c + d) * B0k
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        
        # M3 = a * (B0,k+1 - B1,k+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        
        # M4 = d * (B1k - B0k)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        
        # M5 = (a + b) * B1,k+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        
        # M6 = (c - a) * (B0k + B0,k+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        
        # M7 = (b - d) * (B1k + B1,k+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    # Add three Strassen blocks for columns (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6 (rank 4)
    col_start = 21
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Verify with random matrices
    np.random.seed(42)
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print("C_mat:", C_mat)
        print("C_expected:", C_expected)
        print("Diff:", np.abs(C_mat - C_expected))
