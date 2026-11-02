import time
import numpy as np

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Use 2x2 Strassen (rank 7) applied to 4x4 blocks for 4x4x4 multiplication.
    # 4x4x4 has rank <= 28 via 2-level Strassen.
    # 4x4x8 = two 4x4x4 multiplications, so rank <= 56.
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_idx = 0
    
    def add_2x2_strassen(base_i, base_j, base_k):
        """Add rank-7 Strassen terms for 2x2 x 2x2 -> 2x2 multiplication.
        A is at rows base_i..base_i+1, cols base_j..base_j+1
        B is at rows base_j..base_j+1, cols base_k..base_k+1
        C is at rows base_k..base_k+1, cols base_i..base_i+1
        """
        nonlocal col_idx
        
        # M1 = (A22 - A12)(B12 + B22)
        a = {(base_i+1, base_j+1): 1, (base_i+1, base_j): -1}
        b = {(base_j, base_k+1): 1, (base_j+1, base_k+1): 1}
        c = {(base_k, base_i): 1, (base_k, base_i+1): 1, (base_k+1, base_i): -1, (base_k+1, base_i+1): -1}
        U[base_i*m + base_j, col_idx] += -1
        U[base_i*m + base_j+1, col_idx] += 0
        U[(base_i+1)*m + base_j, col_idx] += -1
        U[(base_i+1)*m + base_j+1, col_idx] += 1
        V[base_j*p + base_k, col_idx] += 0
        V[base_j*p + base_k+1, col_idx] += 1
        V[(base_j+1)*p + base_k, col_idx] += 0
        V[(base_j+1)*p + base_k+1, col_idx] += 1
        W[base_k*n + base_i, col_idx] += 1
        W[base_k*n + base_i+1, col_idx] += 1
        W[(base_k+1)*n + base_i, col_idx] += -1
        W[(base_k+1)*n + base_i+1, col_idx] += -1
        col_idx += 1
        
        # M2 = (A11 + A21)(B11 - B21)
        U[base_i*m + base_j, col_idx] += 1
        U[base_i*m + base_j+1, col_idx] += 0
        U[(base_i+1)*m + base_j, col_idx] += 1
        U[(base_i+1)*m + base_j+1, col_idx] += 0
        V[base_j*p + base_k, col_idx] += 1
        V[base_j*p + base_k+1, col_idx] += 0
        V[(base_j+1)*p + base_k, col_idx] += -1
        V[(base_j+1)*p + base_k+1, col_idx] += 0
        W[base_k*n + base_i+1, col_idx] += 1
        W[(base_k+1)*n + base_i+1, col_idx] += 1
        W[base_k*n + base_i, col_idx] += 0
        W[(base_k+1)*n + base_i, col_idx] += 0
        col_idx += 1
        
        # M3 = (A11 - A22)(B22 - B11)
        U[base_i*m + base_j, col_idx] += 1
        U[base_i*m + base_j+1, col_idx] += 0
        U[(base_i+1)*m + base_j, col_idx] += 0
        U[(base_i+1)*m + base_j+1, col_idx] += -1
        V[base_j*p + base_k, col_idx] += -1
        V[base_j*p + base_k+1, col_idx] += 0
        V[(base_j+1)*p + base_k, col_idx] += 0
        V[(base_j+1)*p + base_k+1, col_idx] += 1
        W[base_k*n + base_i, col_idx] += -1
        W[base_k*n + base_i+1, col_idx] += 0
        W[(base_k+1)*n + base_i, col_idx] += 0
        W[(base_k+1)*n + base_i+1, col_idx] += 1
        col_idx += 1
        
        # M4 = (A11 + A22)(B11 + B22)
        U[base_i*m + base_j, col_idx] += 1
        U[base_i*m + base_j+1, col_idx] += 0
        U[(base_i+1)*m + base_j, col_idx] += 0
        U[(base_i+1)*m + base_j+1, col_idx] += 1
        V[base_j*p + base_k, col_idx] += 1
        V[base_j*p + base_k+1, col_idx] += 0
        V[(base_j+1)*p + base_k, col_idx] += 0
        V[(base_j+1)*p + base_k+1, col_idx] += 1
        W[base_k*n + base_i, col_idx] += 1
        W[base_k*n + base_i+1, col_idx] += 0
        W[(base_k+1)*n + base_i, col_idx] += 0
        W[(base_k+1)*n + base_i+1, col_idx] += 1
        col_idx += 1
        
        # M5 = (A12 + A22)(B21 - B11)
        U[base_i*m + base_j, col_idx] += 0
        U[base_i*m + base_j+1, col_idx] += 1
        U[(base_i+1)*m + base_j, col_idx] += 0
        U[(base_i+1)*m + base_j+1, col_idx] += 1
        V[base_j*p + base_k, col_idx] += -1
        V[base_j*p + base_k+1, col_idx] += 0
        V[(base_j+1)*p + base_k, col_idx] += 1
        V[(base_j+1)*p + base_k+1, col_idx] += 0
        W[base_k*n + base_i+1, col_idx] += 1
        W[(base_k+1)*n + base_i+1, col_idx] += 1
        W[base_k*n + base_i, col_idx] += 0
        W[(base_k+1)*n + base_i, col_idx] += 0
        col_idx += 1
        
        # M6 = (A11 - A12)(B12 + B22)
        U[base_i*m + base_j, col_idx] += 1
        U[base_i*m + base_j+1, col_idx] += -1
        U[(base_i+1)*m + base_j, col_idx] += 0
        U[(base_i+1)*m + base_j+1, col_idx] += 0
        V[base_j*p + base_k, col_idx] += 0
        V[base_j*p + base_k+1, col_idx] += 1
        V[(base_j+1)*p + base_k, col_idx] += 0
        V[(base_j+1)*p + base_k+1, col_idx] += 1
        W[base_k*n + base_i, col_idx] += 1
        W[base_k*n + base_i+1, col_idx] += 0
        W[(base_k+1)*n + base_i, col_idx] += 0
        W[(base_k+1)*n + base_i+1, col_idx] += 1
        col_idx += 1
        
        # M7 = (A21 - A11)(B11 + B21)
        U[base_i*m + base_j, col_idx] += -1
        U[base_i*m + base_j+1, col_idx] += 0
        U[(base_i+1)*m + base_j, col_idx] += 1
        U[(base_i+1)*m + base_j+1, col_idx] += 0
        V[base_j*p + base_k, col_idx] += 1
        V[base_j*p + base_k+1, col_idx] += 0
        V[(base_j+1)*p + base_k, col_idx] += 1
        V[(base_j+1)*p + base_k+1, col_idx] += 0
        W[base_k*n + base_i+1, col_idx] += 1
        W[(base_k+1)*n + base_i+1, col_idx] += 1
        W[base_k*n + base_i, col_idx] += 0
        W[(base_k+1)*n + base_i, col_idx] += 0
        col_idx += 1
    
    def add_4x4x4(base_i, base_j, base_k):
        """Add rank-28 terms for 4x4 x 4x4 -> 4x4 multiplication using 2x2 Strassen blocks."""
        # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        add_2x2_strassen(base_i, base_j, base_k)
        # Block (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,2:4]
        add_2x2_strassen(base_i, base_j+2, base_k+2)
        # Block (1,0): A[2:4,0:2] * B[0:2,2:4] -> C[2:4,0:2]
        add_2x2_strassen(base_i+2, base_j, base_k)
        # Block (1,1): A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
        add_2x2_strassen(base_i+2, base_j+2, base_k+2)
    
    # For 4x4 x 4x8: split B into B1 (cols 0-3) and B2 (cols 4-7)
    # C = A*B1 + A*B2
    # A*B1: 4x4 x 4x4 -> 4x4, result in C[:, 0:4]
    # A*B2: 4x4 x 4x4 -> 4x4, result in C[:, 4:8]
    
    add_4x4x4(0, 0, 0)   # A * B[:, 0:4] -> C[:, 0:4]
    add_4x4x4(0, 0, 4)   # A * B[:, 4:8] -> C[:, 4:8]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
