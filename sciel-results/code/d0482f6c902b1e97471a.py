"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Construct rank-28 decomposition for M_{2,3,5}
    # Strategy: Use Strassen's 7-term algorithm on two 2x2x2 subblocks
    # Block 1: j in {0,1}, k in {0,1} (8 entries, 7 terms)
    # Block 2: j in {0,1}, k in {3,4} (8 entries, 7 terms)
    # Remaining: 14 entries with 14 schoolbook terms
    # Total: 7 + 7 + 14 = 28
    
    rank = 28
    U = np.zeros((n*m, rank))  # (6, 28)
    V = np.zeros((m*p, rank))  # (15, 28)
    W = np.zeros((p*n, rank))  # (10, 28)
    
    col = 0
    
    # Helper: add a Strassen block for given j_range, k_range
    # j_range: which 2 of the 3 middle indices
    # k_range: which 2 of the 5 output indices
    def add_strassen_block(j0, j1, k0, k1):
        nonlocal col
        # A indices: i in {0,1}, j in {j0,j1} -> U[i*3+j]
        # B indices: j in {j0,j1}, k in {k0,k1} -> V[j*5+k]
        # C indices: i in {0,1}, k in {k0,k1} -> W[k*2+i]
        
        # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k0])
        U[0*3+j0, col] = 1; U[1*3+j1, col] = 1
        V[j0*5+k0, col] = 1; V[j1*5+k0, col] = 1
        W[k0*2+0, col] = 1   # C[0,k0] += M1
        W[k1*2+1, col] = 1   # C[1,k1] += M1
        col += 1
        
        # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
        U[1*3+j0, col] = 1; U[1*3+j1, col] = 1
        V[j0*5+k0, col] = 1
        W[k0*2+1, col] = 1   # C[1,k0] += M2
        W[k1*2+1, col] = -1  # C[1,k1] -= M2
        col += 1
        
        # M3 = A[0,j0] * (B[j1,k0]-B[j0,k0])
        U[0*3+j0, col] = 1
        V[j1*5+k0, col] = 1; V[j0*5+k0, col] = -1
        W[k1*2+0, col] = 1   # C[0,k1] += M3
        W[k0*2+1, col] = 1   # C[1,k0] += M3
        col += 1
        
        # M4 = A[1,j1] * (B[j0,k1]-B[j1,k1])
        U[1*3+j1, col] = 1
        V[j0*5+k1, col] = 1; V[j1*5+k1, col] = -1
        W[k0*2+0, col] = 1   # C[0,k0] += M4
        W[k0*2+1, col] = 1   # C[1,k0] += M4
        col += 1
        
        # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
        U[0*3+j0, col] = 1; U[0*3+j1, col] = 1
        V[j1*5+k1, col] = 1
        W[k0*2+0, col] = -1  # C[0,k0] -= M5
        W[k1*2+0, col] = 1   # C[0,k1] += M5
        col += 1
        
        # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j1,k0])
        U[1*3+j0, col] = 1; U[0*3+j0, col] = -1
        V[j0*5+k0, col] = 1; V[j1*5+k0, col] = 1
        W[k0*2+1, col] = 1   # C[1,k0] += M6
        W[k1*2+1, col] = 1   # C[1,k1] += M6
        col += 1
        
        # M7 = (A[0,j1]-A[1,j1]) * (B[j0,k1]+B[j1,k1])
        U[0*3+j1, col] = 1; U[1*3+j1, col] = -1
        V[j0*5+k1, col] = 1; V[j1*5+k1, col] = 1
        W[k1*2+0, col] = 1   # C[0,k1] += M7
        W[k1*2+1, col] = 1   # C[1,k1] += M7
        col += 1
    
    # Add Strassen block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 1, 0, 1)
    
    # Add Strassen block 2: j in {0,1}, k in {3,4}
    add_strassen_block(0, 1, 3, 4)
    
    # Remaining 14 terms (schoolbook):
    # j=2, all k in {0,1,2,3,4}: 2*5 = 10 terms
    # j in {0,1}, k=2: 2*2 = 4 terms
    # Total: 14 terms
    
    # j=2, k in {0,1,2,3,4}
    for k in range(5):
        for i in range(2):
            U[i*3+2, col] = 1
            V[2*5+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    
    # j in {0,1}, k=2
    for j in range(2):
        for i in range(2):
            U[i*3+j, col] = 1
            V[j*5+2, col] = 1
            W[2*2+i, col] = 1
            col += 1
    
    # Verify col == 28
    assert col == 28, f"Expected 28 columns, got {col}"
    
    # Verify the decomposition
    T_target = np.zeros((6, 15, 10))
    for i in range(2):
        for j in range(3):
            for k in range(5):
                T_target[i*3+j, j*5+k, k*2+i] = 1
    
    T_recon = np.zeros((6, 15, 10))
    for c in range(28):
        T_recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook rank 30
        U30 = np.zeros((6, 30))
        V30 = np.zeros((15, 30))
        W30 = np.zeros((10, 30))
        c30 = 0
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    U30[i*3+j, c30] = 1
                    V30[j*5+k, c30] = 1
                    W30[k*2+i, c30] = 1
                    c30 += 1
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

