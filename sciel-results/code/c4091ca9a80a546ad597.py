"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # We'll use block decomposition:
    # Split n=3 into {0,1} and {2}
    # Split m=4 into {0,1} and {2,3}
    # Split p=7 into {0,1}, {2,3}, {4,5}, {6}
    #
    # For 2x2x2 blocks: use Strassen (rank 7)
    # For 2x2x1 blocks: use schoolbook (rank 4)
    # For 1x4x7 block: use schoolbook (rank 28)
    #
    # Total: 4*7 + 4*4 + 28 = 28 + 16 + 28 = 72
    
    rank = 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen 2x2x2 helper
    # A is 2x2 at rows a0:a0+2, cols b0:b0+2
    # B is 2x2 at rows b0:b0+2, cols c0:c0+1
    # C is 2x2 at rows a0:a0+2, cols c0:c0+1
    #
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2(a0, b0, c0):
        nonlocal col
        # A indices: A[a0,b0]=A11, A[a0,b0+1]=A12, A[a0+1,b0]=A21, A[a0+1,b0+1]=A22
        # B indices: B[b0,c0]=B11, B[b0,c0+1]=B12, B[b0+1,c0]=B21, B[b0+1,c0+1]=B22
        # C indices: C[a0,c0]=C11, C[a0,c0+1]=C12, C[a0+1,c0]=C21, C[a0+1,c0+1]=C22
        
        # M1 = (A11+A22)(B11+B22)
        U[a0 * m + b0, col] = 1
        U[(a0+1) * m + (b0+1), col] = 1
        V[b0 * p + c0, col] = 1
        V[(b0+1) * p + (c0+1), col] = 1
        W[c0 * n + a0, col] = 1       # C11
        W[(c0+1) * n + a0, col] = 1   # C12
        W[c0 * n + (a0+1), col] = 1   # C21
        W[(c0+1) * n + (a0+1), col] = 1  # C22
        col += 1
        
        # M2 = (A21+A22)B11
        U[(a0+1) * m + b0, col] = 1
        U[(a0+1) * m + (b0+1), col] = 1
        V[b0 * p + c0, col] = 1
        W[c0 * n + (a0+1), col] = 1   # C21
        W[(c0+1) * n + (a0+1), col] = 1  # C22
        col += 1
        
        # M3 = A11(B21-B22)
        U[a0 * m + b0, col] = 1
        V[(b0+1) * p + c0, col] = 1
        W[(c0+1) * n + a0, col] = 1   # C12
        col += 1
        
        # M4 = A22(B11-B21)
        U[(a0+1) * m + (b0+1), col] = 1
        V[b0 * p + c0, col] = 1
        W[c0 * n + (a0+1), col] = 1   # C21
        col += 1
        
        # M5 = (A11+A12)B22
        U[a0 * m + b0, col] = 1
        U[a0 * m + (b0+1), col] = 1
        V[(b0+1) * p + (c0+1), col] = 1
        W[(c0+1) * n + a0, col] = 1   # C12
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(a0+1) * m + b0, col] = 1
        U[a0 * m + b0, col] = -1
        V[b0 * p + c0, col] = 1
        V[b0 * p + (c0+1), col] = 1
        W[(c0+1) * n + (a0+1), col] = 1  # C22
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a0 * m + (b0+1), col] = 1
        U[(a0+1) * m + (b0+1), col] = -1
        V[(b0+1) * p + c0, col] = 1
        V[(b0+1) * p + (c0+1), col] = 1
        W[c0 * n + a0, col] = 1       # C11
        col += 1
    
    # Schoolbook 2x2x1 helper
    # A is 2x2 at rows a0:a0+2, cols b0:b0+2
    # B is 2x1 at rows b0:b0+2, col c0
    # C is 2x1 at rows a0:a0+2, col c0
    
    def add_schoolbook_2x2x1(a0, b0, c0):
        nonlocal col
        # C[a0, c0] = A[a0,b0]*B[b0,c0] + A[a0,b0+1]*B[b0+1,c0]
        # C[a0+1, c0] = A[a0+1,b0]*B[b0,c0] + A[a0+1,b0+1]*B[b0+1,c0]
        
        # Term 1: A[a0,b0] * B[b0,c0] -> C[a0,c0]
        U[a0 * m + b0, col] = 1
        V[b0 * p + c0, col] = 1
        W[c0 * n + a0, col] = 1
        col += 1
        
        # Term 2: A[a0,b0+1] * B[b0+1,c0] -> C[a0,c0]
        U[a0 * m + (b0+1), col] = 1
        V[(b0+1) * p + c0, col] = 1
        W[c0 * n + a0, col] = 1
        col += 1
        
        # Term 3: A[a0+1,b0] * B[b0,c0] -> C[a0+1,c0]
        U[(a0+1) * m + b0, col] = 1
        V[b0 * p + c0, col] = 1
        W[c0 * n + (a0+1), col] = 1
        col += 1
        
        # Term 4: A[a0+1,b0+1] * B[b0+1,c0] -> C[a0+1,c0]
        U[(a0+1) * m + (b0+1), col] = 1
        V[(b0+1) * p + c0, col] = 1
        W[c0 * n + (a0+1), col] = 1
        col += 1
    
    # Now build the decomposition
    # Blocks for rows 0,1 of A (2x4 matrix):
    # j in {0,1}, k in {0,1}: Strassen 2x2x2
    # j in {0,1}, k in {2,3}: Strassen 2x2x2
    # j in {0,1}, k in {4,5}: Strassen 2x2x2
    # j in {0,1}, k in {6}:    Schoolbook 2x2x1
    # j in {2,3}, k in {0,1}: Strassen 2x2x2
    # j in {2,3}, k in {2,3}: Strassen 2x2x2
    # j in {2,3}, k in {4,5}: Strassen 2x2x2
    # j in {2,3}, k in {6}:    Schoolbook 2x2x1
    
    # 4 Strassen blocks (j in {0,1}, k in {0,1},{2,3},{4,5} and j in {2,3}, k in {0,1},{2,3},{4,5})
    # Wait, that's 3 Strassen for j={0,1} and 3 Strassen for j={2,3}, plus 2 schoolbook
    # Total: 6*7 + 2*4 = 42 + 8 = 50 for the 2x4x7 part
    
    # Actually let me recount:
    # j in {0,1}: k in {0,1} (Strassen), {2,3} (Strassen), {4,5} (Strassen), {6} (schoolbook)
    # j in {2,3}: k in {0,1} (Strassen), {2,3} (Strassen), {4,5} (Strassen), {6} (schoolbook)
    # So 6 Strassen + 2 schoolbook = 6*7 + 2*4 = 42 + 8 = 50
    
    # For row 2 of A (1x4 matrix): schoolbook 1x4x7 = 28
    
    # Total: 50 + 28 = 78
    
    # Let me adjust rank
    rank = 78
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Process j in {0,1}
    for b0 in [0, 2]:
        # k in {0,1}: Strassen
        add_strassen_2x2(0, b0, 0)
        # k in {2,3}: Strassen
        add_strassen_2x2(0, b0, 2)
        # k in {4,5}: Strassen
        add_strassen_2x2(0, b0, 4)
        # k in {6}: Schoolbook 2x2x1
        add_schoolbook_2x2x1(0, b0, 6)
    
    # Process row 2 of A: schoolbook 1x4x7
    for j in range(m):
        for k in range(p):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    
    if col != rank:
        # Truncate or pad if needed
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

