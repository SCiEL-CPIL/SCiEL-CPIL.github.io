import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x16 matrix.
    
    The multiplication is decomposed into 8 independent 2x2x2 multiplications.
    Each 2x2x2 multiplication uses Strassen's algorithm (rank 7).
    Total rank = 8 * 7 = 56.
    
    U: shape (4, 56)  # 4 entries in A
    V: shape (32, 56) # 32 entries in B (2x16)
    W: shape (32, 56) # 32 entries in C (2x16)
    
    Indexing convention:
    A is 2x2: A[i,j] -> U[i*2+j]
    B is 2x16: B[i,k] -> V[i*16+k]
    C is 2x16: C[i,k] -> W[i*16+k] (Note: Task says W[k*n+i, r] for output factor? 
    "The output factor follows the official trace convention W[k*n+i,r]"
    Let's re-read carefully: "W[k*n+i,r]"
    Usually C[i,j] is row i, col j.
    If W is (p*n, rank), and p=16, n=2.
    If the convention is W[k*n+i], where k is the column index of C (0..15) and i is the row index of C (0..1).
    Then W index = k*2 + i.
    This corresponds to C[i,k] if we store column-major in the flattened vector?
    Let's check the current program's W indexing.
    Current program:
    # Indices in W for block b:
    #   C00: 2b*2 + 0 = 4b
    #   C01: (2b+1)*2 + 0 = 4b+2
    #   C10: 2b*2 + 1 = 4b+1
    #   C11: (2b+1)*2 + 1 = 4b+3
    
    Here, block b corresponds to columns 2b and 2b+1 of B and C.
    C00 is C[0, 2b]. Index 4b.
    C01 is C[0, 2b+1]. Index 4b+2.
    C10 is C[1, 2b]. Index 4b+1.
    C11 is C[1, 2b+1]. Index 4b+3.
    
    Let's check the formula W[k*n+i].
    If k is the column index (0..15) and i is the row index (0..1).
    For C[0, 2b]: k=2b, i=0. Index = 2b*2 + 0 = 4b. Matches.
    For C[0, 2b+1]: k=2b+1, i=0. Index = (2b+1)*2 + 0 = 4b+2. Matches.
    For C[1, 2b]: k=2b, i=1. Index = 2b*2 + 1 = 4b+1. Matches.
    For C[1, 2b+1]: k=2b+1, i=1. Index = (2b+1)*2 + 1 = 4b+3. Matches.
    
    So the W indexing in the current program is correct according to the trace convention.
    """
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # A indices in U:
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # For each block b (0..7):
    # B indices in V:
    # B00: 0*16 + 2b = 2b
    # B01: 0*16 + 2b+1 = 2b+1
    # B10: 1*16 + 2b = 16+2b
    # B11: 1*16 + 2b+1 = 16+2b+1
    
    # C indices in W (based on k*n+i):
    # C00 (row 0, col 2b): k=2b, i=0 -> 4b
    # C01 (row 0, col 2b+1): k=2b+1, i=0 -> 4b+2
    # C10 (row 1, col 2b): k=2b, i=1 -> 4b+1
    # C11 (row 1, col 2b+1): k=2b+1, i=1 -> 4b+3
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    for b in range(8):
        base_v_b0 = 2 * b
        base_v_b1 = 16 + 2 * b
        base_w = 4 * b
        
        # r index for this block
        r_base = b * 7
        
        # M1
        r = r_base + 0
        U[0][r] += 1; U[3][r] += 1
        V[base_v_b0][r] += 1; V[base_v_b1 + 1][r] += 1
        W[base_w][r] += 1; W[base_w + 3][r] += 1
        
        # M2
        r = r_base + 1
        U[2][r] += 1; U[3][r] += 1
        V[base_v_b0][r] += 1
        W[base_w + 1][r] += 1; W[base_w + 3][r] += -1
        
        # M3
        r = r_base + 2
        U[0][r] += 1
        V[base_v_b0 + 1][r] += 1; V[base_v_b1 + 1][r] += -1
        W[base_w + 2][r] += 1; W[base_w + 3][r] += 1
        
        # M4
        r = r_base + 3
        U[3][r] += 1
        V[base_v_b1][r] += 1; V[base_v_b0][r] += -1
        W[base_w][r] += 1; W[base_w + 1][r] += 1
        
        # M5
        r = r_base + 4
        U[0][r] += 1; U[1][r] += 1
        V[base_v_b1 + 1][r] += 1
        W[base_w][r] += -1; W[base_w + 2][r] += 1
        
        # M6
        r = r_base + 5
        U[2][r] += 1; U[0][r] += -1
        V[base_v_b0][r] += 1; V[base_v_b0 + 1][r] += 1
        W[base_w + 3][r] += 1
        
        # M7
        r = r_base + 6
        U[1][r] += 1; U[3][r] += -1
        V[base_v_b1][r] += 1; V[base_v_b1 + 1][r] += 1
        W[base_w][r] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
