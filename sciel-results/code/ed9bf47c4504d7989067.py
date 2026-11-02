import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x3 x 3x8 matrix multiplication.
    
    Strategy: Decompose the 3x8 matrix into two 3x4 blocks. For each 3x3 x 3x4 block,
    use Strassen's algorithm (7 terms) for the 2x2 sub-block and direct multiplication
    for the remaining terms. This gives 39 terms per block, total 78... 
    
    Wait, that's worse than 72. Let me reconsider.
    
    Actually for 3x3 x 3x4:
    - Schoolbook: 3*3*4 = 36 terms
    - Strassen on 2x2 block: saves 1 term per 2x2x2, so 36-1 = 35 terms
    Total: 70 terms. Score = 72/70 = 1.0286
    
    Let me implement rank 70.
    """
    n, m, p = 3, 3, 8
    
    # We'll build the decomposition as a list of (u, v, w) triples
    # where u has shape (n*m,), v has shape (m*p,), w has shape (p*n,)
    # Each triple contributes a rank-1 term
    
    terms = []
    
    def add_term(u, v, w):
        """Add a rank-1 term. u, v, w are lists of length 9, 12, 24 respectively."""
        terms.append((u, v, w))
    
    def make_vec(size, values):
        """Create a vector of given size with specified nonzero values."""
        vec = [0] * size
        for idx, val in values.items():
            if idx < size:
                vec[idx] = val
        return vec
    
    # For 3x3 x 3x4, we use:
    # Strassen for the 2x2 top-left block (7 terms)
    # Direct for the remaining parts
    
    # Strassen 2x2x2 terms for indices i,j,k in {0,1}
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b01)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b01+b11)
    # M7 = (a01-a11)(b00+b10)
    # c00 = M1+M4-M5-M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M7
    
    # In tensor form, for each Strassen term, we need to specify
    # which A entries, B entries, and C entries are involved
    
    # For 3x3 x 3x4, we have two 3x4 blocks (columns 0-3 and 4-7)
    
    for block in range(2):
        # block 0: B columns 0-3, block 1: B columns 4-7
        # Offset in B: block * 4
        
        # Strassen terms for A[0:2,0:2] x B[0:2, block*4: block*4+2]
        # These contribute to C[0:2, block*4: block*4+2]
        
        # M1 = (a00+a11)(b00+b11)
        u1 = make_vec(9, {0: 1, 4: 1})  # A[0,0] + A[1,1]
        v1 = make_vec(12, {0: 1, 4: 1})  # B[0, block*4] + B[1, block*4+1]
        # Wait, B indexing: B[j, k] is at index j*4 + k in the 3x4 block
        # But in the full 3x8 matrix, B[j, k] is at index j*8 + k
        
        # Let me redo with proper indexing
        # U: index = i*m + j = i*3 + j for A[i,j]
        # V: index = j*p + k = j*8 + k for B[j,k]  
        # W: index = k*n + i = k*3 + i for C[i,k]
        
        # Actually the problem says W[k*n+i, r] so W index = k*n + i = k*3 + i
        
        pass
    
    # Let me restart with clearer indexing
    terms.clear()
    
    # For each of the two 3x4 blocks (columns 0-3, 4-7)
    for bcol_start in [0, 4]:
        # bcol_start: first column of this 3x4 block in the full 3x8 matrix
        
        # Strassen for A[0:2,0:2] x B[0:2, bcol_start:bcol_start+2]
        # Contributing to C[0:2, bcol_start:bcol_start+2]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,bcol_start]+B[1,bcol_start+1])
        # Contributes to: C[0,bcol_start] += M1, C[1,bcol_start+1] += M1
        u = make_vec(9, {0: 1, 4: 1})  # A[0,0] + A[1,1]
        v = make_vec(24, {0*8+bcol_start: 1, 1*8+bcol_start+1: 1})  # B[0,bcol_start] + B[1,bcol_start+1]
        w = make_vec(24, {bcol_start*3+0: 1, (bcol_start+1)*3+1: 1})  # C[0,bcol_start] + C[1,bcol_start+1]
        add_term(u, v, w)
        
        # M2 = (A[1,0]+A[1,1]) * B[0,bcol_start]
        # Contributes to: C[1,bcol_start] += M2
        u = make_vec(9, {3: 1, 4: 1})  # A[1,0] + A[1,1]
        v = make_vec(24, {0*8+bcol_start: 1})  # B[0,bcol_start]
        w = make_vec(24, {bcol_start*3+1: 1})  # C[1,bcol_start]
        add_term(u, v, w)
        
        # M3 = A[0,0] * (B[1,bcol_start]+B[1,bcol_start+1])
        # Contributes to: C[0,bcol_start+1] += M3
        u = make_vec(9, {0: 1})  # A[0,0]
        v = make_vec(24, {1*8+bcol_start: 1, 1*8+bcol_start+1: 1})  # B[1,bcol_start] + B[1,bcol_start+1]
        w = make_vec(24, {(bcol_start+1)*3+0: 1})  # C[0,bcol_start+1]
        add_term(u, v, w)
        
        # M4 = A[1,1] * B[0,bcol_start+1]
        # Contributes to: C[0,bcol_start] += M4, C[1,bcol_start] += M4
        u = make_vec(9, {4: 1})  # A[1,1]
        v = make_vec(24, {0*8+bcol_start+1: 1})  # B[0,bcol_start+1]
        w = make_vec(24, {bcol_start*3+0: 1, bcol_start*3+1: 1})  # C[0,bcol_start] + C[1,bcol_start]
        add_term(u, v, w)
        
        # M5 = (A[0,0]+A[0,1]) * B[1,bcol_start+1]
        # Contributes to: C[0,bcol_start] -= M5, C[1,bcol_start+1] -= M5
        u = make_vec(9, {0: 1, 1: 1})  # A[0,0] + A[0,1]
        v = make_vec(24, {1*8+bcol_start+1: 1})  # B[1,bcol_start+1]
        w = make_vec(24, {bcol_start*3+0: -1, (bcol_start+1)*3+1: -1})  # -C[0,bcol_start] - C[1,bcol_start+1]
        add_term(u, v, w)
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,bcol_start+1]+B[1,bcol_start+1])
        # Contributes to: C[0,bcol_start+1] += M6
        u = make_vec(9, {3: 1, 0: -1})  # A[1,0] - A[0,0]
        v = make_vec(24, {0*8+bcol_start+1: 1, 1*8+bcol_start+1: 1})  # B[0,bcol_start+1] + B[1,bcol_start+1]
        w = make_vec(24, {(bcol_start+1)*3+0: 1})  # C[0,bcol_start+1]
        add_term(u, v, w)
        
        # M7 = (A[0,1]-A[1,1]) * (B[0,bcol_start]+B[1,bcol_start])
        # Contributes to: C[0,bcol_start] -= M7, C[1,bcol_start+1] += M7
        u = make_vec(9, {1: 1, 4: -1})  # A[0,1] - A[1,1]
        v = make_vec(24, {0*8+bcol_start: 1, 1*8+bcol_start: 1})  # B[0,bcol_start] + B[1,bcol_start]
        w = make_vec(24, {bcol_start*3+0: -1, (bcol_start+1)*3+1: 1})  # -C[0,bcol_start] + C[1,bcol_start+1]
        add_term(u, v, w)
        
        # Now direct terms for the remaining parts of C[0:3, bcol_start:bcol_start+4]
        
        # C[0, bcol_start+2] = A[0,0]*B[0,bcol_start+2] + A[0,1]*B[1,bcol_start+2] + A[0,2]*B[2,bcol_start+2]
        # C[0, bcol_start+3] = A[0,0]*B[0,bcol_start+3] + A[0,1]*B[1,bcol_start+3] + A[0,2]*B[2,bcol_start+3]
        # C[1, bcol_start+2] = A[1,0]*B[0,bcol_start+2] + A[1,1]*B[1,bcol_start+2] + A[1,2]*B[2,bcol_start+2]
        # C[1, bcol_start+3] = A[1,0]*B[0,bcol_start+3] + A[1,1]*B[1,bcol_start+3] + A[1,2]*B[2,bcol_start+3]
        # C[2, bcol_start+0] = A[2,0]*B[0,bcol_start] + A[2,1]*B[1,bcol_start] + A[2,2]*B[2,bcol_start]
        # C[2, bcol_start+1] = A[2,0]*B[0,bcol_start+1] + A[2,1]*B[1,bcol_start+1] + A[2,2]*B[2,bcol_start+1]
        # C[2, bcol_start+2] = A[2,0]*B[0,bcol_start+2] + A[2,1]*B[1,bcol_start+2] + A[2,2]*B[2,bcol_start+2]
        # C[2, bcol_start+3] = A[2,0]*B[0,bcol_start+3] + A[2,1]*B[1,bcol_start+3] + A[2,2]*B[2,bcol_start+3]
        
        # That's 24 direct terms. Plus 7 Strassen terms = 31 per block. Total 62.
        # But wait, I need to be more careful. The Strassen terms already cover C[0:2, bcol_start:bcol_start+2].
        # So I only need direct terms for:
        # - C[0:2, bcol_start+2:bcol_start+4] (4 entries, 12 terms)
        # - C[2, bcol_start:bcol_start+4] (4 entries, 12 terms)
        # Total direct: 24 terms. Total per block: 7+24 = 31. Total: 62.
        
        # C[0, bcol_start+2]
        for j in range(3):
            u = make_vec(9, {j: 1})  # A[0,j]
            v = make_vec(24, {j*8+bcol_start+2: 1})  # B[j,bcol_start+2]
            w = make_vec(24, {(bcol_start+2)*3+0: 1})  # C[0,bcol_start+2]
            add_term(u, v, w)
        
        # C[0, bcol_start+3]
        for j in range(3):
            u = make_vec(9, {j: 1})  # A[0,j]
            v = make_vec(24, {j*8+bcol_start+3: 1})  # B[j,bcol_start+3]
            w = make_vec(24, {(bcol_start+3)*3+0: 1})  # C[0,bcol_start+3]
            add_term(u, v, w)
        
        # C[1, bcol_start+2]
        for j in range(3):
            u = make_vec(9, {3+j: 1})  # A[1,j]
            v = make_vec(24, {j*8+bcol_start+2: 1})  # B[j,bcol_start+2]
            w = make_vec(24, {(bcol_start+2)*3+1: 1})  # C[1,bcol_start+2]
            add_term(u, v, w)
        
        # C[1, bcol_start+3]
        for j in range(3):
            u = make_vec(9, {3+j: 1})  # A[1,j]
            v = make_vec(24, {j*8+bcol_start+3: 1})  # B[j,bcol_start+3]
            w = make_vec(24, {(bcol_start+3)*3+1: 1})  # C[1,bcol_start+3]
            add_term(u, v, w)
        
        # C[2, bcol_start+0]
        for j in range(3):
            u = make_vec(9, {6+j: 1})  # A[2,j]
            v = make_vec(24, {j*8+bcol_start: 1})  # B[j,bcol_start]
            w = make_vec(24, {bcol_start*3+2: 1})  # C[2,bcol_start]
            add_term(u, v, w)
        
        # C[2, bcol_start+1]
        for j in range(3):
            u = make_vec(9, {6+j: 1})  # A[2,j]
            v = make_vec(24, {j*8+bcol_start+1: 1})  # B[j,bcol_start+1]
            w = make_vec(24, {(bcol_start+1)*3+2: 1})  # C[2,bcol_start+1]
            add_term(u, v, w)
        
        # C[2, bcol_start+2]
        for j in range(3):
            u = make_vec(9, {6+j: 1})  # A[2,j]
            v = make_vec(24, {j*8+bcol_start+2: 1})  # B[j,bcol_start+2]
            w = make_vec(24, {(bcol_start+2)*3+2: 1})  # C[2,bcol_start+2]
            add_term(u, v, w)
        
        # C[2, bcol_start+3]
        for j in range(3):
            u = make_vec(9, {6+j: 1})  # A[2,j]
            v = make_vec(24, {j*8+bcol_start+3: 1})  # B[j,bcol_start+3]
            w = make_vec(24, {(bcol_start+3)*3+2: 1})  # C[2,bcol_start+3]
            add_term(u, v, w)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            if u[i] != 0:
                U[i][c] = u[i]
        for i in range(m * p):
            if v[i] != 0:
                V[i][c] = v[i]
        for i in range(p * n):
            if w[i] != 0:
                W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}
