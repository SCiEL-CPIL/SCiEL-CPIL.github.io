import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    R = 68
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    
    def add_term(idx_U, idx_V, idx_W, u_val, v_val, w_val):
        nonlocal col
        U[idx_U, col] = u_val
        V[idx_V, col] = v_val
        W[idx_W, col] = w_val
        col += 1

    # Block 1: 2x2x8 (4x Strassen 2x2x2)
    for pair in range(4):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # Strassen 2x2x2 Rank 7
        # Terms:
        # 1. A00 B00 -> C00
        add_term(0, k0, 3*k0, 1, 1, 1)
        
        # 2. A00 (B10+B11) -> C01
        # This is A00 * B10 and A00 * B11
        # In tensor decomposition, this is not a single rank-1 term unless B10+B11 is a single input?
        # No, the inputs are A and B. The decomposition is on the tensor T.
        # Strassen's algorithm uses 7 multiplications.
        # Each multiplication is of a linear combination of A entries and B entries.
        # This corresponds to a rank-1 term in the tensor decomposition?
        # Yes, if we view the linear combinations as the U and V vectors.
        
        # Term 2: U = A00, V = B10+B11, W = C01
        # U index: 0,0 -> 0
        # V index: 1,k0 and 1,k1? No, V is a vector in the space of B.
        # V vector has 4 entries for 2x2 B.
        # Here B is 2x2 (for the block).
        # V[0] = B00, V[1] = B01, V[2] = B10, V[3] = B11?
        # My V matrix has shape (m*p, R) = (24, R).
        # For the block, j in {0,1}, k in {k0, k1}.
        # V indices: 
        # j=0, k=k0 -> 0*8+k0 = k0
        # j=0, k=k1 -> 0*8+k1 = k1
        # j=1, k=k0 -> 1*8+k0 = 8+k0
        # j=1, k=k1 -> 1*8+k1 = 8+k1
        
        # Term 2: U=A00 (idx 0), V=B10+B11 (idx 8+k0 and 8+k1), W=C01 (idx 3*k1+0? No, C index is k*n+i)
        # C01 in block means i=0, k=k1. W index = k1*3 + 0 = 3*k1.
        # But V is a sum. This is NOT a rank-1 term in the original tensor.
        # Strassen's rank-7 decomposition is a decomposition of the 2x2x2 tensor.
        # The 7 rank-1 tensors are defined by the specific linear forms.
        
        # I will use the explicit rank-7 decomposition of M_2.
        # See: https://en.wikipedia.org/wiki/Strassen_algorithm
        # The 7 products are:
        # M1 = A00 B00
        # M2 = A00 (B10 + B11)
        # M3 = (A10 + A11) B00
        # M4 = A11 (B01 - B00)
        # M5 = (A00 + A11) (B00 + B11)
        # M6 = (A10 - A00) (B00 + B10)
        # M7 = (A01 + A11) (B11 - B01)
        
        # These are NOT rank-1 terms in the tensor decomposition of M_2 directly?
        # Actually, they ARE. Each Mi is a product of a linear form in A and a linear form in B.
        # The contribution to C is a linear combination of Mi.
        # This means the tensor T can be written as sum of 7 rank-1 tensors.
        
        # Term 1: U=A00, V=B00, W=C00
        add_term(0, k0, 3*k0, 1, 1, 1)
        
        # Term 2: U=A00, V=B10+B11, W=C01
        # U idx 0. V idx 8+k0 (1) and 8+k1 (1). W idx 3*k1 (1).
        # But add_term adds ONE rank-1 term.
        # V=B10+B11 is NOT a single entry.
        # So this is NOT a rank-1 term in the standard basis.
        # Strassen's algorithm reduces the number of multiplications, but the tensor decomposition rank is 7.
        # The rank-1 terms in the decomposition are NOT the Mi.
        # The Mi are the 7 scalar multiplications.
        # The tensor decomposition of rank 7 has 7 rank-1 tensors.
        
        # I will use Schoolbook for 2x2x2 (Rank 8) to ensure correctness.
        # Total Rank 72.
        
    # Since I can't easily get the rank-7 Strassen tensor decomposition matrices without a reference, 
    # and the schoolbook is 72, I will return the schoolbook decomposition.
    
    col = 0
    R = 72
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U = i * m + j
                idx_V = j * p + k
                idx_W = k * n + i
                U[idx_U, col] = 1.0
                V[idx_V, col] = 1.0
                W[idx_W, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
