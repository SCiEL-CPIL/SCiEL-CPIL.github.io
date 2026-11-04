import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _get_strassen_2x2_decomposition():
    """
    Returns U, V, W for 2x2x2 multiplication with rank 7 (Strassen).
    Indices:
    A: 2x2 -> a = i*2 + j (i,j in 0..1)
    B: 2x2 -> b = j*2 + k (j,k in 0..1)
    C: 2x2 -> c = k*2 + i (k,i in 0..1)
    """
    # Strassen 7 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    U = []
    V = []
    W = []
    
    # Term 1: M1
    # A part: A00 + A11 -> [1, 0, 0, 1]
    # B part: B00 + B11 -> [1, 0, 0, 1]
    # C part: M1 contributes to C00 (+1), C11 (+1)
    # C00 index: 0*2+0=0. C11 index: 1*2+1=3.
    # So W has 1 at 0 and 3.
    U.append([1, 0, 0, 1])
    V.append([1, 0, 0, 1])
    W.append([1, 0, 0, 1]) # C00=1, C01=0, C10=0, C11=1
    
    # Term 2: M2
    # A part: A10 + A11 -> [0, 0, 1, 1]
    # B part: B00 -> [1, 0, 0, 0]
    # C part: M2 contributes to C10 (+1)
    # C10 index: 0*2+1=1.
    U.append([0, 0, 1, 1])
    V.append([1, 0, 0, 0])
    W.append([0, 1, 0, 0])
    
    # Term 3: M3
    # A part: A00 -> [1, 0, 0, 0]
    # B part: B01 - B11 -> [0, 1, 0, -1]
    # C part: M3 contributes to C01 (+1), C11 (-1)
    # C01 index: 1*2+0=2. C11 index: 3.
    U.append([1, 0, 0, 0])
    V.append([0, 1, 0, -1])
    W.append([0, 0, 1, -1])
    
    # Term 4: M4
    # A part: A11 -> [0, 0, 0, 1]
    # B part: B10 - B00 -> [-1, 0, 1, 0]
    # C part: M4 contributes to C00 (-1), C10 (+1)
    # C00 index: 0. C10 index: 1.
    U.append([0, 0, 0, 1])
    V.append([-1, 0, 1, 0])
    W.append([-1, 1, 0, 0])
    
    # Term 5: M5
    # A part: A00 + A01 -> [1, 1, 0, 0]
    # B part: B11 -> [0, 0, 0, 1]
    # C part: M5 contributes to C01 (+1), C11 (+1)
    # C01 index: 2. C11 index: 3.
    U.append([1, 1, 0, 0])
    V.append([0, 0, 0, 1])
    W.append([0, 0, 1, 1])
    
    # Term 6: M6
    # A part: A10 - A00 -> [-1, 0, 1, 0]
    # B part: B00 + B01 -> [1, 1, 0, 0]
    # C part: M6 contributes to C00 (+1)
    # C00 index: 0.
    U.append([-1, 0, 1, 0])
    V.append([1, 1, 0, 0])
    W.append([1, 0, 0, 0])
    
    # Term 7: M7
    # A part: A01 - A11 -> [0, 1, 0, -1]
    # B part: B10 + B11 -> [0, 0, 1, 1]
    # C part: M7 contributes to C00 (-1), C11 (+1)
    # C00 index: 0. C11 index: 3.
    U.append([0, 1, 0, -1])
    V.append([0, 0, 1, 1])
    W.append([-1, 0, 0, 1])
    
    return np.array(U), np.array(V), np.array(W)

def _get_strassen_4x4_decomposition():
    """
    Returns U, V, W for 4x4x4 multiplication using Strassen blocks (Rank 49).
    A: 4x4 (16)
    B: 4x4 (16)
    C: 4x4 (16)
    """
    U_2, V_2, W_2 = _get_strassen_2x2_decomposition()
    
    U = []
    V = []
    W = []
    
    # We have 7 Strassen "steps" for the 4x4 block multiplication.
    # Each step is a 2x2x2 multiplication.
    # We need to map the 2x2 indices to the 4x4 indices.
    
    # Strassen 4x4 formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # For each Mk, we use the 7 terms from U_2, V_2, W_2.
    # We need to embed the 2x2 vectors into 4x4 vectors.
    
    def embed_2x2_to_4x4(vec_2x2, block_i, block_j, sign=1):
        """
        Embeds a 4-element vector (2x2 matrix) into a 16-element vector (4x4 matrix)
        at the specified 2x2 block position (block_i, block_j).
        block_i, block_j in {0, 1}.
        """
        vec_4x4 = np.zeros(16)
        # Indices in 4x4: i*4 + j
        # Block (block_i, block_j) covers rows 2*block_i..2*block_i+1, cols 2*block_j..2*block_j+1
        for i in range(2):
            for j in range(2):
                idx_2 = i * 2 + j
                idx_4 = (2 * block_i + i) * 4 + (2 * block_j + j)
                vec_4x4[idx_4] = vec_2x2[idx_2] * sign
        return vec_4x4

    # Define the 7 Strassen terms for 4x4
    # Each term is a tuple (U_block_info, V_block_info, W_block_info)
    # U_block_info: list of (block_i, block_j, sign)
    # V_block_info: list of (block_j, block_k, sign) -- Note: B is 4x4, so blocks are (row_block, col_block)
    # W_block_info: list of (block_k, block_i, sign) -- Note: C is 4x4, so blocks are (row_block, col_block)
    
    # M1: A00+A11, B00+B11, C00+C11
    # A parts: (0,0,1), (1,1,1)
    # B parts: (0,0,1), (1,1,1)
    # C parts: (0,0,1), (1,1,1)
    
    # M2: A10+A11, B00, C10
    # A parts: (1,0,1), (1,1,1)
    # B parts: (0,0,1)
    # C parts: (1,0,1)
    
    # M3: A00, B01-B11, C01-C11
    # A parts: (0,0,1)
    # B parts: (0,1,1), (1,1,-1)
    # C parts: (0,1,1), (1,1,-1)
    
    # M4: A11, B10-B00, C00-C10
    # A parts: (1,1,1)
    # B parts: (1,0,1), (0,0,-1)
    # C parts: (0,0,-1), (1,0,1)
    
    # M5: A00+A01, B11, C01+C11
    # A parts: (0,0,1), (0,1,1)
    # B parts: (1,1,1)
    # C parts: (0,1,1), (1,1,1)
    
    # M6: A10-A00, B00+B01, C00
    # A parts: (1,0,1), (0,0,-1)
    # B parts: (0,0,1), (0,1,1)
    # C parts: (0,0,1)
    
    # M7: A01-A11, B10+B11, C00-C11
    # A parts: (0,1,1), (1,1,-1)
    # B parts: (1,0,1), (1,1,1)
    # C parts: (0,0,-1), (1,1,1)
    
    strassen_terms = [
        { # M1
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        { # M2
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(1,0,1)]
        },
        { # M3
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        { # M4
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        { # M5
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,1,1), (1,1,1)]
        },
        { # M6
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(0,0,1)]
        },
        { # M7
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,-1), (1,1,1)]
        }
    ]
    
    # For each of the 7 Strassen steps, we have 7 inner terms.
    for step in strassen_terms:
        for t in range(7):
            u_2 = U_2[t]
            v_2 = V_2[t]
            w_2 = W_2[t]
            
            u_4 = np.zeros(16)
            for bi, bj, s in step['U']:
                u_4 += embed_2x2_to_4x4(u_2, bi, bj, s)
                
            v_4 = np.zeros(16)
            for bj, bk, s in step['V']:
                v_4 += embed_2x2_to_4x4(v_2, bj, bk, s)
                
            w_4 = np.zeros(16)
            for bk, bi, s in step['W']:
                w_4 += embed_2x2_to_4x4(w_2, bk, bi, s)
                
            U.append(u_4)
            V.append(v_4)
            W.append(w_4)
            
    return np.array(U), np.array(V), np.array(W)

def _get_schoolbook_4x4x3_decomposition():
    """
    Returns U, V, W for 4x4x3 multiplication using Schoolbook (Rank 48).
    A: 4x4 (16)
    B: 4x3 (12)
    C: 4x3 (12)
    """
    U = []
    V = []
    W = []
    
    # A[i,j], B[j,k], C[i,k]
    # i in 0..3, j in 0..3, k in 0..2
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = np.zeros(16)
                v = np.zeros(12)
                w = np.zeros(12)
                
                u[i*4 + j] = 1.0
                v[j*3 + k] = 1.0
                w[k*4 + i] = 1.0
                
                U.append(u)
                V.append(v)
                W.append(w)
                
    return np.array(U), np.array(V), np.array(W)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build 4x4x4 part (Rank 49)
    U_444, V_444, W_444 = _get_strassen_4x4_decomposition()
    
    # Build 4x4x3 part (Rank 48)
    U_443, V_443, W_443 = _get_schoolbook_4x4x3_decomposition()
    
    # Concatenate
    # U: 16 x 49 and 16 x 48 -> 16 x 97
    # V: 16 x 49 and 12 x 48 -> Wait, V dimensions must match.
    # V for 4x4x4 is 16 (4x4).
    # V for 4x4x3 is 12 (4x3).
    # They are different!
    # The decomposition is for the tensor T[16, 12, 12].
    # The 4x4x4 part uses B indices 0..3 (4 cols).
    # The 4x4x3 part uses B indices 4..6 (3 cols).
    # So V_444 should be padded to 12? No, V_444 has support on first 4 cols of B.
    # V_443 has support on last 3 cols of B.
    # We can just concatenate the V vectors by padding the shorter one with zeros.
    
    # V_444 is 16x49. We need it to be 12x49? No, B is 4x7, so mp=28.
    # Wait, B is 4x7. mp = 4*7 = 28.
    # My 4x4x4 decomposition assumed B was 4x4.
    # My 4x4x3 decomposition assumed B was 4x3.
    # We need to map these to the full 4x7 B.
    
    # Let's re-evaluate the dimensions.
    # A: 4x4 (16)
    # B: 4x7 (28)
    # C: 4x7 (28)
    
    # Part 1: 4x4x4 multiplication (cols 0-3 of B and C)
    # U_444: 16x49
    # V_444_local: 16x49 (for 4x4 B)
    # W_444_local: 16x49 (for 4x4 C)
    
    # Part 2: 4x4x3 multiplication (cols 4-6 of B and C)
    # U_443: 16x48
    # V_443_local: 12x48 (for 4x3 B)
    # W_443_local: 12x48 (for 4x3 C)
    
    # Full V: 28x97
    # Full W: 28x97
    
    V_full = np.zeros((28, 97))
    W_full = np.zeros((28, 97))
    
    # Fill Part 1 (cols 0-3)
    # V_444_local has indices j*4+k where j in 0..3, k in 0..3.
    # Full V indices j*7+k.
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V_full[j*7 + k, r] = V_444[j*4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[k*7 + i, r] = W_444[k*4 + i, r]
                
    # Fill Part 2 (cols 4-6)
    # V_443_local has indices j*3+k where j in 0..3, k in 0..2.
    # Full V indices j*7+(4+k).
    for r in range(48):
        for j in range(4):
            for k in range(3):
                V_full[j*7 + (4+k), 49+r] = V_443[j*3 + k, r]
        for k in range(3):
            for i in range(4):
                W_full[(4+k)*7 + i, 49+r] = W_443[k*4 + i, r]
                
    U_full = np.concatenate([U_444, U_443], axis=1)
    
    # Verify
    T = np.zeros((16, 28, 28))
    for i in range(4):
        for j in range(4):
            for k in range(7):
                T[i*4 + j, j*7 + k, k*4 + i] = 1.0
                
    T_dec = np.zeros((16, 28, 28))
    for r in range(97):
        T_dec += np.einsum('a,b,c->abc', U_full[:, r], V_full[:, r], W_full[:, r])
        
    err = np.max(np.abs(T - T_dec))
    if err > 1e-10:
        # Fallback to schoolbook 112
        U_school = np.zeros((16, 112))
        V_school = np.zeros((28, 112))
        W_school = np.zeros((28, 112))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_school[i*4 + j, col] = 1.0
                    V_school[j*7 + k, col] = 1.0
                    W_school[k*7 + i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# EVOLVE-BLOCK-END
