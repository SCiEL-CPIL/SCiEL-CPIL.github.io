"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use block structure: split each dimension into 2 and 3
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    # B = [[B11, B12], [B21, B22]] similarly
    # C = A*B has blocks:
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x3)
    # C21 = A21*B11 + A22*B21  (3x2)
    # C22 = A21*B12 + A22*B22  (3x3)
    
    # Use Strassen for 2x2x2 blocks (rank 7)
    # Use schoolbook for mixed-size blocks
    
    # Collect all rank-1 terms
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Strassen for 2x2x2: computes C = A*B where A,B are 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 2x2x2 Strassen, the 7 rank-1 terms have:
    # Each term has U, V, W vectors of appropriate dimensions
    
    # Let's define Strassen terms for 2x2x2 with dimensions n=2,m=2,p=2
    def strassen_2x2x2_terms(offset_i, offset_j, offset_k):
        """Generate the 7 Strassen rank-1 terms for a 2x2x2 sub-problem.
        offset_i, offset_j, offset_k are the starting indices in the full matrix."""
        terms = []
        # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
        # U: A11+A22 means positions (offset_i, offset_j) and (offset_i+1, offset_j+1)
        u1 = [0]*5
        u1[offset_i + offset_j] = 1
        u1[(offset_i+1) + (offset_j+1)] = 1
        v1 = [0]*5
        v1[offset_j + offset_k] = 1
        v1[(offset_j+1) + (offset_k+1)] = 1
        w1 = [0]*5
        w1[offset_k + offset_i] = 1
        w1[(offset_k+1) + (offset_i+1)] = 1
        # M1 contributes +1 to C11 and +1 to C22
        # So W has +1 at (offset_k+offset_i) and +1 at ((offset_k+1)+(offset_i+1))
        terms.append((u1, v1, w1))
        
        # M2 = (A21+A22)B11 -> contributes to C21 and C22
        u2 = [0]*5
        u2[(offset_i+1) + offset_j] = 1
        u2[(offset_i+1) + (offset_j+1)] = 1
        v2 = [0]*5
        v2[offset_j + offset_k] = 1
        w2 = [0]*5
        w2[offset_k + (offset_i+1)] = 1  # C21
        w2[(offset_k+1) + (offset_i+1)] = -1  # C22 (with -1 sign)
        terms.append((u2, v2, w2))
        
        # M3 = A11(B12-B22) -> contributes to C12 and C22
        u3 = [0]*5
        u3[offset_i + offset_j] = 1
        v3 = [0]*5
        v3[offset_j + (offset_k+1)] = 1
        v3[(offset_j+1) + (offset_k+1)] = -1
        w3 = [0]*5
        w3[(offset_k+1) + offset_i] = 1  # C12
        w3[(offset_k+1) + (offset_i+1)] = 1  # C22
        terms.append((u3, v3, w3))
        
        # M4 = A22(B21-B11) -> contributes to C11 and C21
        u4 = [0]*5
        u4[(offset_i+1) + (offset_j+1)] = 1
        v4 = [0]*5
        v4[(offset_j+1) + offset_k] = 1
        v4[offset_j + offset_k] = -1
        w4 = [0]*5
        w4[offset_k + offset_i] = 1  # C11
        w4[offset_k + (offset_i+1)] = 1  # C21
        terms.append((u4, v4, w4))
        
        # M5 = (A11+A12)B22 -> contributes to C11 and C12
        u5 = [0]*5
        u5[offset_i + offset_j] = 1
        u5[offset_i + (offset_j+1)] = 1
        v5 = [0]*5
        v5[(offset_j+1) + (offset_k+1)] = 1
        w5 = [0]*5
        w5[offset_k + offset_i] = -1  # C11
        w5[(offset_k+1) + offset_i] = 1  # C12
        terms.append((u5, v5, w5))
        
        # M6 = (A21-A11)(B11+B12) -> contributes to C22
        u6 = [0]*5
        u6[(offset_i+1) + offset_j] = 1
        u6[offset_i + offset_j] = -1
        v6 = [0]*5
        v6[offset_j + offset_k] = 1
        v6[offset_j + (offset_k+1)] = 1
        w6 = [0]*5
        w6[(offset_k+1) + (offset_i+1)] = 1  # C22
        terms.append((u6, v6, w6))
        
        # M7 = (A12-A22)(B21+B22) -> contributes to C11
        u7 = [0]*5
        u7[offset_i + (offset_j+1)] = 1
        u7[(offset_i+1) + (offset_j+1)] = -1
        v7 = [0]*5
        v7[(offset_j+1) + offset_k] = 1
        v7[(offset_j+1) + (offset_k+1)] = 1
        w7 = [0]*5
        w7[offset_k + offset_i] = 1  # C11
        terms.append((u7, v7, w7))
        
        return terms
    
    # Now build the full 5x5x5 decomposition
    # Split into blocks: indices 0-1 are "block 1" (size 2), indices 2-4 are "block 2" (size 3)
    
    # C11 = A11*B11 + A12*B21 (2x2 result)
    # A11*B11: use Strassen on 2x2x2 with offsets (0,0,0)
    # A12*B21: A12 is 2x3, B21 is 3x2 -> schoolbook 2*3*2 = 12 terms
    
    # C12 = A11*B12 + A12*B22 (2x3 result)
    # A11*B12: A11 is 2x2, B12 is 2x3 -> schoolbook 2*2*3 = 12 terms
    # A12*B22: A12 is 2x3, B22 is 3x3 -> schoolbook 2*3*3 = 18 terms
    
    # C21 = A21*B11 + A22*B21 (3x2 result)
    # A21*B11: A21 is 3x2, B11 is 2x2 -> schoolbook 3*2*2 = 12 terms
    # A22*B21: A22 is 3x3, B21 is 3x2 -> schoolbook 3*3*2 = 18 terms
    
    # C22 = A21*B12 + A22*B22 (3x3 result)
    # A21*B12: A21 is 3x2, B12 is 2x3 -> schoolbook 3*2*3 = 18 terms
    # A22*B22: A22 is 3x3, B22 is 3x3 -> schoolbook 3*3*3 = 27 terms
    
    # Total schoolbook for non-Strassen parts: 12+12+18+12+18+18+27 = 117
    # Plus 7 for Strassen on A11*B11
    # Total: 117 + 7 = 124
    
    # That's only 1 less than schoolbook 125. Let me reconsider.
    
    # Actually, I need to be more careful. Let me just use a cleaner approach.
    # Let me use Strassen for ALL 2x2x2 sub-blocks that are pure 2x2.
    
    # The 4 2x2x2 sub-products that are pure 2x2*2x2:
    # - A11*B11 (for C11 part)
    # But wait, C11 = A11*B11 + A12*B21, so A11*B11 is 2x2*2x2 -> 2x2 result
    # Similarly, are there other pure 2x2*2x2 products? No, because the other blocks are mixed sizes.
    
    # So we only save 64-7=57 on one block, but add 12 for A12*B21
    # Net: 125 - 64 + 7 + 12 = 125 - 45 = 80? No, that's wrong.
    
    # Schoolbook total: 125 terms
    # If I replace A11*B11 (which is 8 schoolbook terms) with Strassen (7 terms), I save 1.
    # Total: 125 - 8 + 7 = 124
    
    # That's marginal. Let me think of a better approach.
    
    # Alternative: use a completely different decomposition strategy.
    # Let me just implement a correct schoolbook and see if I can find any simple improvement.
    
    # For now, let me return a correct schoolbook decomposition to ensure validity.
    
    U_full = []
    V_full = []
    W_full = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                U_full.append(u)
                V_full.append(v)
                W_full.append(w)
    
    U_mat = np.array(U_full).T  # shape (n*m, rank)
    V_mat = np.array(V_full).T  # shape (m*p, rank)
    W_mat = np.array(W_full).T  # shape (p*n, rank)
    
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
# EVOLVE-BLOCK-END
