import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    total_terms = n * m * p  # 96
    
    # Schoolbook decomposition
    def schoolbook():
        rank = total_terms
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Strassen-based decomposition
    # M_{3,4,8} can be decomposed using Strassen's 2x2x2 trick
    # Key insight: 3x4 = 2x2 + 1x2, and 8 = 4+4
    # We can use tensor product structure
    
    def strassen_2x2_factors():
        """Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen)."""
        # Strassen's 7 products for 2x2 matrices:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b10)
        # m7 = (a01-a11)*(b01+b11)
        # 
        # U factors (2x2 -> 4 entries), V factors (2x2 -> 4 entries), W factors (2x2 -> 4 entries)
        # For each of the 7 products, we need U[i,j], V[j,k], W[k,i]
        
        U_terms = []
        V_terms = []
        W_terms = []
        
        # m1: (a00+a11)*(b00+b11)
        # U: a00=1, a11=1 -> U[0,0]=1, U[1,1]=1
        # V: b00=1, b11=1 -> V[0,0]=1, V[1,1]=1
        # W: c00 += m1, c11 += m1 -> W[0,0]=1, W[1,1]=1
        U_terms.append([1, 0, 0, 1])
        V_terms.append([1, 0, 0, 1])
        W_terms.append([1, 0, 0, 1])
        
        # m2: (a10+a11)*b00
        # U: a10=1, a11=1 -> U[1,0]=1, U[1,1]=1
        # V: b00=1 -> V[0,0]=1
        # W: c10 += m2 -> W[0,1]=1
        U_terms.append([0, 0, 1, 1])
        V_terms.append([1, 0, 0, 0])
        W_terms.append([0, 1, 0, 0])
        
        # m3: a00*(b01-b11)
        # U: a00=1 -> U[0,0]=1
        # V: b01=1, b11=-1 -> V[0,1]=1, V[1,1]=-1
        # W: c01 += m3 -> W[1,0]=1
        U_terms.append([1, 0, 0, 0])
        V_terms.append([0, 1, 0, -1])
        W_terms.append([0, 0, 1, 0])
        
        # m4: a11*(b10-b00)
        # U: a11=1 -> U[1,1]=1
        # V: b10=1, b00=-1 -> V[1,0]=1, V[0,0]=-1
        # W: c11 += m4 -> W[1,1]=1
        U_terms.append([0, 0, 0, 1])
        V_terms.append([-1, 1, 0, 0])
        W_terms.append([0, 0, 0, 1])
        
        # m5: (a00+a01)*b11
        # U: a00=1, a01=1 -> U[0,0]=1, U[0,1]=1
        # V: b11=1 -> V[1,1]=1
        # W: c01 += m5 -> W[1,0]=1
        U_terms.append([1, 1, 0, 0])
        V_terms.append([0, 0, 0, 1])
        W_terms.append([0, 0, 1, 0])
        
        # m6: (a10-a00)*(b00+b10)
        # U: a10=1, a00=-1 -> U[1,0]=1, U[0,0]=-1
        # V: b00=1, b10=1 -> V[0,0]=1, V[1,0]=1
        # W: c10 += m6 -> W[0,1]=1
        U_terms.append([-1, 0, 1, 0])
        V_terms.append([1, 0, 1, 0])
        W_terms.append([0, 1, 0, 0])
        
        # m7: (a01-a11)*(b01+b11)
        # U: a01=1, a11=-1 -> U[0,1]=1, U[1,1]=-1
        # V: b01=1, b11=1 -> V[0,1]=1, V[1,1]=1
        # W: c11 += m7 -> W[1,1]=1
        U_terms.append([0, 1, 0, -1])
        V_terms.append([0, 1, 0, 1])
        W_terms.append([0, 0, 0, 1])
        
        return np.array(U_terms), np.array(V_terms), np.array(W_terms)
    
    def build_strassen_4x4():
        """
        Build a decomposition for 4x4 x 4x4 matrix multiplication using Strassen.
        4x4 = 2x2 blocks.
        We partition A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
        B (4x4) into 2x2 blocks: B = [[B00, B01], [B10, B11]]
        C (4x4) into 2x2 blocks: C = [[C00, C01], [C10, C11]]
        
        Using Strassen:
        C00 = M1 - M4 + M6 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M3 + M5 + M6
        
        Each Mi is a 2x2x2 tensor product.
        Total rank = 7 * 4 = 28 (since each 2x2x2 has rank 7, and we need 4 such for the 4x4 blocks)
        Wait, that's not right. Let me reconsider.
        
        Actually, for 4x4 x 4x4:
        We have 4 block multiplications of 2x2 x 2x2, but Strassen reduces this to 7 products.
        Each product Mi is itself a 2x2 x 2x2 multiplication (rank 7).
        So total rank = 7 * 7 = 49? No, that's not right either.
        
        Let me think more carefully. The 4x4x4 tensor can be seen as a tensor product:
        M_{4,4,4} = M_{2,2,2} ⊗ M_{2,2,2} in some sense, but Strassen gives a rank-7 decomposition
        of M_{2,2,2}, so M_{4,4,4} has rank at most 7*7 = 49.
        
        But actually, the standard approach is:
        M_{4,4,4} uses 7 Strassen products, each of which is a 2x2x2 multiplication.
        Each 2x2x2 multiplication has rank 7 (Strassen).
        So the total rank is 7 * 7 = 49.
        
        Hmm, but that seems high. Let me reconsider.
        
        Actually, I think the correct way to think about it is:
        - The 4x4 matrix multiplication tensor has a natural decomposition into 16 scalar multiplications (schoolbook).
        - Strassen's algorithm for 2x2x2 gives rank 7.
        - For 4x4x4, we can view it as 4 blocks of 2x2x2, but with Strassen, we need 7 such blocks.
        - Each block is a 2x2x2 tensor, which has rank 7.
        - So total rank = 7 * 7 = 49.
        
        But wait, that's not how it works. The 7 Strassen products are linear combinations of the 4x4 entries,
        and each product is a scalar. So we have 7 scalar products, each involving 4x4 = 16 entries from A and B.
        But each scalar product can be decomposed into at most 4 rank-1 terms (since it's a 2x2x2 tensor).
        So total rank = 7 * 4 = 28? No, that's not right either.
        
        Let me just implement it directly.
        """
        # For now, let's use a simpler approach: schoolbook for 4x4
        # Then we can try to improve
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4 + j, col] = 1
                    V[j*4 + k, col] = 1
                    W[k*4 + i, col] = 1
                    col += 1
        return U, V, W
    
    # Let's try a direct approach for 3x4 x 4x8
    # We can partition the 3x4 matrix as:
    # A = [A_top (2x4); A_bottom (1x4)]
    # And the 4x8 matrix as:
    # B = [B_left (4x4) B_right (4x4)]
    # Then C = A*B is 3x8:
    # C = [C_top (2x8); C_bottom (1x8)]
    # C_top = A_top * B
    # C_bottom = A_bottom * B
    
    # For C_top (2x8 = 2x4 + 2x4):
    # C_top_left = A_top * B_left (2x4 x 4x4 = 2x4)
    # C_top_right = A_top * B_right (2x4 x 4x4 = 2x4)
    
    # For C_bottom (1x8 = 1x4 + 1x4):
    # C_bottom_left = A_bottom * B_left (1x4 x 4x4 = 1x4)
    # C_bottom_right = A_bottom * B_right (1x4 x 4x4 = 1x4)
    
    # Each of these can be decomposed using Strassen or other methods.
    
    # Let's try to use the known result that M_{3,4,8} has rank at most 96 (schoolbook)
    # and see if we can do better.
    
    # For now, return schoolbook as baseline
    U, V, W = schoolbook()
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
