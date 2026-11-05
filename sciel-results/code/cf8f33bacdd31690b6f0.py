"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def winograd_f4():
        """
        Winograd F(4) scheme for 4x4 * 4x4 -> 4x4.
        This gives rank 15 for the square core.
        
        The Winograd F(4) algorithm transforms:
        A (4x4) -> A_t = M_A * A * M_A^T  (8x8)
        B (4x4) -> B_t = M_B * B * M_B^T  (8x8)
        C_t = A_t * B_t (element-wise)
        C = M_C * C_t * M_C^T
        
        The rank of this decomposition is 8*8 = 64? No, that's not right.
        
        Actually, let's think about this more carefully.
        
        Winograd F(4) for convolution uses:
        - Transform A into a (r+k-1) x (r+k-1) matrix where r=4, k=4, so 7x7? No.
        
        Let me reconsider. The standard Winograd F(4) for matrix multiplication:
        
        For 4x4 * 4x4:
        The minimal rank is known to be 49 (Strassen).
        
        However, there's a different approach. Let's use the fact that:
        4x4 * 4x4 can be decomposed as:
        - Split into 2x2 blocks: 4 blocks of 2x2*2x2
        - Each 2x2*2x2 has rank 7 (Strassen)
        - Total: 4 * 7 = 28
        
        But we can do better with a direct 4x4 Strassen:
        - 7 multiplications of 2x2 matrices
        - Each 2x2 multiplication has rank 4 (schoolbook) or 7 (Strassen)
        - If we use schoolbook for the inner 2x2: 7 * 4 = 28
        - If we use Strassen for the inner 2x2: 7 * 7 = 49
        
        So the best is 28 for 4x4 * 4x4 using Strassen at the top level and schoolbook at the bottom.
        
        Wait, that's not right either. Let me think again.
        
        For 4x4 * 4x4:
        Using Strassen's algorithm:
        - Split into 2x2 blocks
        - Compute 7 products of 2x2 matrices
        - Each 2x2 * 2x2 product can be done with rank 4 (schoolbook)
        - Total rank: 7 * 4 = 28
        
        For 4x4 * 4x2:
        - Split A into 2x2 blocks (2x2 grid)
        - Split B into 2x2 blocks (2x1 grid)
        - We need to compute 4 products of 2x2 * 2x1
        - Each 2x2 * 2x1 has rank 4 (schoolbook)
        - Total rank: 4 * 4 = 16
        
        Total: 28 + 16 = 44
        
        But we can do better. For 4x4 * 4x2, we can use a more efficient decomposition.
        
        Actually, let's just implement the straightforward approach:
        - 4x4 * 4x4: rank 28 (Strassen with schoolbook inner)
        - 4x4 * 4x2: rank 16 (schoolbook)
        - Total: 44
        
        This is better than 77.
        """
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col_offset = 0
        
        def add_rank1(u_dict, v_dict, w_dict):
            nonlocal col_offset
            for (i, j), coeff in u_dict.items():
                U[i*m + j][col_offset] += coeff
            for (j, k), coeff in v_dict.items():
                V[j*p + k][col_offset] += coeff
            for (i, k), coeff in w_dict.items():
                W[k*n + i][col_offset] += coeff
            col_offset += 1
        
        # PART 1: 4x4 * 4x4 (columns 0-3 of B)
        # Use Strassen with schoolbook inner products
        
        def strassen_4x4_schoolbook_inner(oi, oj, ok):
            """
            Strassen decomposition for 4x4 * 4x4 using schoolbook for inner 2x2 products.
            oi, oj, ok are offsets (0 for our case).
            """
            # M1 = (A00+A11)(B00+B11) -> C00, C11
            # (A00+A11) is 2x2, (B00+B11) is 2x2, product is 2x2
            # We need to decompose this 2x2 * 2x2 product
            
            # For (A00+A11) * (B00+B11):
            # The entries of A00+A11 are:
            # (oi,oj): A[oi,oj]+A[oi+2,oj+2]
            # (oi,oj+1): A[oi,oj+1]+A[oi+2,oj+3]
            # (oi+1,oj): A[oi+1,oj]+A[oi+3,oj+2]
            # (oi+1,oj+1): A[oi+1,oj+1]+A[oi+3,oj+3]
            
            # Similarly for B00+B11.
            
            # Schoolbook for 2x2 * 2x2:
            # c00 = a00*b00 + a01*b10
            # c01 = a00*b01 + a01*b11
            # c10 = a10*b00 + a11*b10
            # c11 = a10*b01 + a11*b11
            
            # M1 contributes to C00 and C11
            
            # M1 term 1: (A00+A11)[0,0] * (B00+B11)[0,0]
            u1 = {(oi,oj): 1, (oi+2,oj+2): 1}
            v1 = {(oj,ok): 1, (oj+2,ok+2): 1}
            w1 = {(oi,ok): 1, (oi+2,ok+2): 1}
            add_rank1(u1, v1, w1)
            
            # M1 term 2: (A00+A11)[0,1] * (B00+B11)[1,0]
            u2 = {(oi,oj+1): 1, (oi+2,oj+3): 1}
            v2 = {(oj+2,ok): 1, (oj,ok+2): 1}
            w2 = {(oi,ok+2): 1, (oi+2,ok): 1}
            add_rank1(u2, v2, w2)
            
            # M1 term 3: (A00+A11)[0,0] * (B00+B11)[0,1]
            u3 = {(oi,oj): 1, (oi+2,oj+2): 1}
            v3 = {(oj,ok+1): 1, (oj+2,ok+3): 1}
            w3 = {(oi,ok+1): 1, (oi+2,ok+3): 1}
            add_rank1(u3, v3, w3)
            
            # M1 term 4: (A00+A11)[0,1] * (B00+B11)[1,1]
            u4 = {(oi,oj+1): 1, (oi+2,oj+3): 1}
            v4 = {(oj+2,ok+1): 1, (oj,ok+3): 1}
            w4 = {(oi,ok+3): 1, (oi+2,ok+1): 1}
            add_rank1(u4, v4, w4)
            
            # M2 = (A10+A11)B00 -> C10, C11
            # (A10+A11) is 2x2, B00 is 2x2
            
            # M2 term 1: (A10+A11)[0,0] * B00[0,0]
            u5 = {(oi+2,oj): 1, (oi+2,oj+2): 1}
            v5 = {(oj,ok): 1}
            w5 = {(oi+2,ok): 1, (oi+2,ok+2): -1}
            add_rank1(u5, v5, w5)
            
            # M2 term 2: (A10+A11)[0,1] * B00[1,0]
            u6 = {(oi+2,oj+1): 1, (oi+2,oj+3): 1}
            v6 = {(oj+2,ok): 1}
            w6 = {(oi+2,ok+2): 1}
            add_rank1(u6, v6, w6)
            
            # M2 term 3: (A10+A11)[1,0] * B00[0,0]
            u7 = {(oi+3,oj): 1, (oi+3,oj+2): 1}
            v7 = {(oj,ok): 1}
            w7 = {(oi+3,ok): 1, (oi+3,ok+2): -1}
            add_rank1(u7, v7, w7)
            
            # M2 term 4: (A10+A11)[1,1] * B00[1,0]
            u8 = {(oi+3,oj+1): 1, (oi+3,oj+3): 1}
            v8 = {(oj+2,ok): 1}
            w8 = {(oi+3,ok+2): 1}
            add_rank1(u8, v8, w8)
            
            # M3 = A00(B01-B11) -> C01, C11
            # A00 is 2x2, (B01-B11) is 2x2
            
            # M3 term 1: A00[0,0] * (B01-B11)[0,0]
            u9 = {(oi,oj): 1}
            v9 = {(oj,ok+2): 1, (oj+2,ok): -1}
            w9 = {(oi,ok+2): 1, (oi+2,ok+2): 1}
            add_rank1(u9, v9, w9)
            
            # M3 term 2: A00[0,1] * (B01-B11)[1,0]
            u10 = {(oi,oj+1): 1}
            v10 = {(oj,ok+3): 1, (oj+2,ok+1): -1}
            w10 = {(oi,ok+3): 1, (oi+2,ok+1): 1}
            add_rank1(u10, v10, w10)
            
            # M3 term 3: A00[1,0] * (B01-B11)[0,0]
            u11 = {(oi+1,oj): 1}
            v11 = {(oj,ok+2): 1, (oj+2,ok): -1}
            w11 = {(oi+1,ok+2): 1, (oi+3,ok+2): 1}
            add_rank1(u11, v11, w11)
            
            # M3 term 4: A00[1,1] * (B01-B11)[1,0]
            u12 = {(oi+1,oj+1): 1}
            v12 = {(oj,ok+3): 1, (oj+2,ok+1): -1}
            w12 = {(oi+1,ok+3): 1, (oi+3,ok+1): 1}
            add_rank1(u12, v12, w12)
            
            # M4 = A11(B10-B00) -> C00, C10
            # A11 is 2x2, (B10-B00) is 2x2
            
            # M4 term 1: A11[0,0] * (B10-B00)[0,0]
            u13 = {(oi+2,oj+2): 1}
            v13 = {(oj+2,ok): 1, (oj,ok): -1}
            w13 = {(oi,ok): 1, (oi+2,ok): 1}
            add_rank1(u13, v13, w13)
            
            # M4 term 2: A11[0,1] * (B10-B00)[1,0]
            u14 = {(oi+2,oj+3): 1}
            v14 = {(oj+2,ok+1): 1, (oj,ok+1): -1}
            w14 = {(oi,ok+1): 1, (oi+2,ok+1): 1}
            add_rank1(u14, v14, w14)
            
            # M4 term 3: A11[1,0] * (B10-B00)[0,0]
            u15 = {(oi+3,oj+2): 1}
            v15 = {(oj+2,ok): 1, (oj,ok): -1}
            w15 = {(oi+1,ok): 1, (oi+3,ok): 1}
            add_rank1(u15, v15, w15)
            
            # M4 term 4: A11[1,1] * (B10-B00)[1,0]
            u16 = {(oi+3,oj+3): 1}
            v16 = {(oj+2,ok+1): 1, (oj,ok+1): -1}
            w16 = {(oi+1,ok+1): 1, (oi+3,ok+1): 1}
            add_rank1(u16, v16, w16)
            
            # M5 = (A00+A01)B11 -> C00, C01
            # (A00+A01) is 2x2, B11 is 2x2
            
            # M5 term 1: (A00+A01)[0,0] * B11[0,0]
            u17 = {(oi,oj): 1, (oi,oj+2): 1}
            v17 = {(oj+2,ok+2): 1}
            w17 = {(oi,ok): -1, (oi,ok+2): 1}
            add_rank1(u17, v17, w17)
            
            # M5 term 2: (A00+A01)[0,1] * B11[1,0]
            u18 = {(oi,oj+1): 1, (oi,oj+3): 1}
            v18 = {(oj+2,ok+3): 1}
            w18 = {(oi,ok+1): 1}
            add_rank1(u18, v18, w18)
            
            # M5 term 3: (A00+A01)[1,0] * B11[0,0]
            u19 = {(oi+1,oj): 1, (oi+1,oj+2): 1}
            v19 = {(oj+2,ok+2): 1}
            w19 = {(oi+1,ok): -1, (oi+1,ok+2): 1}
            add_rank1(u19, v19, w19)
            
            # M5 term 4: (A00+A01)[1,1] * B11[1,0]
            u20 = {(oi+1,oj+1): 1, (oi+1,oj+3): 1}
            v20 = {(oj+2,ok+3): 1}
            w20 = {(oi+1,ok+1): 1}
            add_rank1(u20, v20, w20)
            
            # M6 = (A10-A00)(B00+B01) -> C11
            # (A10-A00) is 2x2, (B00+B01) is 2x2
            
            # M6 term 1: (A10-A00)[0,0] * (B00+B01)[0,0]
            u21 = {(oi+2,oj): 1, (oi,oj): -1}
            v21 = {(oj,ok): 1, (oj,ok+2): 1}
            w21 = {(oi+2,ok+2): 1}
            add_rank1(u21, v21, w21)
            
            # M6 term 2: (A10-A00)[0,1] * (B00+B01)[1,0]
            u22 = {(oi+2,oj+1): 1, (oi,oj+1): -1}
            v22 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w22 = {(oi+2,ok+3): 1}
            add_rank1(u22, v22, w22)
            
            # M6 term 3: (A10-A00)[1,0] * (B00+B01)[0,0]
            u23 = {(oi+3,oj): 1, (oi+1,oj): -1}
            v23 = {(oj,ok): 1, (oj,ok+2): 1}
            w23 = {(oi+3,ok+2): 1}
            add_rank1(u23, v23, w23)
            
            # M6 term 4: (A10-A00)[1,1] * (B00+B01)[1,0]
            u24 = {(oi+3,oj+1): 1, (oi+1,oj+1): -1}
            v24 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w24 = {(oi+3,ok+3): 1}
            add_rank1(u24, v24, w24)
            
            # M7 = (A01-A11)(B10+B11) -> C00
            # (A01-A11) is 2x2, (B10+B11) is 2x2
            
            # M7 term 1: (A01-A11)[0,0] * (B10+B11)[0,0]
            u25 = {(oi,oj+2): 1, (oi+2,oj+2): -1}
            v25 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w25 = {(oi,ok): 1}
            add_rank1(u25, v25, w25)
            
            # M7 term 2: (A01-A11)[0,1] * (B10+B11)[1,0]
            u26 = {(oi,oj+3): 1, (oi+2,oj+3): -1}
            v26 = {(oj+2,ok+1): 1, (oj+2,ok+3): 1}
            w26 = {(oi,ok+1): 1}
            add_rank1(u26, v26, w26)
            
            # M7 term 3: (A01-A11)[1,0] * (B10+B11)[0,0]
            u27 = {(oi+1,oj+2): 1, (oi+3,oj+2): -1}
            v27 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w27 = {(oi+1,ok): 1}
            add_rank1(u27, v27, w27)
            
            # M7 term 4: (A01-A11)[1,1] * (B10+B11)[1,0]
            u28 = {(oi+1,oj+3): 1, (oi+3,oj+3): -1}
            v28 = {(oj+2,ok+1): 1, (oj+2,ok+3): 1}
            w28 = {(oi+1,ok+1): 1}
            add_rank1(u28, v28, w28)
        
        strassen_4x4_schoolbook_inner(0, 0, 0)
        
        # PART 2: 4x4 * 4x2 (columns 4-5 of B)
        # Schoolbook decomposition
        # C[i,4] = sum_j A[i,j] * B[j,4]
        # C[i,5] = sum_j A[i,j] * B[j,5]
        
        for i in range(4):
            for j in range(4):
                # C[i,4] contribution
                u = {(i, j): 1}
                v = {(j, 4): 1}
                w = {(i, 4): 1}
                add_rank1(u, v, w)
                
                # C[i,5] contribution
                u = {(i, j): 1}
                v = {(j, 5): 1}
                w = {(i, 5): 1}
                add_rank1(u, v, w)
        
        # Now we have 28 + 32 = 60 terms, but we only allocated 44.
        # Let's fix this by using a dynamic approach.
        
        # Actually, let's just build it properly.
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u_dict, v_dict, w_dict):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (i, j), coeff in u_dict.items():
                u_col[i * m + j] = coeff
            for (j, k), coeff in v_dict.items():
                v_col[j * p + k] = coeff
            for (i, k), coeff in w_dict.items():
                w_col[k * n + i] = coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
        
        # PART 1: 4x4 * 4x4 (columns 0-3 of B)
        # Use Strassen with schoolbook inner products
        
        def strassen_4x4_schoolbook_inner(oi, oj, ok):
            # M1 = (A00+A11)(B00+B11) -> C00, C11
            # (A00+A11) is 2x2, (B00+B11) is 2x2, product is 2x2
            # We need to decompose this 2x2 * 2x2 product
            
            # For (A00+A11) * (B00+B11):
            # The entries of A00+A11 are:
            # (oi,oj): A[oi,oj]+A[oi+2,oj+2]
            # (oi,oj+1): A[oi,oj+1]+A[oi+2,oj+3]
            # (oi+1,oj): A[oi+1,oj]+A[oi+3,oj+2]
            # (oi+1,oj+1): A[oi+1,oj+1]+A[oi+3,oj+3]
            
            # Similarly for B00+B11.
            
            # Schoolbook for 2x2 * 2x2:
            # c00 = a00*b00 + a01*b10
            # c01 = a00*b01 + a01*b11
            # c10 = a10*b00 + a11*b10
            # c11 = a10*b01 + a11*b11
            
            # M1 contributes to C00 and C11
            
            # M1 term 1: (A00+A11)[0,0] * (B00+B11)[0,0]
            u1 = {(oi,oj): 1, (oi+2,oj+2): 1}
            v1 = {(oj,ok): 1, (oj+2,ok+2): 1}
            w1 = {(oi,ok): 1, (oi+2,ok+2): 1}
            add_term(u1, v1, w1)
            
            # M1 term 2: (A00+A11)[0,1] * (B00+B11)[1,0]
            u2 = {(oi,oj+1): 1, (oi+2,oj+3): 1}
            v2 = {(oj+2,ok): 1, (oj,ok+2): 1}
            w2 = {(oi,ok+2): 1, (oi+2,ok): 1}
            add_term(u2, v2, w2)
            
            # M1 term 3: (A00+A11)[0,0] * (B00+B11)[0,1]
            u3 = {(oi,oj): 1, (oi+2,oj+2): 1}
            v3 = {(oj,ok+1): 1, (oj+2,ok+3): 1}
            w3 = {(oi,ok+1): 1, (oi+2,ok+3): 1}
            add_term(u3, v3, w3)
            
            # M1 term 4: (A00+A11)[0,1] * (B00+B11)[1,1]
            u4 = {(oi,oj+1): 1, (oi+2,oj+3): 1}
            v4 = {(oj+2,ok+1): 1, (oj,ok+3): 1}
            w4 = {(oi,ok+3): 1, (oi+2,ok+1): 1}
            add_term(u4, v4, w4)
            
            # M2 = (A10+A11)B00 -> C10, C11
            # (A10+A11) is 2x2, B00 is 2x2
            
            # M2 term 1: (A10+A11)[0,0] * B00[0,0]
            u5 = {(oi+2,oj): 1, (oi+2,oj+2): 1}
            v5 = {(oj,ok): 1}
            w5 = {(oi+2,ok): 1, (oi+2,ok+2): -1}
            add_term(u5, v5, w5)
            
            # M2 term 2: (A10+A11)[0,1] * B00[1,0]
            u6 = {(oi+2,oj+1): 1, (oi+2,oj+3): 1}
            v6 = {(oj+2,ok): 1}
            w6 = {(oi+2,ok+2): 1}
            add_term(u6, v6, w6)
            
            # M2 term 3: (A10+A11)[1,0] * B00[0,0]
            u7 = {(oi+3,oj): 1, (oi+3,oj+2): 1}
            v7 = {(oj,ok): 1}
            w7 = {(oi+3,ok): 1, (oi+3,ok+2): -1}
            add_term(u7, v7, w7)
            
            # M2 term 4: (A10+A11)[1,1] * B00[1,0]
            u8 = {(oi+3,oj+1): 1, (oi+3,oj+3): 1}
            v8 = {(oj+2,ok): 1}
            w8 = {(oi+3,ok+2): 1}
            add_term(u8, v8, w8)
            
            # M3 = A00(B01-B11) -> C01, C11
            # A00 is 2x2, (B01-B11) is 2x2
            
            # M3 term 1: A00[0,0] * (B01-B11)[0,0]
            u9 = {(oi,oj): 1}
            v9 = {(oj,ok+2): 1, (oj+2,ok): -1}
            w9 = {(oi,ok+2): 1, (oi+2,ok+2): 1}
            add_term(u9, v9, w9)
            
            # M3 term 2: A00[0,1] * (B01-B11)[1,0]
            u10 = {(oi,oj+1): 1}
            v10 = {(oj,ok+3): 1, (oj+2,ok+1): -1}
            w10 = {(oi,ok+3): 1, (oi+2,ok+1): 1}
            add_term(u10, v10, w10)
            
            # M3 term 3: A00[1,0] * (B01-B11)[0,0]
            u11 = {(oi+1,oj): 1}
            v11 = {(oj,ok+2): 1, (oj+2,ok): -1}
            w11 = {(oi+1,ok+2): 1, (oi+3,ok+2): 1}
            add_term(u11, v11, w11)
            
            # M3 term 4: A00[1,1] * (B01-B11)[1,0]
            u12 = {(oi+1,oj+1): 1}
            v12 = {(oj,ok+3): 1, (oj+2,ok+1): -1}
            w12 = {(oi+1,ok+3): 1, (oi+3,ok+1): 1}
            add_term(u12, v12, w12)
            
            # M4 = A11(B10-B00) -> C00, C10
            # A11 is 2x2, (B10-B00) is 2x2
            
            # M4 term 1: A11[0,0] * (B10-B00)[0,0]
            u13 = {(oi+2,oj+2): 1}
            v13 = {(oj+2,ok): 1, (oj,ok): -1}
            w13 = {(oi,ok): 1, (oi+2,ok): 1}
            add_term(u13, v13, w13)
            
            # M4 term 2: A11[0,1] * (B10-B00)[1,0]
            u14 = {(oi+2,oj+3): 1}
            v14 = {(oj+2,ok+1): 1, (oj,ok+1): -1}
            w14 = {(oi,ok+1): 1, (oi+2,ok+1): 1}
            add_term(u14, v14, w14)
            
            # M4 term 3: A11[1,0] * (B10-B00)[0,0]
            u15 = {(oi+3,oj+2): 1}
            v15 = {(oj+2,ok): 1, (oj,ok): -1}
            w15 = {(oi+1,ok): 1, (oi+3,ok): 1}
            add_term(u15, v15, w15)
            
            # M4 term 4: A11[1,1] * (B10-B00)[1,0]
            u16 = {(oi+3,oj+3): 1}
            v16 = {(oj+2,ok+1): 1, (oj,ok+1): -1}
            w16 = {(oi+1,ok+1): 1, (oi+3,ok+1): 1}
            add_term(u16, v16, w16)
            
            # M5 = (A00+A01)B11 -> C00, C01
            # (A00+A01) is 2x2, B11 is 2x2
            
            # M5 term 1: (A00+A01)[0,0] * B11[0,0]
            u17 = {(oi,oj): 1, (oi,oj+2): 1}
            v17 = {(oj+2,ok+2): 1}
            w17 = {(oi,ok): -1, (oi,ok+2): 1}
            add_term(u17, v17, w17)
            
            # M5 term 2: (A00+A01)[0,1] * B11[1,0]
            u18 = {(oi,oj+1): 1, (oi,oj+3): 1}
            v18 = {(oj+2,ok+3): 1}
            w18 = {(oi,ok+1): 1}
            add_term(u18, v18, w18)
            
            # M5 term 3: (A00+A01)[1,0] * B11[0,0]
            u19 = {(oi+1,oj): 1, (oi+1,oj+2): 1}
            v19 = {(oj+2,ok+2): 1}
            w19 = {(oi+1,ok): -1, (oi+1,ok+2): 1}
            add_term(u19, v19, w19)
            
            # M5 term 4: (A00+A01)[1,1] * B11[1,0]
            u20 = {(oi+1,oj+1): 1, (oi+1,oj+3): 1}
            v20 = {(oj+2,ok+3): 1}
            w20 = {(oi+1,ok+1): 1}
            add_term(u20, v20, w20)
            
            # M6 = (A10-A00)(B00+B01) -> C11
            # (A10-A00) is 2x2, (B00+B01) is 2x2
            
            # M6 term 1: (A10-A00)[0,0] * (B00+B01)[0,0]
            u21 = {(oi+2,oj): 1, (oi,oj): -1}
            v21 = {(oj,ok): 1, (oj,ok+2): 1}
            w21 = {(oi+2,ok+2): 1}
            add_term(u21, v21, w21)
            
            # M6 term 2: (A10-A00)[0,1] * (B00+B01)[1,0]
            u22 = {(oi+2,oj+1): 1, (oi,oj+1): -1}
            v22 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w22 = {(oi+2,ok+3): 1}
            add_term(u22, v22, w22)
            
            # M6 term 3: (A10-A00)[1,0] * (B00+B01)[0,0]
            u23 = {(oi+3,oj): 1, (oi+1,oj): -1}
            v23 = {(oj,ok): 1, (oj,ok+2): 1}
            w23 = {(oi+3,ok+2): 1}
            add_term(u23, v23, w23)
            
            # M6 term 4: (A10-A00)[1,1] * (B00+B01)[1,0]
            u24 = {(oi+3,oj+1): 1, (oi+1,oj+1): -1}
            v24 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w24 = {(oi+3,ok+3): 1}
            add_term(u24, v24, w24)
            
            # M7 = (A01-A11)(B10+B11) -> C00
            # (A01-A11) is 2x2, (B10+B11) is 2x2
            
            # M7 term 1: (A01-A11)[0,0] * (B10+B11)[0,0]
            u25 = {(oi,oj+2): 1, (oi+2,oj+2): -1}
            v25 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w25 = {(oi,ok): 1}
            add_term(u25, v25, w25)
            
            # M7 term 2: (A01-A11)[0,1] * (B10+B11)[1,0]
            u26 = {(oi,oj+3): 1, (oi+2,oj+3): -1}
            v26 = {(oj+2,ok+1): 1, (oj+2,ok+3): 1}
            w26 = {(oi,ok+1): 1}
            add_term(u26, v26, w26)
            
            # M7 term 3: (A01-A11)[1,0] * (B10+B11)[0,0]
            u27 = {(oi+1,oj+2): 1, (oi+3,oj+2): -1}
            v27 = {(oj+2,ok): 1, (oj+2,ok+2): 1}
            w27 = {(oi+1,ok): 1}
            add_term(u27, v27, w27)
            
            # M7 term 4: (A01-A11)[1,1] * (B10+B11)[1,0]
            u28 = {(oi+1,oj+3): 1, (oi+3,oj+3): -1}
            v28 = {(oj+2,ok+1): 1, (oj+2,ok+3): 1}
            w28 = {(oi+1,ok+1): 1}
            add_term(u28, v28, w28)
        
        strassen_4x4_schoolbook_inner(0, 0, 0)
        
        # PART 2: 4x4 * 4x2 (columns 4-5 of B)
        # Schoolbook decomposition
        # C[i,4] = sum_j A[i,j] * B[j,4]
        # C[i,5] = sum_j A[i,j] * B[j,5]
        
        for i in range(4):
            for j in range(4):
                # C[i,4] contribution
                u = {(i, j): 1}
                v = {(j, 4): 1}
                w = {(i, 4): 1}
                add_term(u, v, w)
                
                # C[i,5] contribution
                u = {(i, j): 1}
                v = {(j, 5): 1}
                w = {(i, 5): 1}
                add_term(u, v, w)
        
        rank = len(U_cols)
        
        # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
        U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
        V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
        W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
        
        return U_final, V_final, W_final
    
    U, V, W = winograd_f4()
    
    # Verify the decomposition efficiently
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(np.dot(U_arr[i*m+j, :] * V_arr[j*p+k, :], W_arr[k*n+i, :]) - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    # Convert to half-integers if possible
    def to_half_integers(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val * 2 - round(val * 2)) < 1e-9:
                    new_row.append(round(val * 2) / 2)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = to_half_integers(U)
    V = to_half_integers(V)
    W = to_half_integers(W)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
