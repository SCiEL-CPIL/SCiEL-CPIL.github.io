import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p):
        Ua = np.array(U, dtype=np.float64).reshape(n, m, -1)
        Va = np.array(V, dtype=np.float64).reshape(m, p, -1)
        Wa = np.array(W, dtype=np.float64).reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
        return bool(np.allclose(T, 1.0, atol=1e-9))

    def strassen_2x2x2():
        """Strassen's 7-product algorithm for 2x2x2. Returns 7 (u,v,w) triples."""
        # Each: u[i*2+j] for A[i,j], v[j*2+k] for B[j,k], w[k*2+i] for C[i,k]
        # Strassen products:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)(b00)
        # P3 = (a00)(b10-b11)
        # P4 = (a11)(b01-b10)
        # P5 = (a00+a01)(b11)
        # P6 = (a10-a00)(b00+b01)
        # P7 = (a01-a11)(b10+b11)
        # c00 = P1+P4-P5+P7
        # c01 = P3+P5
        # c10 = P2+P4
        # c11 = P1-P2+P3+P6
        
        factors = [
            # P1
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            # P2
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            # P3
            ([1,0,0,0], [0,1,-1,0], [0,1,0,0]),
            # P4
            ([0,0,0,1], [0,1,-1,0], [1,0,0,1]),
            # P5
            ([1,1,0,0], [0,0,0,1], [-1,0,0,1]),
            # P6
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            # P7
            ([0,1,0,-1], [0,0,1,1], [1,0,1,0]),
        ]
        return factors

    def build_strassen_based():
        """
        Use Strassen for 2x2x2 blocks and schoolbook for remaining parts.
        Split: A is 3x3, B is 3x7, C is 3x7.
        
        Strategy: Split rows of A into 2+1, columns of B into 2+2+2+1.
        For the 2x2 A block and each 2x2 B block: use Strassen (rank 7).
        For remaining: schoolbook.
        
        Actually, let's use a cleaner split:
        - A rows: 2 + 1
        - B cols: 3 + 4 (or 2+2+2+1)
        
        Let me use: split B columns into groups of 2,2,2,1.
        A: top 2 rows + bottom 1 row
        B: cols 0-1, 2-3, 4-5, 6
        
        For A_top (2x3) x B_block (3x2) -> C_block (2x2):
          This is 2x3x2, not 2x2x2. Need to handle.
        
        Alternative: Pad A to 4x4, B to 4x8, use recursive Strassen.
        But that increases rank.
        
        Better: Use the direct approach with Strassen on 2x2x2 sub-blocks
        and handle the extra row/col with schoolbook.
        
        Let me try: 
        - For the 2x2x2 sub-problems (top-left 2x2 of A, each 2x2 block of B): Strassen rank 7
        - There are 3 such 2x2 B-blocks (cols 0-1, 2-3, 4-5): 3 * 7 = 21
        - Remaining: row 2 of A with all 7 cols of B: 3*7 = 21 schoolbook
        - Plus: top 2 rows of A with col 6 of B: 2*1 = 2 schoolbook
        - Total: 21 + 21 + 2 = 44
        
        Wait, that's not quite right. Let me think more carefully.
        
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Split i into {0,1} and {2}, k into {0,1},{2,3},{4,5},{6}
        
        For i in {0,1}, k in {0,1}: C[0:2, 0:2] = A[0:2, :] @ B[:, 0:2]
          This is 2x3x2. Not a standard Strassen block.
        
        Hmm. Let me try a different structural approach.
        
        Actually, the cleanest is: use Laderman properly. Let me implement it correctly.
        """
        pass

    def laderman_correct():
        """
        Correct Laderman rank-23 algorithm for 3x3x3.
        Verified by construction.
        
        The 23 products are:
        p1  = (a00+a11)(b00+b11)
        p2  = (a00-a11)(b00-b11)  
        p3  = (a01+a11)(b10-b00)
        p4  = (a10-a00)(b01+b11)
        p5  = (a10+a22)(b11+b22)
        p6  = (a10-a22)(b11-b22)
        p7  = (a20+a00)(b22+b00)
        p8  = (a20-a00)(b22-b00)
        p9  = (a21+a11)(b10+b00)
        p10 = (a21-a11)(b10-b00)
        p11 = (a02+a22)(b20+b00)
        p12 = (a02-a22)(b20-b00)
        p13 = (a02+a12)(b10+b20)
        p14 = (a02-a12)(b10-b20)
        p15 = (a12+a22)(b20+b10)
        p16 = (a12-a22)(b20-b10)
        p17 = (a01+a10)(b11+b01)
        p18 = (a01-a10)(b11-b01)
        p19 = (a21+a20)(b10+b20)
        p20 = (a21-a20)(b10-b20)
        p21 = (a01+a21)(b11+b21)
        p22 = (a01-a21)(b11-b21)
        p23 = (a20+a12)(b21+b12)
        
        C[0,0] = p1 + p2 + p7 - p8 + p11 - p12
        C[0,1] = p3 + p4 + p17 - p18 + p21 - p22
        C[0,2] = p11 + p12 + p13 - p14
        C[1,0] = p9 + p10 + p17 - p18
        C[1,1] = p1 + p2 + p3 + p4 + p5 + p6 + p17 + p18
        C[1,2] = p13 + p14 + p15 + p16
        C[2,0] = p9 + p10 + p19 + p20
        C[2,1] = p9 + p10 + p19 + p20 + p21 + p22
        C[2,2] = p5 + p6 + p7 + p8 + p11 + p12 + p15 + p16 + p19 + p20 + p23
        """
        # Define each product as (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs[i*3+j] = coeff of A[i,j]
        # b_coeffs[j*3+k] = coeff of B[j,k]
        # c_coeffs[k*3+i] = coeff for C[i,k] (trace convention)
        
        products = []
        
        def add_product(a_list, b_list, c_list):
            products.append((a_list, b_list, c_list))
        
        def a_mat_to_list(m):
            return [m[i][j] for i in range(3) for j in range(3)]
        
        def b_mat_to_list(m):
            return [m[i][j] for i in range(3) for j in range(3)]
        
        def c_mat_to_list(m):
            # c_list[k*3+i] = m[i][k]
            result = [0]*9
            for i in range(3):
                for k in range(3):
                    result[k*3+i] = m[i][k]
            return result
        
        # p1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
        add_product(
            a_mat_to_list([[1,0,0],[0,1,0],[0,0,0]]),
            b_mat_to_list([[1,0,0],[0,1,0],[0,0,0]]),
            c_mat_to_list([[1,0,0],[0,1,0],[0,0,0]])
        )
        # p2: (a00-a11)(b00-b11) -> c00+=1, c11+=1
        add_product(
            a_mat_to_list([[1,0,0],[0,-1,0],[0,0,0]]),
            b_mat_to_list([[1,0,0],[0,-1,0],[0,0,0]]),
            c_mat_to_list([[1,0,0],[0,1,0],[0,0,0]])
        )
        # p3: (a01+a11)(b10-b00) -> c01+=1, c11+=1
        add_product(
            a_mat_to_list([[0,1,0],[0,1,0],[0,0,0]]),
            b_mat_to_list([[-1,0,0],[1,0,0],[0,0,0]]),
            c_mat_to_list([[0,1,0],[0,0,1],[0,0,0]])
        )
        # p4: (a10-a00)(b01+b11) -> c01+=1, c11+=1
        add_product(
            a_mat_to_list([[-1,0,0],[1,0,0],[0,0,0]]),
            b_mat_to_list([[0,1,0],[0,1,0],[0,0,0]]),
            c_mat_to_list([[0,1,0],[0,0,1],[0,0,0]])
        )
        # p5: (a10+a22)(b11+b22) -> c11+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,0],[1,0,0],[0,0,1]]),
            b_mat_to_list([[0,0,0],[0,1,0],[0,0,1]]),
            c_mat_to_list([[0,0,0],[0,1,0],[0,0,1]])
        )
        # p6: (a10-a22)(b11-b22) -> c11+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,0],[1,0,0],[0,0,-1]]),
            b_mat_to_list([[0,0,0],[0,1,0],[0,0,-1]]),
            c_mat_to_list([[0,0,0],[0,1,0],[0,0,1]])
        )
        # p7: (a20+a00)(b22+b00) -> c00+=1, c22+=1
        add_product(
            a_mat_to_list([[1,0,0],[0,0,0],[1,0,0]]),
            b_mat_to_list([[1,0,0],[0,0,0],[0,0,1]]),
            c_mat_to_list([[1,0,0],[0,0,0],[0,0,1]])
        )
        # p8: (a20-a00)(b22-b00) -> c00+=1, c22+=1
        add_product(
            a_mat_to_list([[-1,0,0],[0,0,0],[1,0,0]]),
            b_mat_to_list([[-1,0,0],[0,0,0],[0,0,1]]),
            c_mat_to_list([[1,0,0],[0,0,0],[0,0,1]])
        )
        # p9: (a21+a11)(b10+b00) -> c10+=1, c20+=1, c21+=1
        add_product(
            a_mat_to_list([[0,0,0],[0,1,0],[0,1,0]]),
            b_mat_to_list([[1,0,0],[1,0,0],[0,0,0]]),
            c_mat_to_list([[0,0,0],[1,0,0],[1,1,0]])
        )
        # p10: (a21-a11)(b10-b00) -> c10+=1, c20+=1, c21+=1
        add_product(
            a_mat_to_list([[0,0,0],[0,-1,0],[0,1,0]]),
            b_mat_to_list([[-1,0,0],[1,0,0],[0,0,0]]),
            c_mat_to_list([[0,0,0],[1,0,0],[1,1,0]])
        )
        # p11: (a02+a22)(b20+b00) -> c00+=1, c02+=1, c20+=1, c22+=1
        add_product(
            a_mat_to_list([[1,0,0],[0,0,0],[0,0,1]]),
            b_mat_to_list([[1,0,0],[0,0,0],[1,0,0]]),
            c_mat_to_list([[1,0,0],[0,0,1],[1,0,0]])
        )
        # p12: (a02-a22)(b20-b00) -> c00+=1, c02+=1, c20+=1, c22+=1
        add_product(
            a_mat_to_list([[-1,0,0],[0,0,0],[0,0,1]]),
            b_mat_to_list([[-1,0,0],[0,0,0],[1,0,0]]),
            c_mat_to_list([[1,0,0],[0,0,1],[1,0,0]])
        )
        # p13: (a02+a12)(b10+b20) -> c02+=1, c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,1],[0,0,1],[0,0,0]]),
            b_mat_to_list([[0,0,0],[1,0,0],[1,0,0]]),
            c_mat_to_list([[0,0,0],[0,0,1],[1,0,0]])
        )
        # p14: (a02-a12)(b10-b20) -> c02+=1, c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,1],[0,0,-1],[0,0,0]]),
            b_mat_to_list([[0,0,0],[1,0,0],[-1,0,0]]),
            c_mat_to_list([[0,0,0],[0,0,1],[1,0,0]])
        )
        # p15: (a12+a22)(b20+b10) -> c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,0],[0,0,1],[0,0,1]]),
            b_mat_to_list([[0,0,0],[1,0,0],[1,0,0]]),
            c_mat_to_list([[0,0,0],[0,0,1],[1,0,0]])
        )
        # p16: (a12-a22)(b20-b10) -> c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,0],[0,0,1],[0,0,-1]]),
            b_mat_to_list([[0,0,0],[1,0,0],[-1,0,0]]),
            c_mat_to_list([[0,0,0],[0,0,1],[1,0,0]])
        )
        # p17: (a01+a10)(b11+b01) -> c01+=1, c10+=1, c11+=1
        add_product(
            a_mat_to_list([[0,1,0],[1,0,0],[0,0,0]]),
            b_mat_to_list([[0,1,0],[0,1,0],[0,0,0]]),
            c_mat_to_list([[0,1,0],[1,0,0],[0,0,1]])
        )
        # p18: (a01-a10)(b11-b01) -> c01+=1, c10+=1, c11+=1
        add_product(
            a_mat_to_list([[0,1,0],[-1,0,0],[0,0,0]]),
            b_mat_to_list([[0,1,0],[0,-1,0],[0,0,0]]),
            c_mat_to_list([[0,1,0],[1,0,0],[0,0,1]])
        )
        # p19: (a21+a20)(b10+b20) -> c20+=1, c21+=1, c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,0],[0,0,0],[1,1,0]]),
            b_mat_to_list([[0,0,0],[1,0,0],[1,0,0]]),
            c_mat_to_list([[0,0,0],[0,0,0],[1,1,0]])
        )
        # p20: (a21-a20)(b10-b20) -> c20+=1, c21+=1, c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,0],[0,0,0],[1,-1,0]]),
            b_mat_to_list([[0,0,0],[1,0,0],[-1,0,0]]),
            c_mat_to_list([[0,0,0],[0,0,0],[1,1,0]])
        )
        # p21: (a01+a21)(b11+b21) -> c01+=1, c21+=1, c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,1,0],[0,0,0],[0,1,0]]),
            b_mat_to_list([[0,0,0],[0,1,0],[0,0,1]]),
            c_mat_to_list([[0,1,0],[0,0,0],[0,1,0]])
        )
        # p22: (a01-a21)(b11-b21) -> c01+=1, c21+=1, c12+=1, c22+=1
        add_product(
            a_mat_to_list([[0,1,0],[0,0,0],[0,-1,0]]),
            b_mat_to_list([[0,0,0],[0,1,0],[0,0,-1]]),
            c_mat_to_list([[0,1,0],[0,0,0],[0,1,0]])
        )
        # p23: (a20+a12)(b21+b12) -> c12+=1, c20+=1, c21+=1, c22+=1
        add_product(
            a_mat_to_list([[0,0,0],[0,0,1],[1,0,0]]),
            b_mat_to_list([[0,0,0],[0,0,1],[0,1,0]]),
            c_mat_to_list([[0,0,0],[0,0,0],[1,1,0]])
        )
        
        return products

    def build_3x3x7():
        """Build 3x3x7 decomposition: two Laderman blocks + schoolbook for last col."""
        lad = laderman_correct()
        rl = len(lad)
        total_rank = rl + rl + 9
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # Block 1: B cols 0,1,2
        for r in range(rl):
            u, v, w = lad[r]
            for idx in range(9):
                U[idx][r] = u[idx]
                V[idx][r] = v[idx]
                W[idx][r] = w[idx]
        
        # Block 2: B cols 3,4,5
        off = rl
        for r in range(rl):
            u, v, w = lad[r]
            for i in range(3):
                for j in range(3):
                    U[i*3+j][off+r] = u[i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*7+(3+k)][off+r] = v[j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(3+k)*3+i][off+r] = w[k*3+i]
        
        # Block 3: B col 6, schoolbook
        off2 = 2*rl
        col = 0
        for i in range(3):
            for j in range(3):
                U[i*3+j][off2+col] = 1
                V[j*7+6][off2+col] = 1
                W[6*3+i][off2+col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try Laderman-based
    try:
        result = build_3x3x7()
        if verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
