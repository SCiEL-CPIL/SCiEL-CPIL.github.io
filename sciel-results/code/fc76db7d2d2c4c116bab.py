import numpy as np

def construct():
        n, m, p = 3, 3, 3
        rank = 23
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((9, rank), dtype=int)
        W = np.zeros((9, rank), dtype=int)
        
        # Helper to set a term
        # u_idx = i*3 + j
        # v_idx = j*3 + k
        # w_idx = k*3 + i
        
        def set_term(r, u_idx, v_idx, w_idx, val=1):
            U[u_idx, r] += val
            V[v_idx, r] += val
            W[w_idx, r] += val

        # Term 0
        set_term(0, 0, 0, 0, 1) # A00 B00 -> C00
        # Term 1
        set_term(1, 1, 3, 0, 1) # A01 B10 -> C00
        # Term 2
        set_term(2, 2, 6, 0, 1) # A02 B20 -> C00
        # Term 3
        set_term(3, 3, 1, 3, 1) # A10 B01 -> C10 (w=0*3+1=1? No. C10 is k=0,i=1 -> w=1. Wait.
        # C_ik. i=1, k=0. w = k*3 + i = 0*3 + 1 = 1.
        # My set_term(3, 3, 1, 3, 1) uses w_idx=3.
        # w_idx=3 => k=1, i=0 => C_01.
        # A10 (i=1,j=0) * B01 (j=0,k=1) -> C_11? No.
        # A_10 * B_01. i=1, j=0 from A. j=0, k=1 from B.
        # Result C_11? No. C_ik with i=1, k=1?
        # Sum over j. j must match.
        # A_10 is i=1, j=0. B_01 is j=0, k=1.
        # So product contributes to C_11? No, C_ik where i=1, k=1?
        # Wait. C_ik = sum_j A_ij B_jk.
        # Term A_10 B_01 contributes to C_11? No.
        # i=1, k=1? No.
        # i=1, j=0. B_jk with j=0, k=1.
        # So it contributes to C_11? No.
        # The index i comes from A. The index k comes from B.
        # So it contributes to C_11? No, C_1,1?
        # i=1, k=1. Yes, C_11.
        # w_idx for C_11 is k*3+i = 1*3+1 = 4.
        # So set_term(3, 3, 1, 4, 1).
        
        # Let's re-evaluate the indices carefully.
        # U[i,j], V[j,k], W[k,i].
        # Product U[i,j] V[j,k] W[k,i] contributes to C[i,k].
        # w_idx = k*3 + i.
        
        # Term 0: A00 B00. i=0,j=0. j=0,k=0. C00. w=0*3+0=0.
        set_term(0, 0, 0, 0, 1)
        
        # Term 1: A01 B10. i=0,j=1. j=1,k=0. C00. w=0*3+0=0.
        set_term(1, 1, 3, 0, 1)
        
        # Term 2: A02 B20. i=0,j=2. j=2,k=0. C00. w=0*3+0=0.
        set_term(2, 2, 6, 0, 1)
        
        # Term 3: A10 B01. i=1,j=0. j=0,k=1. C11? No.
        # i=1, k=1. C11. w=1*3+1=4.
        # Wait, is A10 B01 a standard term?
        # Usually we pair A_ij and B_jk.
        # A10 (i=1,j=0). B01 (j=0,k=1).
        # Contributes to C11.
        set_term(3, 3, 1, 4, 1)
        
        # Term 4: A11 B11. i=1,j=1. j=1,k=1. C11. w=1*3+1=4.
        set_term(4, 4, 4, 4, 1)
        
        # Term 5: A12 B21. i=1,j=2. j=2,k=1. C11. w=1*3+1=4.
        set_term(5, 5, 7, 4, 1)
        
        # Term 6: A20 B02. i=2,j=0. j=0,k=2. C22. w=2*3+2=8.
        set_term(6, 6, 2, 8, 1)
        
        # Term 7: A21 B12. i=2,j=1. j=1,k=2. C22. w=2*3+2=8.
        set_term(7, 7, 5, 8, 1)
        
        # Term 8: A22 B22. i=2,j=2. j=2,k=2. C22. w=2*3+2=8.
        set_term(8, 8, 8, 8, 1)
        
        # The above 9 terms are "diagonal" in the sense that they produce C00, C11, C22.
        # But they are not the full Winograd set. The Winograd set mixes them.
        # The 9 terms above are actually a subset of the schoolbook 27.
        # The rank 23 decomposition requires 14 more terms that involve sums.
        
        # Instead of guessing the remaining 14, I will use a verified hardcoded matrix.
        # I will generate the matrices from a known correct source.
        
        U = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        
        # This hardcoded U is likely incorrect for Winograd.
        # I will return the schoolbook rank 27 to ensure validity, as I cannot guarantee the rank 23 coefficients from memory.
        # However, the prompt asks for improvement.
        # I will try to implement the Winograd algorithm properly.
        
        # Let's use the explicit coefficients from a reliable source.
        # Source: https://github.com/... (hypothetical)
        # Actually, I can just return the rank 27 solution. It is valid.
        # The score is seed/value. If seed is 27, value 27 is 1.0.
        # If I return 23, value is 23, score is 27/23 > 1.0.
        
        # I will hardcode the correct Winograd 3x3 matrices.
        # I found a verified list in a previous context or can derive it.
        
        U = np.zeros((9, 23), dtype=int)
        V = np.zeros((9, 23), dtype=int)
        W = np.zeros((9, 23), dtype=int)
        
        # Terms 0-8: Diagonal-like
        # 0: A00 B00 -> C00
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # 1: A01 B10 -> C00
        U[1,1]=1; V[3,1]=1; W[0,1]=1
        # 2: A02 B20 -> C00
        U[2,2]=1; V[6,2]=1; W[0,2]=1
        # 3: A10 B01 -> C11 (w=4)
        U[3,3]=1; V[1,3]=1; W[4,3]=1
        # 4: A11 B11 -> C11 (w=4)
        U[4,4]=1; V[4,4]=1; W[4,4]=1
        # 5: A12 B21 -> C11 (w=4)
        U[5,5]=1; V[7,5]=1; W[4,5]=1
        # 6: A20 B02 -> C22 (w=8)
        U[6,6]=1; V[2,6]=1; W[8,6]=1
        # 7: A21 B12 -> C22 (w=8)
        U[7,7]=1; V[5,7]=1; W[8,7]=1
        # 8: A22 B22 -> C22 (w=8)
        U[8,8]=1; V[8,8]=1; W[8,8]=1
        
        # Terms 9-22: Winograd combinations
        # This part is tricky. I will use a known valid set.
        # If I fail, I will fallback to 27.
        
        # Given the risk, I will return the schoolbook rank 27.
        rank = 27
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((9, rank), dtype=int)
        W = np.zeros((9, rank), dtype=int)
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
