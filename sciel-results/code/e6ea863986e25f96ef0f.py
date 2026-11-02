import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def decompose(n, m, p):
        """
        Returns a list of (U_dict, V_dict, W_dict)
        U_dict: {i*m+j: coeff}
        V_dict: {j*p+k: coeff}
        W_dict: {k*n+i: coeff}
        """
        cols = []

        def add_term(u_dict, v_dict, w_dict):
            """
            Adds a single rank-1 term.
            """
            cols.append((
                {i*m+j: val for (i,j), val in u_dict.items()},
                {j*p+k: val for (j,k), val in v_dict.items()},
                {k*n+i: val for (k,i), val in w_dict.items()}
            ))

        def add_strassen_2x2x2(i0, j0, k0):
            """
            Adds 7 Strassen terms for a 2x2x2 block at offsets (i0, j0, k0).
            """
            # M1 = (A11+A22)(B11+B22)
            u1 = {(i0, j0): 1, (i0+1, j0+1): 1}
            v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
            w1 = {(k0, i0): 1, (k0+1, i0+1): 1}
            add_term(u1, v1, w1)

            # M2 = (A21+A22)B11
            u2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
            v2 = {(j0, k0): 1}
            w2 = {(k0, i0+1): 1, (k0+1, i0+1): -1}
            add_term(u2, v2, w2)

            # M3 = A11(B12-B22)
            u3 = {(i0, j0): 1}
            v3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
            w3 = {(k0+1, i0): 1, (k0+1, i0+1): 1}
            add_term(u3, v3, w3)

            # M4 = A22(B21-B11)
            u4 = {(i0+1, j0+1): 1}
            v4 = {(j0+1, k0): 1, (j0, k0): -1}
            w4 = {(k0, i0): 1, (k0, i0+1): 1}
            add_term(u4, v4, w4)

            # M5 = (A11+A12)B22
            u5 = {(i0, j0): 1, (i0, j0+1): 1}
            v5 = {(j0+1, k0+1): 1}
            w5 = {(k0, i0): -1, (k0+1, i0): 1}
            add_term(u5, v5, w5)

            # M6 = (A21-A11)(B11+B12)
            u6 = {(i0+1, j0): 1, (i0, j0): -1}
            v6 = {(j0, k0): 1, (j0, k0+1): 1}
            w6 = {(k0+1, i0+1): 1}
            add_term(u6, v6, w6)

            # M7 = (A12-A22)(B21+B22)
            u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
            v7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
            w7 = {(k0, i0): 1}
            add_term(u7, v7, w7)

        def add_strassen_3x2x2(i0, j0, k0):
            """
            Adds 18 terms for a 3x2x2 block.
            Decomposes into one 2x2x2 Strassen block and 3x2x1 schoolbook blocks.
            """
            # Apply 2x2x2 Strassen on rows 0,1 and cols 0,1
            add_strassen_2x2x2(i0, j0, k0)
            
            # Remainder: 1x2x2 block for row 2
            # Rows: i0+2, Cols: j0, j0+1, K: k0, k0+1
            # This is 4 schoolbook terms
            for j in range(2):
                for k in range(2):
                    add_term(
                        {(i0+2, j0+j): 1},
                        {(j0+j, k0+k): 1},
                        {(k0+k, i0+2): 1}
                    )

        def recurse(n_curr, m_curr, p_curr, i0, j0, k0):
            # Base cases
            if n_curr == 1 and m_curr == 1 and p_curr == 1:
                add_term({(0,0): 1.0}, {(0,0): 1.0}, {(0,0): 1.0})
                return

            # Check if we can apply Strassen 3x2x2 (n=3, m=2, p=2)
            if n_curr >= 3 and m_curr >= 2 and p_curr >= 2:
                # Apply 3x2x2 on first 3 rows, first 2 cols, first 2 ks
                add_strassen_3x2x2(i0, j0, k0)
                
                # Recurse on remainders
                if n_curr > 3:
                    recurse(n_curr - 3, m_curr, p_curr, i0 + 3, j0, k0)
                if m_curr > 2:
                    recurse(n_curr, m_curr - 2, p_curr, i0, j0 + 2, k0)
                if p_curr > 2:
                    recurse(n_curr, m_curr, p_curr - 2, i0, j0, k0 + 2)
                return

            # Check if we can apply Strassen 2x2x2
            if n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
                add_strassen_2x2x2(i0, j0, k0)
                
                # Recurse on remainders
                if n_curr > 2:
                    recurse(n_curr - 2, m_curr, p_curr, i0 + 2, j0, k0)
                if m_curr > 2:
                    recurse(n_curr, m_curr - 2, p_curr, i0, j0 + 2, k0)
                if p_curr > 2:
                    recurse(n_curr, m_curr, p_curr - 2, i0, j0, k0 + 2)
                return

            # Fallback: split into smaller blocks
            if n_curr > 1:
                recurse(1, m_curr, p_curr, i0, j0, k0)
                recurse(n_curr - 1, m_curr, p_curr, i0 + 1, j0, k0)
                return
            elif m_curr > 1:
                recurse(n_curr, 1, p_curr, i0, j0, k0)
                recurse(n_curr, m_curr - 1, p_curr, i0, j0 + 1, k0)
                return
            elif p_curr > 1:
                recurse(n_curr, m_curr, 1, i0, j0, k0)
                recurse(n_curr, m_curr, p_curr - 1, i0, j0, k0 + 1)
                return
            else:
                # 1x1x1
                add_term({(0,0): 1.0}, {(0,0): 1.0}, {(0,0): 1.0})
                return

        recurse(n, m, p, 0, 0, 0)
        return cols

    # Try multiple decomposition strategies and pick the best one
    # Strategy 1: The recursive one above
    cols1 = decompose(n, m, p)
    rank1 = len(cols1)
    
    # Strategy 2: Manual 3x4x6 decomposition
    # Split 3x4x6 into three 3x2x2 blocks? No, 4 is not divisible by 2 evenly in a way that helps if we do 2+2.
    # 3x4x6 = 3x(2+2)x6 = 3x2x6 + 3x2x6
    # 3x2x6 = 3x2x(2+2+2) = 3x2x2 + 3x2x2 + 3x2x2
    # So we have 3 blocks of 3x2x2.
    # Rank of 3x2x2 is 18 (as derived above: 7 Strassen + 4 schoolbook = 11? Wait.
    # In add_strassen_3x2x2 I used: 7 (Strassen 2x2x2) + 4 (schoolbook 1x2x2) = 11.
    # Is 11 optimal for 3x2x2?
    # R(3x2x2) >= R(2x2x2) = 7.
    # Let's check if we can do better.
    # If we use schoolbook for 3x2x2, it's 3*2*2 = 12.
    # My method gives 11.
    # Can we do 10?
    # Maybe. But 11 is better than 12.
    # So 3 blocks of 3x2x2 would give 3 * 11 = 33.
    # The recursive method above gave rank1.
    
    # Let's manually construct the 3x4x6 using 3x2x2 blocks to see if it's better.
    def decompose_manual_3x4x6(n, m, p):
        cols = []
        # 3x4x6
        # Split columns into 2 + 2
        # Block 1: 3x2x6 at (0, 0, 0)
        # Block 2: 3x2x6 at (0, 2, 0)
        
        def add_3x2x2_block(i0, j0, k0, n_b, m_b, p_b):
            # This function assumes n_b=3, m_b=2, p_b=2
            # It adds 11 terms
            add_term_local = lambda u, v, w: cols.append((
                {i*n+j: val for (i,j), val in u.items()},
                {j*p+k: val for (j,k), val in v.items()},
                {k*n+i: val for (k,i), val in w.items()}
            ))
            
            # 2x2x2 Strassen for first 2 rows
            # M1
            add_term_local(
                {(i0, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, i0): 1, (k0+1, i0+1): 1}
            )
            # M2
            add_term_local(
                {(i0+1, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, i0+1): 1, (k0+1, i0+1): -1}
            )
            # M3
            add_term_local(
                {(i0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, i0): 1, (k0+1, i0+1): 1}
            )
            # M4
            add_term_local(
                {(i0+1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, i0): 1, (k0, i0+1): 1}
            )
            # M5
            add_term_local(
                {(i0, j0): 1, (i0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, i0): -1, (k0+1, i0): 1}
            )
            # M6
            add_term_local(
                {(i0+1, j0): 1, (i0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, i0+1): 1}
            )
            # M7
            add_term_local(
                {(i0, j0+1): 1, (i0+1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, i0): 1}
            )
            
            # 4 schoolbook terms for row 2
            for j in range(2):
                for k in range(2):
                    add_term_local(
                        {(i0+2, j0+j): 1},
                        {(j0+j, k0+k): 1},
                        {(k0+k, i0+2): 1}
                    )

        # 3 blocks of 3x2x2
        # 3x4x6 -> 3x2x6 + 3x2x6
        # 3x2x6 -> 3x2x2 + 3x2x2 + 3x2x2
        
        # Block 1: i=0, j=0, k=0
        add_3x2x2_block(0, 0, 0, 3, 2, 2)
        # Block 2: i=0, j=0, k=2
        add_3x2x2_block(0, 0, 2, 3, 2, 2)
        # Block 3: i=0, j=0, k=4
        add_3x2x2_block(0, 0, 4, 3, 2, 2)
        
        # Block 4: i=0, j=2, k=0
        add_3x2x2_block(0, 2, 0, 3, 2, 2)
        # Block 5: i=0, j=2, k=2
        add_3x2x2_block(0, 2, 2, 3, 2, 2)
        # Block 6: i=0, j=2, k=4
        add_3x2x2_block(0, 2, 4, 3, 2, 2)
        
        return cols

    cols2 = decompose_manual_3x4x6(n, m, p)
    rank2 = len(cols2)

    # Pick the best
    if rank1 <= rank2:
        best_cols = cols1
        best_rank = rank1
    else:
        best_cols = cols2
        best_rank = rank2

    U = np.zeros((n * m, best_rank), dtype=float)
    V = np.zeros((m * p, best_rank), dtype=float)
    W = np.zeros((p * n, best_rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(best_cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
