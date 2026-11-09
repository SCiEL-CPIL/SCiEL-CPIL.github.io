import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Strategy: Partition 4x5x5 into blocks:
    # i-partition: [0,1] and [2,3] (size 2 each)
    # j-partition: [0,1] and [2,3,4] (sizes 2 and 3)
    # k-partition: [0,1] and [2,3,4] (sizes 2 and 3)
    #
    # This gives 8 sub-blocks:
    # (2x2x2) at (i1,j1,k1): use Strassen (rank 7) -> 4 such blocks
    # (2x2x3) at (i1,j1,k2): need decomposition
    # (2x3x2) at (i1,j2,k1): need decomposition
    # (2x3x3) at (i1,j2,k2): need decomposition
    # Same for i2
    #
    # For 2x2x2: Strassen rank 7 (saves 1 vs standard 8)
    # For 2x2x3: standard is 12, can we do better?
    #   Think of it as 2x2 block in (i,j) with k having 3 values.
    #   For each k, we have a 2x2 multiplication. Use Strassen per k-pair?
    #   Actually, 2x2x3 = multiply 2x2 by 2x3.
    #   Standard: 2*2*3 = 12 terms.
    #   We can group k into {k1,k2} and {k3}.
    #   For (i,j) in 2x2, k in {k1,k2}: use Strassen (7 terms)
    #   For (i,j) in 2x2, k = k3: 4 terms (standard)
    #   Total: 7 + 4 = 11 (saves 1)
    #
    # For 2x3x2: multiply 2x3 by 3x2.
    #   Standard: 2*3*2 = 12 terms.
    #   Group j into {j1,j2} and {j3}.
    #   For i in 2, j in {j1,j2}, k in 2: use Strassen (7 terms)
    #   For i in 2, j = j3, k in 2: 4 terms
    #   Total: 7 + 4 = 11 (saves 1)
    #
    # For 2x3x3: multiply 2x3 by 3x3.
    #   Standard: 2*3*3 = 18 terms.
    #   Group j into {j1,j2} and {j3}, k into {k1,k2} and {k3}.
    #   (i in 2, j in {j1,j2}, k in {k1,k2}): Strassen 7
    #   (i in 2, j in {j1,j2}, k = k3): 4 terms
    #   (i in 2, j = j3, k in {k1,k2}): 4 terms
    #   (i in 2, j = j3, k = k3): 2 terms
    #   Total: 7 + 4 + 4 + 2 = 17 (saves 1)
    #
    # Total rank:
    # 4 blocks of 2x2x2: 4 * 7 = 28
    # 4 blocks of 2x2x3: 4 * 11 = 44
    # 4 blocks of 2x3x2: 4 * 11 = 44
    # 4 blocks of 2x3x3: 4 * 17 = 68
    # Wait, that's 8 blocks total per (i-group, j-group, k-group) combination.
    # Actually: 2 i-groups * 2 j-groups * 2 k-groups = 8 blocks.
    # But the block types depend on which groups:
    # i-group size 2, j-group size 2 or 3, k-group size 2 or 3.
    #
    # Let me recount:
    # i-partition: I1=[0,1] (size 2), I2=[2,3] (size 2)
    # j-partition: J1=[0,1] (size 2), J2=[2,3,4] (size 3)
    # k-partition: K1=[0,1] (size 2), K2=[2,3,4] (size 3)
    #
    # 8 blocks:
    # (I1,J1,K1): 2x2x2 -> Strassen 7
    # (I1,J1,K2): 2x2x3 -> 11
    # (I1,J2,K1): 2x3x2 -> 11
    # (I1,J2,K2): 2x3x3 -> 17
    # (I2,J1,K1): 2x2x2 -> Strassen 7
    # (I2,J1,K2): 2x2x3 -> 11
    # (I2,J2,K1): 2x3x2 -> 11
    # (I2,J2,K2): 2x3x3 -> 17
    #
    # Total: 2*(7+11+11+17) = 2*46 = 92
    #
    # This gives the same rank 92 as the previous approach, but with a cleaner
    # structure that might allow further optimization.
    #
    # Actually, let me try a different partition to see if we can do better.
    # What if we use a 1x4x5 split? No, that doesn't help.
    #
    # Let me try: can we find a better decomposition for 2x3x3?
    # Standard is 18. My decomposition gives 17.
    # Is 17 optimal for 2x3x3? Let me think...
    # 2x3x3 tensor: T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # This is the tensor for multiplying a 2x3 matrix by a 3x3 matrix.
    # The rank of this tensor... 
    #
    # Actually, let me just implement the 92-rank decomposition cleanly and
    # verify it works. Then I'll try to see if I can shave off more terms.
    
    def build_decomposition():
        """Build a rank-92 decomposition using block-wise Strassen optimization."""
        # Define partitions
        I1, I2 = [0, 1], [2, 3]
        J1, J2 = [0, 1], [2, 3, 4]
        K1, K2 = [0, 1], [2, 3, 4]
        
        # We'll build columns incrementally
        U_cols = []  # list of 20-element vectors
        V_cols = []  # list of 25-element vectors
        W_cols = []  # list of 20-element vectors
        
        def add_term(U_entries, V_entries, W_entries):
            """Add a term with given non-zero entries.
            U_entries: dict {row: value}
            V_entries: dict {row: value}
            W_entries: dict {row: value}
            """
            U_vec = [0] * nm
            V_vec = [0] * mp
            W_vec = [0] * pn
            for row, val in U_entries.items():
                U_vec[row] = val
            for row, val in V_entries.items():
                V_vec[row] = val
            for row, val in W_entries.items():
                W_vec[row] = val
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
        
        def strassen_2x2x2(I, J, K):
            """Add Strassen terms for a 2x2x2 block."""
            i1, i2 = I
            j1, j2 = J
            k1, k2 = K
            
            # M1 = (a11 + a22)(b11 + b22)
            add_term(
                {i1*m+j1: 1, i2*m+j2: 1},
                {j1*p+k1: 1, j2*p+k2: 1},
                {k1*n+i1: 1, k2*n+i2: 1}
            )
            # M2 = (a21 + a22)b11
            add_term(
                {i2*m+j1: 1, i2*m+j2: 1},
                {j1*p+k1: 1},
                {k1*n+i2: 1, k2*n+i2: -1}
            )
            # M3 = a11(b12 - b22)
            add_term(
                {i1*m+j1: 1},
                {j1*p+k2: 1, j2*p+k2: -1},
                {k2*n+i1: 1, k2*n+i2: 1}
            )
            # M4 = a22(b21 - b11)
            add_term(
                {i2*m+j2: 1},
                {j2*p+k1: 1, j1*p+k1: -1},
                {k1*n+i1: 1, k1*n+i2: 1}
            )
            # M5 = (a11 + a12)b22
            add_term(
                {i1*m+j1: 1, i1*m+j2: 1},
                {j2*p+k2: 1},
                {k1*n+i1: -1, k2*n+i1: 1}
            )
            # M6 = (a21 - a11)(b11 + b12)
            add_term(
                {i2*m+j1: 1, i1*m+j1: -1},
                {j1*p+k1: 1, j1*p+k2: 1},
                {k2*n+i2: 1}
            )
            # M7 = (a12 - a22)(b21 + b22)
            add_term(
                {i1*m+j2: 1, i2*m+j2: -1},
                {j2*p+k1: 1, j2*p+k2: 1},
                {k1*n+i1: 1}
            )
        
        def standard_2x2x1(I, J, k):
            """Add standard terms for a 2x2x1 block (i in I, j in J, fixed k)."""
            for i in I:
                for j in J:
                    add_term(
                        {i*m+j: 1},
                        {j*p+k: 1},
                        {k*n+i: 1}
                    )
        
        def standard_2x1x1(I, j, k):
            """Add standard terms for a 2x1x1 block."""
            for i in I:
                add_term(
                    {i*m+j: 1},
                    {j*p+k: 1},
                    {k*n+i: 1}
                )
        
        def standard_1x1x1(i, j, k):
            """Add a single standard term."""
            add_term(
                {i*m+j: 1},
                {j*p+k: 1},
                {k*n+i: 1}
            )
        
        def decompose_2x2x3(I, J, K):
            """Decompose 2x2x3 block: Strassen on (I,J,K[:2]) + standard on (I,J,K[2])."""
            strassen_2x2x2(I, J, K[:2])
            for k in K[2:]:
                standard_2x2x1(I, J, k)
        
        def decompose_2x3x2(I, J, K):
            """Decompose 2x3x2 block: Strassen on (I,J[:2],K) + standard on (I,J[2:],K)."""
            strassen_2x2x2(I, J[:2], K)
            for j in J[2:]:
                for k in K:
                    standard_1x1x1(I[0], j, k)
                    standard_1x1x1(I[1], j, k)
        
        def decompose_2x3x3(I, J, K):
            """Decompose 2x3x3 block: 
            Strassen on (I,J[:2],K[:2]) + standard on remaining.
            """
            strassen_2x2x2(I, J[:2], K[:2])
            # Remaining: (I, J[:2], K[2:]) and (I, J[2:], K)
            for j in J[:2]:
                for k in K[2:]:
                    for i in I:
                        standard_1x1x1(i, j, k)
            for j in J[2:]:
                for k in K:
                    for i in I:
                        standard_1x1x1(i, j, k)
        
        # Now build all 8 blocks
        for I in [I1, I2]:
            for J in [J1, J2]:
                for K in [K1, K2]:
                    if len(I) == 2 and len(J) == 2 and len(K) == 2:
                        strassen_2x2x2(I, J, K)
                    elif len(I) == 2 and len(J) == 2 and len(K) == 3:
                        decompose_2x2x3(I, J, K)
                    elif len(I) == 2 and len(J) == 3 and len(K) == 2:
                        decompose_2x3x2(I, J, K)
                    elif len(I) == 2 and len(J) == 3 and len(K) == 3:
                        decompose_2x3x3(I, J, K)
                    else:
                        # Fallback: standard terms
                        for i in I:
                            for j in J:
                                for k in K:
                                    standard_1x1x1(i, j, k)
        
        rank = len(U_cols)
        U = [U_cols[r] for r in range(rank)]  # transpose: U[i][r]
        V = [V_cols[r] for r in range(rank)]
        W = [W_cols[r] for r in range(rank)]
        
        # Transpose to get U[i][r] format
        U_final = [[U[r][i] for r in range(rank)] for i in range(nm)]
        V_final = [[V[r][i] for r in range(rank)] for i in range(mp)]
        W_final = [[W[r][i] for r in range(rank)] for i in range(pn)]
        
        return U_final, V_final, W_final, rank
    
    U, V, W, rank = build_decomposition()
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is correct."""
        nm, mp, pn = n*m, m*p, p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to integer/half-integer format
    U_final = []
    for i in range(nm):
        row = []
        for r in range(rank):
            val = U[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        U_final.append(row)
    
    V_final = []
    for i in range(mp):
        row = []
        for r in range(rank):
            val = V[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        V_final.append(row)
    
    W_final = []
    for i in range(pn):
        row = []
        for r in range(rank):
            val = W[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        W_final.append(row)
    
    return {"U": U_final, "V": V_final, "W": W_final}
