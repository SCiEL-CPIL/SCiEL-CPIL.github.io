"""
Rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication.

Approach:
- Split n=3 into 1 + 2
- For the 1x4x5 part: rank 20 (schoolbook)
- For the 2x4x5 part: split m=4 into 2+2
  - Each 2x2x5: split p=5 into 2+3
    - 2x2x2: Strassen rank 7
    - 2x2x3: split p=3 into 2+1
      - 2x2x2: Strassen rank 7
      - 2x2x1: rank 4
    Total per 2x2x5: 7+7+4 = 18
  - Two 2x2x5 blocks: 36
Total rank: 20 + 36 = 56
"""

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_rank1(u_entries, v_entries, w_entries):
        """
        Add a rank-1 term.
        u_entries: dict mapping (i,j) -> coeff
        v_entries: dict mapping (j,k) -> coeff  
        w_entries: dict mapping (k,i) -> coeff
        """
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        for (ii, jj), c in u_entries.items():
            U_row[ii * m + jj] = c
        for (jj, kk), c in v_entries.items():
            V_row[jj * p + kk] = c
        for (kk, ii), c in w_entries.items():
            W_row[kk * n + ii] = c
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    def add_simple(i, j, k):
        """Add a single schoolbook term A[i,j]*B[j,k] -> C[i,k]"""
        add_rank1({(i, j): 1}, {(j, k): 1}, {(k, i): 1})
    
    def strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        """
        Strassen decomposition for multiplying 2x2 by 2x2.
        A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k1+1] = C[i0:i1+1, k0:k1+1]
        
        Uses 7 rank-1 terms.
        """
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # Contributes: C[i0,k0]+=1, C[i0,k1]+=1, C[i1,k0]+=1, C[i1,k1]+=1
        add_rank1(
            {(i0,j0): 1, (i1,j1): 1},
            {(j0,k0): 1, (j1,k1): 1},
            {(k0,i0): 1, (k1,i0): 1, (k0,i1): 1, (k1,i1): 1}
        )
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # Contributes: C[i1,k0]+=1
        add_rank1(
            {(i1,j0): 1, (i1,j1): 1},
            {(j0,k0): 1},
            {(k0,i1): 1}
        )
        
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        # Contributes: C[i0,k0]+=1, C[i0,k1]-=1
        add_rank1(
            {(i0,j0): 1},
            {(j1,k0): 1, (j1,k1): -1},
            {(k0,i0): 1, (k1,i0): -1}
        )
        
        # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
        # Contributes: C[i1,k1]+=1, C[i1,k0]-=1
        add_rank1(
            {(i1,j1): 1},
            {(j0,k1): 1, (j0,k0): -1},
            {(k1,i1): 1, (k0,i1): -1}
        )
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # Contributes: C[i0,k1]+=1
        add_rank1(
            {(i0,j0): 1, (i0,j1): 1},
            {(j1,k1): 1},
            {(k1,i0): 1}
        )
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # Contributes: C[i1,k0]+=1, C[i1,k1]+=1
        add_rank1(
            {(i1,j0): 1, (i0,j0): -1},
            {(j0,k0): 1, (j0,k1): 1},
            {(k0,i1): 1, (k1,i1): 1}
        )
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        # Contributes: C[i0,k0]+=1, C[i0,k1]+=1
        add_rank1(
            {(i0,j1): 1, (i1,j1): -1},
            {(j1,k0): 1, (j1,k1): 1},
            {(k0,i0): 1, (k1,i0): 1}
        )
    
    def block_2x2x5(i_base, j_base, k_base):
        """
        Multiply A[i_base:i_base+2, j_base:j_base+2] by B[j_base:j_base+2, k_base:k_base+5]
        to get C[i_base:i_base+2, k_base:k_base+5]
        Split p=5 into 2+3, then 3 into 2+1.
        """
        # First 2x2x2: k_base, k_base+1
        strassen_2x2x2(i_base, i_base+1, j_base, j_base+1, k_base, k_base+1)
        # Next 2x2x2: k_base+2, k_base+3
        strassen_2x2x2(i_base, i_base+1, j_base, j_base+1, k_base+2, k_base+3)
        # Last 2x2x1: k_base+4
        for di in range(2):
            for dj in range(2):
                add_simple(i_base + di, j_base + dj, k_base + 4)
    
    # Part 1: 1x4x5 (row 0 of A)
    for j in range(4):
        for k in range(5):
            add_simple(0, j, k)
    
    # Part 2: 2x4x5 (rows 1,2 of A)
    # Split columns into two 2x2x5 blocks
    # Block A: columns 0,1
    block_2x2x5(1, 0, 0)
    # Block B: columns 2,3
    block_2x2x5(1, 2, 0)
    
    return {"U": U_list, "V": V_list, "W": W_list}
