"""Optimized rank-88 decomposition for 3x4x8 matrix multiplication using Strassen."""
DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # Split m=4 into m1=2, m2=2
    # For each half, split n=3 into n1=2, n2=1
    # For the 2x2x8 part, split p=8 into four 2x2x2 blocks and use Strassen (rank 7)
    # For the 1x2x8 part, split p=8 into four 1x2x2 blocks and use schoolbook (rank 4)
    # Total rank = 2 * (4*7 + 4*4) = 2 * (28 + 16) = 2 * 44 = 88
    
    rank = 88
    U = [[0]*rank for _ in range(n*m)]  # 12 x 88
    V = [[0]*rank for _ in range(m*p)]  # 32 x 88
    W = [[0]*rank for _ in range(p*n)]  # 24 x 88
    
    def strassen_222(offset):
        """
        Add Strassen decomposition for a 2x2x2 block.
        Returns 7 rank offsets used.
        Convention: For A(2x2)*B(2x2)=C(2x2), 
        W[k*2+i, r] where i is row of C (0,1), k is col of C (0,1)
        U[i*2+j, r] where i is row of A, j is col of A
        V[j*2+k, r] where j is row of B, k is col of B
        
        Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B10)
        M5 = (A00+A10)B11
        M6 = (A11-A01)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # Indices for this 2x2x2 block
        # i in {0,1} (row of A/C), j in {0,1} (col of A / row of B), k in {0,1} (col of B/C)
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=(0,0)->idx 0, A11=(1,1)->idx 3
        # V: B00=(0,0)->idx 0, B11=(1,1)->idx 3
        # W: C00=(0,0)->idx 0, C11=(1,1)->idx 3
        r = offset
        U[0][r] = 1; U[3][r] = 1
        V[0][r] = 1; V[3][r] = 1
        W[0][r] = 1; W[3][r] = 1
        
        # M2 = (A10+A11)B00
        # U: A10=(1,0)->idx 2, A11=(1,1)->idx 3
        # V: B00=(0,0)->idx 0
        # W: C10=(0,1)->idx 2, C11=(1,1)->idx 3
        r = offset + 1
        U[2][r] = 1; U[3][r] = 1
        V[0][r] = 1
        W[2][r] = 1; W[3][r] = 1
        
        # M3 = A00(B10-B11)
        # U: A00=(0,0)->idx 0
        # V: B10=(1,0)->idx 2, B11=(1,1)->idx 3 (with -1)
        # W: C01=(1,0)->idx 1, C11=(1,1)->idx 3
        r = offset + 2
        U[0][r] = 1
        V[2][r] = 1; V[3][r] = -1
        W[1][r] = 1; W[3][r] = 1
        
        # M4 = A11(B01-B10)
        # U: A11=(1,1)->idx 3
        # V: B01=(0,1)->idx 1, B10=(1,0)->idx 2 (with -1)
        # W: C00=(0,0)->idx 0, C10=(0,1)->idx 2
        r = offset + 3
        U[3][r] = 1
        V[1][r] = 1; V[2][r] = -1
        W[0][r] = 1; W[2][r] = 1
        
        # M5 = (A00+A10)B11
        # U: A00=(0,0)->idx 0, A10=(1,0)->idx 2
        # V: B11=(1,1)->idx 3
        # W: C00=(0,0)->idx 0 (with -1), C01=(1,0)->idx 1
        r = offset + 4
        U[0][r] = 1; U[2][r] = 1
        V[3][r] = 1
        W[0][r] = -1; W[1][r] = 1
        
        # M6 = (A11-A01)(B00+B10)
        # U: A11=(1,1)->idx 3, A01=(0,1)->idx 1 (with -1)
        # V: B00=(0,0)->idx 0, B10=(1,0)->idx 2
        # W: C11=(1,1)->idx 3
        r = offset + 5
        U[3][r] = 1; U[1][r] = -1
        V[0][r] = 1; V[2][r] = 1
        W[3][r] = 1
        
        # M7 = (A01-A11)(B01+B11)
        # U: A01=(0,1)->idx 1, A11=(1,1)->idx 3 (with -1)
        # V: B01=(0,1)->idx 1, B11=(1,1)->idx 3
        # W: C00=(0,0)->idx 0
        r = offset + 6
        U[1][r] = 1; U[3][r] = -1
        V[1][r] = 1; V[3][r] = 1
        W[0][r] = 1
        
        return 7
    
    def schoolbook_122(offset):
        """
        Schoolbook for 1x2x2.
        A(1x2) * B(2x2) = C(1x2)
        C0 = A0*B0 + A1*B2
        C1 = A0*B1 + A1*B3
        
        U indices: i=0 always, j in {0,1} -> idx j
        V indices: j in {0,1}, k in {0,1} -> idx j*2+k
        W indices: k in {0,1}, i=0 -> idx k
        
        4 terms:
        T1: A0*B0 -> C0
        T2: A0*B1 -> C1
        T3: A1*B2 -> C0
        T4: A1*B3 -> C1
        """
        r = offset
        U[0][r] = 1
        V[0][r] = 1
        W[0][r] = 1
        
        r = offset + 1
        U[0][r] = 1
        V[1][r] = 1
        W[1][r] = 1
        
        r = offset + 2
        U[1][r] = 1
        V[2][r] = 1
        W[0][r] = 1
        
        r = offset + 3
        U[1][r] = 1
        V[3][r] = 1
        W[1][r] = 1
        
        return 4
    
    # Now we map the global indices to local block indices
    # Global U index: i*m + j where i in {0,1,2} (row of A), j in {0,1,2,3} (col of A / row of B)
    # Global V index: j*p + k where j in {0,1,2,3} (row of B), k in {0..7} (col of B / col of C)
    # Global W index: k*n + i where k in {0..7} (col of C), i in {0,1,2} (row of C)
    
    # We split m into m1=2 (j=0,1) and m2=2 (j=2,3)
    # For each m-half, we split n into n1=2 (i=0,1) and n2=1 (i=2)
    # We split p into four blocks of size 2: k_block in {0,1,2,3}, local k in {0,1}, global k = k_block*2 + local_k
    
    offset = 0
    for m_half in range(2):  # m1 or m2
        j_base = m_half * 2  # 0 or 2
        
        # Part 1: n1=2 (i=0,1), p split into 4 blocks of 2x2x2
        for k_block in range(4):
            k_base = k_block * 2
            
            # Local to global mapping for this 2x2x2 block:
            # Local i (0,1) -> global i = i
            # Local j (0,1) -> global j = j_base + local_j
            # Local k (0,1) -> global k = k_base + local_k
            
            # We need to apply Strassen but with adjusted indices
            # Let's manually handle this by creating a helper that uses global indices
            
            # For this block:
            # A rows: i=0,1; A cols: j=j_base, j_base+1
            # B rows: j=j_base, j_base+1; B cols: k=k_base, k_base+1
            # C rows: i=0,1; C cols: k=k_base, k_base+1
            
            # Strassen M1 = (A00+A11)(B00+B11)
            # A00: i=0, j=j_base -> U index 0*m + j_base = j_base
            # A11: i=1, j=j_base+1 -> U index 1*m + j_base+1 = m + j_base + 1
            # B00: j=j_base, k=k_base -> V index j_base*p + k_base
            # B11: j=j_base+1, k=k_base+1 -> V index (j_base+1)*p + k_base+1
            # C00: k=k_base, i=0 -> W index k_base*n + 0
            # C11: k=k_base+1, i=1 -> W index (k_base+1)*n + 1
            
            r = offset
            U[j_base][r] = 1
            U[m + j_base + 1][r] = 1
            V[j_base * p + k_base][r] = 1
            V[(j_base + 1) * p + k_base + 1][r] = 1
            W[k_base * n + 0][r] = 1
            W[(k_base + 1) * n + 1][r] = 1
            
            # M2 = (A10+A11)B00
            # A10: i=1, j=j_base -> U index m + j_base
            # A11: i=1, j=j_base+1 -> U index m + j_base + 1
            # B00: j=j_base, k=k_base -> V index j_base*p + k_base
            # C10: k=k_base, i=1 -> W index k_base*n + 1
            # C11: k=k_base+1, i=1 -> W index (k_base+1)*n + 1
            
            r = offset + 1
            U[m + j_base][r] = 1
            U[m + j_base + 1][r] = 1
            V[j_base * p + k_base][r] = 1
            W[k_base * n + 1][r] = 1
            W[(k_base + 1) * n + 1][r] = 1
            
            # M3 = A00(B10-B11)
            # A00: i=0, j=j_base -> U index j_base
            # B10: j=j_base+1, k=k_base -> V index (j_base+1)*p + k_base
            # B11: j=j_base+1, k=k_base+1 -> V index (j_base+1)*p + k_base+1
            # C01: k=k_base+1, i=0 -> W index (k_base+1)*n + 0
            # C11: k=k_base+1, i=1 -> W index (k_base+1)*n + 1
            
            r = offset + 2
            U[j_base][r] = 1
            V[(j_base + 1) * p + k_base][r] = 1
            V[(j_base + 1) * p + k_base + 1][r] = -1
            W[(k_base + 1) * n + 0][r] = 1
            W[(k_base + 1) * n + 1][r] = 1
            
            # M4 = A11(B01-B10)
            # A11: i=1, j=j_base+1 -> U index m + j_base + 1
            # B01: j=j_base, k=k_base+1 -> V index j_base*p + k_base+1
            # B10: j=j_base+1, k=k_base -> V index (j_base+1)*p + k_base
            # C00: k=k_base, i=0 -> W index k_base*n + 0
            # C10: k=k_base, i=1 -> W index k_base*n + 1
            
            r = offset + 3
            U[m + j_base + 1][r] = 1
            V[j_base * p + k_base + 1][r] = 1
            V[(j_base + 1) * p + k_base][r] = -1
            W[k_base * n + 0][r] = 1
            W[k_base * n + 1][r] = 1
            
            # M5 = (A00+A10)B11
            # A00: i=0, j=j_base -> U index j_base
            # A10: i=1, j=j_base -> U index m + j_base
            # B11: j=j_base+1, k=k_base+1 -> V index (j_base+1)*p + k_base+1
            # C00: k=k_base, i=0 -> W index k_base*n + 0 (with -1)
            # C01: k=k_base+1, i=0 -> W index (k_base+1)*n + 0
            
            r = offset + 4
            U[j_base][r] = 1
            U[m + j_base][r] = 1
            V[(j_base + 1) * p + k_base + 1][r] = 1
            W[k_base * n + 0][r] = -1
            W[(k_base + 1) * n + 0][r] = 1
            
            # M6 = (A11-A01)(B00+B10)
            # A11: i=1, j=j_base+1 -> U index m + j_base + 1
            # A01: i=0, j=j_base+1 -> U index j_base + 1
            # B00: j=j_base, k=k_base -> V index j_base*p + k_base
            # B10: j=j_base+1, k=k_base -> V index (j_base+1)*p + k_base
            # C11: k=k_base+1, i=1 -> W index (k_base+1)*n + 1
            
            r = offset + 5
            U[m + j_base + 1][r] = 1
            U[j_base + 1][r] = -1
            V[j_base * p + k_base][r] = 1
            V[(j_base + 1) * p + k_base][r] = 1
            W[(k_base + 1) * n + 1][r] = 1
            
            # M7 = (A01-A11)(B01+B11)
            # A01: i=0, j=j_base+1 -> U index j_base + 1
            # A11: i=1, j=j_base+1 -> U index m + j_base + 1
            # B01: j=j_base, k=k_base+1 -> V index j_base*p + k_base+1
            # B11: j=j_base+1, k=k_base+1 -> V index (j_base+1)*p + k_base+1
            # C00: k=k_base, i=0 -> W index k_base*n + 0
            
            r = offset + 6
            U[j_base + 1][r] = 1
            U[m + j_base + 1][r] = -1
            V[j_base * p + k_base + 1][r] = 1
            V[(j_base + 1) * p + k_base + 1][r] = 1
            W[k_base * n + 0][r] = 1
            
            offset += 7
        
        # Part 2: n2=1 (i=2), p split into 4 blocks of 1x2x2
        for k_block in range(4):
            k_base = k_block * 2
            
            # A: i=2, j=j_base, j_base+1
            # B: j=j_base, j_base+1; k=k_base, k_base+1
            # C: i=2, k=k_base, k_base+1
            
            # T1: A(2,j_base)*B(j_base,k_base) -> C(k_base,2)
            r = offset
            U[2 * m + j_base][r] = 1
            V[j_base * p + k_base][r] = 1
            W[k_base * n + 2][r] = 1
            
            # T2: A(2,j_base)*B(j_base,k_base+1) -> C(k_base+1,2)
            r = offset + 1
            U[2 * m + j_base][r] = 1
            V[j_base * p + k_base + 1][r] = 1
            W[(k_base + 1) * n + 2][r] = 1
            
            # T3: A(2,j_base+1)*B(j_base+1,k_base) -> C(k_base,2)
            r = offset + 2
            U[2 * m + j_base + 1][r] = 1
            V[(j_base + 1) * p + k_base][r] = 1
            W[k_base * n + 2][r] = 1
            
            # T4: A(2,j_base+1)*B(j_base+1,k_base+1) -> C(k_base+1,2)
            r = offset + 3
            U[2 * m + j_base + 1][r] = 1
            V[(j_base + 1) * p + k_base + 1][r] = 1
            W[(k_base + 1) * n + 2][r] = 1
            
            offset += 4
    
    return {"U": U, "V": V, "W": W}
