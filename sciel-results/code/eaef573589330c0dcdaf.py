"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2_triples():
    """Return list of (u, v, w) index triples for Strassen 2x2 rank-7 decomposition.
    For 2x2*2x2: C = A*B where A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}.
    Strassen:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
      C00 = M1+M4-M5+M7
      C01 = M3+M5
      C10 = M2+M4
      C11 = M1-M2+M3+M6
    Each M is a product of a linear combo of A entries and a linear combo of B entries.
    We represent each rank-1 as U-row-vector (over A indices), V-row-vector (over B indices), W-row-vector (over C indices).
    A indices: (i,j) -> idx = i*2+j  (0..3)
    B indices: (j,k) -> idx = j*2+k  (0..3)
    C indices: (i,k) -> idx = i*2+k  (0..3)
    """
    # Define linear forms for each M
    # A linear form: dict mapping (i,j) -> coeff
    # B linear form: dict mapping (j,k) -> coeff
    # C linear form: dict mapping (i,k) -> coeff (with signs)
    
    def Aform(pairs):
        return pairs  # list of ((i,j), coeff)
    def Bform(pairs):
        return pairs
    def Cform(pairs):
        return pairs
    
    # M1 = (A00+A11)*(B00+B11)
    M1 = (Aform([((0,0),1),((1,1),1)]), Bform([((0,0),1),((1,1),1)]))
    M2 = (Aform([((1,0),1),((1,1),1)]), Bform([((0,0),1)]))
    M3 = (Aform([((0,0),1)]), Bform([((0,1),1),((1,1),-1)]))
    M4 = (Aform([((1,1),1)]), Bform([((1,0),1),((0,0),-1)]))
    M5 = (Aform([((0,0),1),((0,1),1)]), Bform([((1,1),1)]))
    M6 = (Aform([((1,0),1),((0,0),-1)]), Bform([((0,0),1),((0,1),1)]))
    M7 = (Aform([((0,1),1),((1,1),-1)]), Bform([((1,0),1),((1,1),1)]))
    
    # C entries as linear combos of Ms
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    c_forms = {
        (0,0): [(0,1),(3,1),(4,-1),(6,1)],   # M1,M4,-M5,M7
        (0,1): [(2,1),(4,1)],                  # M3,M5
        (1,0): [(1,1),(3,1)],                  # M2,M4
        (1,1): [(0,1),(1,-1),(2,1),(5,1)],     # M1,-M2,M3,M6
    }
    
    ms = [M1, M2, M3, M4, M5, M6, M7]
    triples = []
    for m_idx, (af, bf) in enumerate(ms):
        for (c_idx, coeff) in [e for e in c_forms.values() for e in [e] if False]:
            pass
        # Collect which C entries use this M
        c_entries = []
        for c_key, ms_list in c_forms.items():
            for (m_i, c) in ms_list:
                if m_i == m_idx:
                    c_entries.append((c_key, c))
        triples.append((af, bf, c_entries))
    return triples

def construct():
    n, m, p = DIMENSIONS  # 4,4,4
    # Partition 4x4 into 2x2 blocks.
    # A[i,j] for i,j in 0..3. Block (bi,bj) covers rows bi*2..bi*2+1, cols bj*2..bj*2+1.
    # A block (bi,bj): local (li,lj) -> global (bi*2+li, bj*2+lj)
    # C[i,k] block (bi,bk): local (li,lk) -> global (bi*2+li, bk*2+lk)
    #
    # C_{bi,bk} = sum_{bj} A_{bi,bj} * B_{bj,bk}
    # 8 terms: (bi,bj,bk) for bi,bj,bk in {0,1}
    
    # For each of the 8 block-multiplications, use Strassen rank-7.
    # A global index: i*4+j  (row-major, i in 0..3, j in 0..3)
    # B global index: j*4+k
    # C global index (trace convention W[k*n+i]): k*4+i
    
    rank_total = 8 * 7  # 56
    
    U = [[0]*rank_total for _ in range(n*m)]  # 16 x 56
    V = [[0]*rank_total for _ in range(m*p)]  # 16 x 56
    W = [[0]*rank_total for _ in range(p*n)]  # 16 x 56
    
    triples = _strassen_2x2_triples()
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for (af, bf, c_entries) in triples:
                    # af: list of ((li,lj), coeff) for A block (bi,bj)
                    # bf: list of ((lj,lk), coeff) for B block (bj,bk)
                    # c_entries: list of ((li,lk), coeff) for C block (bi,bk)
                    for ((li, lj), ca) in af:
                        gi = bi*2 + li
                        gj = bj*2 + lj
                        U[gi*4 + gj][col] += ca
                    for ((lj, lk), cb) in bf:
                        gj = bj*2 + lj
                        gk = bk*2 + lk
                        V[gj*4 + gk][col] += cb
                    for ((li, lk), cc) in c_entries:
                        gi = bi*2 + li
                        gk = bk*2 + lk
                        W[gk*4 + gi][col] += cc
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
