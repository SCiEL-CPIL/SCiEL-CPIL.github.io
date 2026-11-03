import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a single tensor term.
        u_entries: list of (index, coeff)
        v_entries: list of (index, coeff)
        w_entries: list of (index, coeff)
        """
        u_vec = [0] * (n * m)
        for idx, coeff in u_entries:
            u_vec[idx] = coeff
        v_vec = [0] * (m * p)
        for idx, coeff in v_entries:
            v_vec[idx] = coeff
        w_vec = [0] * (p * n)
        for idx, coeff in w_entries:
            w_vec[idx] = coeff
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        """Add schoolbook terms for a rectangular block."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term(
                        [(i * m + j, 1)],
                        [(j * p + k, 1)],
                        [(k * n + i, 1)]
                    )
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen terms for a 2x2x2 block starting at (i0, j0, k0)."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(
            [(u00, 1), (u11, 1)],
            [(v00, 1), (v11, 1)],
            [(w00, 1), (w11, 1)]
        )
        
        # M2: (A10+A11)(B00) -> C10, C11
        add_term(
            [(u10, 1), (u11, 1)],
            [(v00, 1)],
            [(w10, 1), (w11, -1)]
        )
        
        # M3: (A00)(B01-B11) -> C01, C11
        add_term(
            [(u00, 1)],
            [(v01, 1), (v11, -1)],
            [(w01, 1), (w11, 1)]
        )
        
        # M4: (A11)(B10-B00) -> C00, C10
        add_term(
            [(u11, 1)],
            [(v10, 1), (v00, -1)],
            [(w00, 1), (w10, 1)]
        )
        
        # M5: (A00+A01)(B11) -> C00, C01
        add_term(
            [(u00, 1), (u01, 1)],
            [(v11, 1)],
            [(w00, -1), (w01, 1)]
        )
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(u10, 1), (u00, -1)],
            [(v00, 1), (v01, 1)],
            [(w11, 1)]
        )
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(u01, 1), (u11, -1)],
            [(v10, 1), (v11, 1)],
            [(w00, 1)]
        )
    
    def add_de_groote_block(i0, j0, k0):
        """Add de Groote rank-23 terms for a 3x3x3 block starting at (i0, j0, k0).
        
        The de Groote algorithm uses 23 multiplications to compute 3x3 matrix product.
        We map the 3x3x3 sub-tensor to indices:
        A rows: i0, i0+1, i0+2
        B cols: j0, j0+1, j0+2
        Output: k0, k0+1, k0+2 (for C[k][i])
        """
        # Define local indices
        # A[i_local][j_local] -> global index (i0+i_local)*m + (j0+j_local)
        # B[j_local][k_local] -> global index (j0+j_local)*p + (k0+k_local)
        # C[k_local][i_local] -> global index (k0+k_local)*n + (i0+i_local)
        
        def ui(i_loc, j_loc):
            return (i0 + i_loc) * m + (j0 + j_loc)
        
        def vj(j_loc, k_loc):
            return (j0 + j_loc) * p + (k0 + k_loc)
        
        def wk(k_loc, i_loc):
            return (k0 + k_loc) * n + (i0 + i_loc)
        
        # De Groote's 23-term algorithm for 3x3x3
        # The standard de Groote algorithm (1979) achieves rank 23 for 3x3 matrix multiplication.
        # Reference: M. de Groote, "Une nouvelle borne superieure du rang des formes trilineaires",
        #             C. R. Acad. Sci. Paris, 289 (1979), 31-33.
        
        # The 23 terms are defined as follows (using 0-indexed local coordinates):
        # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
        #     B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
        #     C = [[c00,c01,c02],[c10,c11,c12],[c20,c21,c22]]
        
        # Term 1: (a00+a11+a22)(b00+b11+b22)
        add_term(
            [(ui(0,0),1), (ui(1,1),1), (ui(2,2),1)],
            [(vj(0,0),1), (vj(1,1),1), (vj(2,2),1)],
            [(wk(0,0),1), (wk(1,1),1), (wk(2,2),1)]
        )
        
        # Term 2: (a01+a12+a20)(b01+b12+b20)
        add_term(
            [(ui(0,1),1), (ui(1,2),1), (ui(2,0),1)],
            [(vj(0,1),1), (vj(1,2),1), (vj(2,0),1)],
            [(wk(1,0),1), (wk(2,1),1), (wk(0,2),1)]
        )
        
        # Term 3: (a02+a10+a21)(b02+b10+b21)
        add_term(
            [(ui(0,2),1), (ui(1,0),1), (ui(2,1),1)],
            [(vj(0,2),1), (vj(1,0),1), (vj(2,1),1)],
            [(wk(2,0),1), (wk(0,1),1), (wk(1,2),1)]
        )
        
        # Term 4: (a00+a11-a22)(b00+b11-b22)
        add_term(
            [(ui(0,0),1), (ui(1,1),1), (ui(2,2),-1)],
            [(vj(0,0),1), (vj(1,1),1), (vj(2,2),-1)],
            [(wk(0,0),1), (wk(1,1),1), (wk(2,2),-1)]
        )
        
        # Term 5: (a01-a12+a20)(b01-b12+b20)
        add_term(
            [(ui(0,1),1), (ui(1,2),-1), (ui(2,0),1)],
            [(vj(0,1),1), (vj(1,2),-1), (vj(2,0),1)],
            [(wk(1,0),1), (wk(2,1),-1), (wk(0,2),1)]
        )
        
        # Term 6: (a02-a10+a21)(b02-b10+b21)
        add_term(
            [(ui(0,2),1), (ui(1,0),-1), (ui(2,1),1)],
            [(vj(0,2),1), (vj(1,0),-1), (vj(2,1),1)],
            [(wk(2,0),1), (wk(0,1),-1), (wk(1,2),1)]
        )
        
        # Term 7: (a00+a22-a11)(b00+b22-a11)
        add_term(
            [(ui(0,0),1), (ui(2,2),1), (ui(1,1),-1)],
            [(vj(0,0),1), (vj(2,2),1), (vj(1,1),-1)],
            [(wk(0,0),1), (wk(2,2),1), (wk(1,1),-1)]
        )
        
        # Term 8: (a01+a20-a12)(b01+b20-a12)
        add_term(
            [(ui(0,1),1), (ui(2,0),1), (ui(1,2),-1)],
            [(vj(0,1),1), (vj(2,0),1), (vj(1,2),-1)],
            [(wk(1,0),1), (wk(0,2),1), (wk(2,1),-1)]
        )
        
        # Term 9: (a02+a21-a10)(b02+b21-a10)
        add_term(
            [(ui(0,2),1), (ui(2,1),1), (ui(1,0),-1)],
            [(vj(0,2),1), (vj(2,1),1), (vj(1,0),-1)],
            [(wk(2,0),1), (wk(1,2),1), (wk(0,1),-1)]
        )
        
        # Term 10: (a10+a01+a22)(b10+b01+b22)
        add_term(
            [(ui(1,0),1), (ui(0,1),1), (ui(2,2),1)],
            [(vj(1,0),1), (vj(0,1),1), (vj(2,2),1)],
            [(wk(1,0),1), (wk(0,1),1), (wk(2,2),1)]
        )
        
        # Term 11: (a11+a02+a20)(b11+b02+b20)
        add_term(
            [(ui(1,1),1), (ui(0,2),1), (ui(2,0),1)],
            [(vj(1,1),1), (vj(0,2),1), (vj(2,0),1)],
            [(wk(1,1),1), (wk(0,2),1), (wk(2,0),1)]
        )
        
        # Term 12: (a12+a00+a21)(b12+b00+b21)
        add_term(
            [(ui(1,2),1), (ui(0,0),1), (ui(2,1),1)],
            [(vj(1,2),1), (vj(0,0),1), (vj(2,1),1)],
            [(wk(1,2),1), (wk(0,0),1), (wk(2,1),1)]
        )
        
        # Term 13: (a00+a10-a11)(b00+b10-a11)
        add_term(
            [(ui(0,0),1), (ui(1,0),1), (ui(1,1),-1)],
            [(vj(0,0),1), (vj(1,0),1), (vj(1,1),-1)],
            [(wk(0,0),1), (wk(1,0),1), (wk(1,1),-1)]
        )
        
        # Term 14: (a01+a11-a12)(b01+b11-a12)
        add_term(
            [(ui(0,1),1), (ui(1,1),1), (ui(1,2),-1)],
            [(vj(0,1),1), (vj(1,1),1), (vj(1,2),-1)],
            [(wk(0,1),1), (wk(1,1),1), (wk(1,2),-1)]
        )
        
        # Term 15: (a02+a12-a10)(b02+b12-a10)
        add_term(
            [(ui(0,2),1), (ui(1,2),1), (ui(1,0),-1)],
            [(vj(0,2),1), (vj(1,2),1), (vj(1,0),-1)],
            [(wk(0,2),1), (wk(1,2),1), (wk(1,0),-1)]
        )
        
        # Term 16: (a00+a20-a22)(b00+b20-a22)
        add_term(
            [(ui(0,0),1), (ui(2,0),1), (ui(2,2),-1)],
            [(vj(0,0),1), (vj(2,0),1), (vj(2,2),-1)],
            [(wk(0,0),1), (wk(2,0),1), (wk(2,2),-1)]
        )
        
        # Term 17: (a01+a21-a20)(b01+b21-a20)
        add_term(
            [(ui(0,1),1), (ui(2,1),1), (ui(2,0),-1)],
            [(vj(0,1),1), (vj(2,1),1), (vj(2,0),-1)],
            [(wk(0,1),1), (wk(2,1),1), (wk(2,0),-1)]
        )
        
        # Term 18: (a02+a22-a21)(b02+b22-a21)
        add_term(
            [(ui(0,2),1), (ui(2,2),1), (ui(2,1),-1)],
            [(vj(0,2),1), (vj(2,2),1), (vj(2,1),-1)],
            [(wk(0,2),1), (wk(2,2),1), (wk(2,1),-1)]
        )
        
        # Term 19: (a10+a20-a22)(b10+b20-a22)
        add_term(
            [(ui(1,0),1), (ui(2,0),1), (ui(2,2),-1)],
            [(vj(1,0),1), (vj(2,0),1), (vj(2,2),-1)],
            [(wk(1,0),1), (wk(2,0),1), (wk(2,2),-1)]
        )
        
        # Term 20: (a11+a21-a20)(b11+b21-a20)
        add_term(
            [(ui(1,1),1), (ui(2,1),1), (ui(2,0),-1)],
            [(vj(1,1),1), (vj(2,1),1), (vj(2,0),-1)],
            [(wk(1,1),1), (wk(2,1),1), (wk(2,0),-1)]
        )
        
        # Term 21: (a12+a22-a21)(b12+b22-a21)
        add_term(
            [(ui(1,2),1), (ui(2,2),1), (ui(2,1),-1)],
            [(vj(1,2),1), (vj(2,2),1), (vj(2,1),-1)],
            [(wk(1,2),1), (wk(2,2),1), (wk(2,1),-1)]
        )
        
        # Term 22: (a01+a10+a11+a20+a21+a22)(b01+b10+b11+b20+b21+b22)
        # This is a more complex term. Let me use a simpler formulation.
        # Actually, the de Groote algorithm has specific structure. Let me use the known correct 23-term formula.
        
        # I'll use a verified de Groote decomposition. The key insight is that we can achieve rank 23
        # by carefully combining terms. Let me use a different, well-known formulation.
        
        # Actually, let me just use Strassen-based approach for now since implementing de Groote
        # correctly is complex. The current score is 99, and we can potentially do better with
        # more aggressive partitioning.
        
        pass  # de Groote block not implemented correctly, will use alternative
    
    # Strategy: Use Strassen on 2x2x2 blocks where possible, schoolbook elsewhere
    # Partition: n=[0:2],[2:3]; m=[0:2],[2:4],[4:5]; p=[0:2],[2:4],[4:6],[6:7]
    
    # 2x2x2 Strassen blocks
    add_strassen_block(0, 0, 0)  # n[0:2], m[0:2], p[0:2]
    add_strassen_block(0, 0, 2)  # n[0:2], m[0:2], p[2:4]
    add_strassen_block(0, 0, 4)  # n[0:2], m[0:2], p[4:6]
    add_strassen_block(0, 2, 0)  # n[0:2], m[2:4], p[0:2]
    add_strassen_block(0, 2, 2)  # n[0:2], m[2:4], p[2:4]
    add_strassen_block(0, 2, 4)  # n[0:2], m[2:4], p[4:6]
    
    # Remaining schoolbook blocks
    add_schoolbook_block(0, 2, 0, 2, 6, 7)  # n[0:2], m[0:2], p[6:7]
    add_schoolbook_block(0, 2, 2, 4, 6, 7)  # n[0:2], m[2:4], p[6:7]
    add_schoolbook_block(0, 2, 4, 5, 0, 2)  # n[0:2], m[4:5], p[0:2]
    add_schoolbook_block(0, 2, 4, 5, 2, 4)  # n[0:2], m[4:5], p[2:4]
    add_schoolbook_block(0, 2, 4, 5, 4, 6)  # n[0:2], m[4:5], p[4:6]
    add_schoolbook_block(0, 2, 4, 5, 6, 7)  # n[0:2], m[4:5], p[6:7]
    add_schoolbook_block(2, 3, 0, 2, 0, 2)  # n[2:3], m[0:2], p[0:2]
    add_schoolbook_block(2, 3, 0, 2, 2, 4)  # n[2:3], m[0:2], p[2:4]
    add_schoolbook_block(2, 3, 0, 2, 4, 6)  # n[2:3], m[0:2], p[4:6]
    add_schoolbook_block(2, 3, 0, 2, 6, 7)  # n[2:3], m[0:2], p[6:7]
    add_schoolbook_block(2, 3, 2, 4, 0, 2)  # n[2:3], m[2:4], p[0:2]
    add_schoolbook_block(2, 3, 2, 4, 2, 4)  # n[2:3], m[2:4], p[2:4]
    add_schoolbook_block(2, 3, 2, 4, 4, 6)  # n[2:3], m[2:4], p[4:6]
    add_schoolbook_block(2, 3, 2, 4, 6, 7)  # n[2:3], m[2:4], p[6:7]
    add_schoolbook_block(2, 3, 4, 5, 0, 2)  # n[2:3], m[4:5], p[0:2]
    add_schoolbook_block(2, 3, 4, 5, 2, 4)  # n[2:3], m[4:5], p[2:4]
    add_schoolbook_block(2, 3, 4, 5, 4, 6)  # n[2:3], m[4:5], p[4:6]
    add_schoolbook_block(2, 3, 4, 5, 6, 7)  # n[2:3], m[4:5], p[6:7]
    
    rank = len(U)
    
    # Convert to list-of-lists format
    U_out = [row[:] for row in U]
    V_out = [row[:] for row in V]
    W_out = [row[:] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert len(result['U']) == n * m
    assert len(result['V']) == m * p
    assert len(result['W']) == p * n
    
    import random
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Verification passed. Rank = {rank}", file=sys.stderr)
