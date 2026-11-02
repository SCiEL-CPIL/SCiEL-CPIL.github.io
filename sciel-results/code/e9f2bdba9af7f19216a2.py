"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition factors
    # Local indices:
    # U row: i*2 + j
    # V row: j*2 + k
    # W row: k*2 + i
    # 7 columns
    
    # M1 = (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1 -> rows 0, 3
    # V: (0,0)=1, (1,1)=1 -> rows 0, 3
    # W: C00+=1, C11+=1 -> rows 0, 3
    s1 = {
        'U': [(0, 1), (3, 1)],
        'V': [(0, 1), (3, 1)],
        'W': [(0, 1), (3, 1)]
    }
    
    # M2 = (A10+A11)B00
    # U: (1,0)=1, (1,1)=1 -> rows 2, 3
    # V: (0,0)=1 -> row 0
    # W: C10+=1, C11-=1 -> rows 2, 3
    s2 = {
        'U': [(2, 1), (3, 1)],
        'V': [(0, 1)],
        'W': [(2, 1), (3, -1)]
    }
    
    # M3 = A00(B01-B11)
    # U: (0,0)=1 -> row 0
    # V: (0,1)=1, (1,1)=-1 -> rows 1, 3
    # W: C01+=1, C11+=1 -> rows 1, 3
    s3 = {
        'U': [(0, 1)],
        'V': [(1, 1), (3, -1)],
        'W': [(1, 1), (3, 1)]
    }
    
    # M4 = A11(B10-B00)
    # U: (1,1)=1 -> row 3
    # V: (1,0)=1, (0,0)=-1 -> rows 2, 0
    # W: C00+=1, C10+=1 -> rows 0, 2
    s4 = {
        'U': [(3, 1)],
        'V': [(2, 1), (0, -1)],
        'W': [(0, 1), (2, 1)]
    }
    
    # M5 = (A00+A01)B11
    # U: (0,0)=1, (0,1)=1 -> rows 0, 1
    # V: (1,1)=1 -> row 3
    # W: C00-=1 -> row 0
    s5 = {
        'U': [(0, 1), (1, 1)],
        'V': [(3, 1)],
        'W': [(0, -1)]
    }
    
    # M6 = (A10-A00)(B00+B01)
    # U: (1,0)=1, (0,0)=-1 -> rows 2, 0
    # V: (0,0)=1, (0,1)=1 -> rows 0, 1
    # W: C11+=1 -> row 3
    s6 = {
        'U': [(2, 1), (0, -1)],
        'V': [(0, 1), (1, 1)],
        'W': [(3, 1)]
    }
    
    # M7 = (A01-A11)(B10+B11)
    # U: (0,1)=1, (1,1)=-1 -> rows 1, 3
    # V: (1,0)=1, (1,1)=1 -> rows 2, 3
    # W: C00+=1 -> row 0
    s7 = {
        'U': [(1, 1), (3, -1)],
        'V': [(2, 1), (3, 1)],
        'W': [(0, 1)]
    }
    
    strassen_factors = [s1, s2, s3, s4, s5, s6, s7]
    
    # Block definitions:
    # (A_offset_i, A_offset_j, B_offset_j, B_offset_k, C_offset_k, C_offset_i)
    # Global U row: i*4 + j
    # Global V row: j*4 + k
    # Global W row: k*2 + i
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i in 0..1, j in 0..1, k in 0..1
    # U map: local u (i*2+j) -> global i*4+j
    # V map: local v (j*2+k) -> global j*4+k
    # W map: local w (k*2+i) -> global k*2+i
    
    blocks = [
        # (u_map, v_map, w_map)
        # Block 1
        (
            [0, 1, 4, 5], # u: 0->0, 1->1, 2->4, 3->5
            [0, 1, 4, 5], # v: 0->0, 1->1, 2->4, 3->5
            [0, 1, 2, 3]  # w: 0->0, 1->1, 2->2, 3->3
        ),
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # i in 0..1, j' in 0..1 (j=j'+2), k in 0..1
        # U map: local u (i*2+j') -> global i*4+(j'+2)
        # V map: local v (j'*2+k) -> global (j'+2)*4+k
        # W map: local w (k*2+i) -> global k*2+i
        (
            [2, 3, 6, 7], # u: 0->2, 1->3, 2->6, 3->7
            [8, 9, 12, 13], # v: 0->8, 1->9, 2->12, 3->13
            [0, 1, 2, 3]  # w: 0->0, 1->1, 2->2, 3->3
        ),
        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        # i in 0..1, j in 0..1, k' in 0..1 (k=k'+2)
        # U map: local u (i*2+j) -> global i*4+j
        # V map: local v (j*2+k') -> global j*4+(k'+2)
        # W map: local w (k'*2+i) -> global (k'+2)*2+i
        (
            [0, 1, 4, 5], # u: 0->0, 1->1, 2->4, 3->5
            [2, 3, 6, 7], # v: 0->2, 1->3, 2->6, 3->7
            [4, 5, 6, 7]  # w: 0->4, 1->5, 2->6, 3->7
        ),
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        # i in 0..1, j' in 0..1 (j=j'+2), k' in 0..1 (k=k'+2)
        # U map: local u (i*2+j') -> global i*4+(j'+2)
        # V map: local v (j'*2+k') -> global (j'+2)*4+(k'+2)
        # W map: local w (k'*2+i) -> global (k'+2)*2+i
        (
            [2, 3, 6, 7], # u: 0->2, 1->3, 2->6, 3->7
            [10, 11, 14, 15], # v: 0->10, 1->11, 2->14, 3->15
            [4, 5, 6, 7]  # w: 0->4, 1->5, 2->6, 3->7
        )
    ]
    
    col_idx = 0
    for u_map, v_map, w_map in blocks:
        for s in strassen_factors:
            for row, val in s['U']:
                U[u_map[row]][col_idx] += val
            for row, val in s['V']:
                V[v_map[row]][col_idx] += val
            for row, val in s['W']:
                W[w_map[row]][col_idx] += val
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
