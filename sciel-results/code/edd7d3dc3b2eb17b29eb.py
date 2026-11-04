"""Rewritten program for matrix multiplication 2x3 by 3x3 using optimized rank."""

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Ranks:
    # 1: j=2 term
    # 7: Strassen for j in {0,1}, l in {0,1}
    # 4: Schoolbook for j in {0,1}, l=2
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, l): return j * p + l
    def w_idx(l, i): return l * n + i
    
    # Rank 0: j=2 term
    # A_{i,2} B_{2,l} -> C_{i,l}
    # U: i in {0,1}, j=2
    U[u_idx(0, 2)][0] = 1
    U[u_idx(1, 2)][0] = 1
    # V: j=2, l in {0,1,2}
    V[v_idx(2, 0)][0] = 1
    V[v_idx(2, 1)][0] = 1
    V[v_idx(2, 2)][0] = 1
    # W: l in {0,1,2}, i in {0,1}
    for l in range(p):
        for i in range(n):
            W[w_idx(l, i)][0] = 1

    # Ranks 1-7: Strassen for j in {0,1}, l in {0,1}
    # Indices involved:
    # A: u00(0,0), u01(0,1), u10(1,0), u11(1,1)
    # B: v00(0,0), v01(0,1), v10(1,0), v11(1,1)
    # C: c00(0,0), c01(1,0), c10(0,1), c11(1,1) -> w_idx(l,i)
    # w00 = w(0,0) = 0
    # w01 = w(1,0) = 1  (l=1, i=0)
    # w10 = w(0,1) = 2  (l=0, i=1)
    # w11 = w(1,1) = 3
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    U[u_idx(0,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1; V[v_idx(1,1)][1] = 1
    W[w_idx(0,0)][1] = 1; W[w_idx(1,1)][1] = 1
    
    # M2: (A10+A11)B00 -> C10-C11
    U[u_idx(1,0)][2] = 1; U[u_idx(1,1)][2] = 1
    V[v_idx(0,0)][2] = 1
    W[w_idx(0,1)][2] = 1; W[w_idx(1,1)][2] = -1
    
    # M3: A00(B01-B11) -> C01+C11
    U[u_idx(0,0)][3] = 1
    V[v_idx(0,1)][3] = 1; V[v_idx(1,1)][3] = -1
    W[w_idx(1,0)][3] = 1; W[w_idx(1,1)][3] = 1
    
    # M4: A11(B10-B00) -> C00+C10
    U[u_idx(1,1)][4] = 1
    V[v_idx(1,0)][4] = 1; V[v_idx(0,0)][4] = -1
    W[w_idx(0,0)][4] = 1; W[w_idx(0,1)][4] = 1
    
    # M5: (A00+A01)B11 -> -C00+C01
    U[u_idx(0,0)][5] = 1; U[u_idx(0,1)][5] = 1
    V[v_idx(1,1)][5] = 1
    W[w_idx(0,0)][5] = -1; W[w_idx(1,0)][5] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[u_idx(1,0)][6] = 1; U[u_idx(0,0)][6] = -1
    V[v_idx(0,0)][6] = 1; V[v_idx(0,1)][6] = 1
    W[w_idx(1,1)][6] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00+C11
    U[u_idx(0,1)][7] = 1; U[u_idx(1,1)][7] = -1
    V[v_idx(1,0)][7] = 1; V[v_idx(1,1)][7] = 1
    W[w_idx(0,0)][7] = 1; W[w_idx(1,1)][7] = 1
    
    # Ranks 8-11: Schoolbook for j in {0,1}, l=2
    # C_{i,2} = A_{i,0}B_{0,2} + A_{i,1}B_{1,2}
    # u_idx(0,0), u_idx(1,0) for B_{0,2} (v_idx(0,2))
    # u_idx(0,1), u_idx(1,1) for B_{1,2} (v_idx(1,2))
    # w_idx(2,0), w_idx(2,1)
    
    # Rank 8: A00 B02
    U[u_idx(0,0)][8] = 1
    V[v_idx(0,2)][8] = 1
    W[w_idx(2,0)][8] = 1
    
    # Rank 9: A10 B02
    U[u_idx(1,0)][9] = 1
    V[v_idx(0,2)][9] = 1
    W[w_idx(2,1)][9] = 1
    
    # Rank 10: A01 B12
    U[u_idx(0,1)][10] = 1
    V[v_idx(1,2)][10] = 1
    W[w_idx(2,0)][10] = 1
    
    # Rank 11: A11 B12
    U[u_idx(1,1)][11] = 1
    V[v_idx(1,2)][11] = 1
    W[w_idx(2,1)][11] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
