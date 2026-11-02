import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 5, 6
    
    U = []
    V = []
    W = []
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # We'll build the decomposition in blocks
    # Block 1: j=(0,1,2), k=(0,1,2) -> 2x3x3, rank 15
    # Block 2: j=(0,1,2), k=(3,4,5) -> 2x3x3, rank 15
    # Block 3: j=(3,4), k=(0,1,2) -> 2x2x3, rank 11
    # Block 4: j=(3,4), k=(3,4,5) -> 2x2x3, rank 11
    # Total rank: 15+15+11+11 = 52

    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    # 2x3x3 rank 15 decomposition
    # Using known optimal decomposition for M_{2,3,3}
    def add_2x3x3(j1, j2, j3, k1, k2, k3):
        # This is a known rank-15 decomposition for 2x3x3
        # a is 2x3, b is 3x3, c is 2x3
        # Rows of a: i=0,1; Cols of a: j1,j2,j3
        # Rows of b: j1,j2,j3; Cols of b: k1,k2,k3
        # C is 2x3: rows i=0,1; cols k1,k2,k3
        
        # Using a construction based on Strassen-like ideas
        # m1 = a00*b00
        add_rank_step([(u_idx(0,j1), 1)], [(v_idx(j1,k1), 1)], [(w_idx(k1,0), 1)])
        # m2 = a00*b01
        add_rank_step([(u_idx(0,j1), 1)], [(v_idx(j1,k2), 1)], [(w_idx(k2,0), 1)])
        # m3 = a00*b02
        add_rank_step([(u_idx(0,j1), 1)], [(v_idx(j1,k3), 1)], [(w_idx(k3,0), 1)])
        # m4 = a01*b10
        add_rank_step([(u_idx(0,j2), 1)], [(v_idx(j2,k1), 1)], [(w_idx(k1,0), 1)])
        # m5 = a01*b11
        add_rank_step([(u_idx(0,j2), 1)], [(v_idx(j2,k2), 1)], [(w_idx(k2,0), 1)])
        # m6 = a01*b12
        add_rank_step([(u_idx(0,j2), 1)], [(v_idx(j2,k3), 1)], [(w_idx(k3,0), 1)])
        # m7 = a02*b20
        add_rank_step([(u_idx(0,j3), 1)], [(v_idx(j3,k1), 1)], [(w_idx(k1,0), 1)])
        # m8 = a02*b21
        add_rank_step([(u_idx(0,j3), 1)], [(v_idx(j3,k2), 1)], [(w_idx(k2,0), 1)])
        # m9 = a02*b22
        add_rank_step([(u_idx(0,j3), 1)], [(v_idx(j3,k3), 1)], [(w_idx(k3,0), 1)])
        # m10 = a10*b00
        add_rank_step([(u_idx(1,j1), 1)], [(v_idx(j1,k1), 1)], [(w_idx(k1,1), 1)])
        # m11 = a10*b01
        add_rank_step([(u_idx(1,j1), 1)], [(v_idx(j1,k2), 1)], [(w_idx(k2,1), 1)])
        # m12 = a10*b02
        add_rank_step([(u_idx(1,j1), 1)], [(v_idx(j1,k3), 1)], [(w_idx(k3,1), 1)])
        # m13 = a11*b10
        add_rank_step([(u_idx(1,j2), 1)], [(v_idx(j2,k1), 1)], [(w_idx(k1,1), 1)])
        # m14 = a11*b11
        add_rank_step([(u_idx(1,j2), 1)], [(v_idx(j2,k2), 1)], [(w_idx(k2,1), 1)])
        # m15 = a11*b12
        add_rank_step([(u_idx(1,j2), 1)], [(v_idx(j2,k3), 1)], [(w_idx(k3,1), 1)])
        # m16 = a12*b20
        add_rank_step([(u_idx(1,j3), 1)], [(v_idx(j3,k1), 1)], [(w_idx(k1,1), 1)])
        # m17 = a12*b21
        add_rank_step([(u_idx(1,j3), 1)], [(v_idx(j3,k2), 1)], [(w_idx(k2,1), 1)])
        # m18 = a12*b22
        add_rank_step([(u_idx(1,j3), 1)], [(v_idx(j3,k3), 1)], [(w_idx(k3,1), 1)])
        # That's 18, not 15. I need a better decomposition.
        # Let me use a known rank-15 construction.
        
    # Actually, the above is schoolbook (18). I need to find the rank-15 one.
    # Let me use a different approach - I'll use the known result that rank(M_{2,3,3})=15
    # and implement it properly.
    
    # For now, let me try a different strategy: use the current 54-rank solution but verify it works
    # and see if I can find any small improvement.
    
    # Let me reset and use the proven 54-rank approach but double-check correctness
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_rank_step2(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    def add_strassen_block(j1, j2, k1, k2):
        # m1 = (a00+a11)(b00+b11)
        add_rank_step2([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                      [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                      [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
        # m2 = (a10+a11)*b00
        add_rank_step2([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                      [(v_idx(j1,k1), 1)], 
                      [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
        # m3 = a00*(b01-b11)
        add_rank_step2([(u_idx(0,j1), 1)], 
                      [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                      [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
        # m4 = a11*(b10-b00)
        add_rank_step2([(u_idx(1,j2), 1)], 
                      [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                      [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
        # m5 = (a00+a01)*b11
        add_rank_step2([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                      [(v_idx(j2,k2), 1)], 
                      [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
        # m6 = (a10-a00)*(b00+b01)
        add_rank_step2([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                      [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                      [(w_idx(k2,1), 1)])
        # m7 = (a01-a11)*(b10+b11)
        add_rank_step2([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                      [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                      [(w_idx(k1,0), 1)])

    def add_schoolbook_2x1x2(j, k1, k2):
        add_rank_step2([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
        add_rank_step2([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
        add_rank_step2([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
        add_rank_step2([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])

    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    for j_range in j_groups:
        for k_range in k_groups:
            if len(j_range) == 2 and len(k_range) == 2:
                add_strassen_block(j_range[0], j_range[1], k_range[0], k_range[1])
            elif len(j_range) == 1 and len(k_range) == 2:
                add_schoolbook_2x1x2(j_range[0], k_range[0], k_range[1])
            else:
                for j in j_range:
                    for k in k_range:
                        for i in range(n):
                            add_rank_step2([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: for all i,j,k: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should be 1 if this is a valid rank-1 term
    # Actually, the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 for the matrix multiplication tensor
    
    # Let's verify by checking a few entries
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    # Wait, no. The matrix multiplication tensor T_{ijk} = delta_{...} 
    # Actually for matrix multiplication A(2x5) * B(5x6) = C(2x6):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where i in [0,2), j in [0,5), k in [0,6)
    # T[i,j,k] = 1 (it's the identity in the middle dimension)
    
    # The rank-r decomposition: T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where W[k*n+i, r] corresponds to W[k,i,r]
    
    # Verify
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    
    if errors > 0:
        # Fallback to pure schoolbook if there's an error
        rank = 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
