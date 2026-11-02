import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_decomp():
        # 2x2x2 rank 7
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-d)(f+g)
        # P7 = (b-d)(e+g)
        # C11 = P1+P4-P5+P7
        # C12 = P3+P5
        # C21 = P2+P4
        # C22 = P1-P2+P3+P6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Indices: U[0]=a, U[1]=b, U[2]=c, U[3]=d
        # V[0]=e, V[1]=f, V[2]=g, V[3]=h
        # W[0]=C11, W[1]=C12, W[2]=C21, W[3]=C22
        
        # P1: (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C11, C22
        
        # P2: (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1 # C21, C22
        
        # P3: a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1 # C12, C22
        
        # P4: d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C11, C21
        
        # P5: (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # C11, C12
        
        # P6: (c-d)(f+g)
        U[2][5] = 1; U[3][5] = -1
        V[1][5] = 1; V[2][5] = 1
        W[3][5] = 1 # C22
        
        # P7: (b-d)(e+g)
        U[1][6] = 1; U[3][6] = -1
        V[0][6] = 1; V[2][6] = 1
        W[0][6] = 1 # C11
        
        return U, V, W

    def combine_decomps(decomps, n, m, p, i_off, j_off, k_off):
        """
        Combines a list of sub-decompositions into one.
        decomps: list of (U, V, W, n_sub, m_sub, p_sub)
        n, m, p: dimensions of the parent problem
        i_off, j_off, k_off: offsets for the parent indices
        """
        total_rank = sum(len(u[0]) for u, v, w, _, _, _ in decomps)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        rank_offset = 0
        for u_sub, v_sub, w_sub, n_sub, m_sub, p_sub in decomps:
            r_sub = len(u_sub[0])
            
            # Map U
            for i_local in range(n_sub):
                for j_local in range(m_sub):
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    u_row = i_global * m + j_global
                    u_sub_row = i_local * m_sub + j_local
                    for r in range(r_sub):
                        U[u_row][rank_offset + r] = u_sub[u_sub_row][r]
                        
            # Map V
            for j_local in range(m_sub):
                for k_local in range(p_sub):
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    v_row = j_global * p + k_global
                    v_sub_row = j_local * p_sub + k_local
                    for r in range(r_sub):
                        V[v_row][rank_offset + r] = v_sub[v_sub_row][r]
                        
            # Map W
            for k_local in range(p_sub):
                for i_local in range(n_sub):
                    k_global = k_off + k_local
                    i_global = i_off + i_local
                    w_row = k_global * n + i_global
                    w_sub_row = k_local * n_sub + i_local
                    for r in range(r_sub):
                        W[w_row][rank_offset + r] = w_sub[w_sub_row][r]
                        
            rank_offset += r_sub
            
        return U, V, W

    # Recursive solver
    def solve(n, m, p):
        if time.time() > deadline:
             return schoolbook_decomp(n, m, p)
             
        # Base cases: if any dimension is 1, use schoolbook
        if n == 1 or m == 1 or p == 1:
            return schoolbook_decomp(n, m, p)
            
        # Try to split the largest dimension
        # Heuristic: Split m first if m >= 2, then p, then n
        # We want to create 2x2x2 blocks for Strassen
        
        # Strategy: Split m into 2 and m-2 if m >= 4
        # Or split p into 2 and p-2 if p >= 4
        # Or split n into 2 and n-2 if n >= 4
        
        # For 3x4x7:
        # Split m=4 into 2, 2.
        # Sub 1: 3x2x7
        # Sub 2: 3x2x7
        
        # For 3x2x7:
        # Split p=7 into 2, 2, 2, 1.
        # Sub 1: 3x2x2
        # Sub 2: 3x2x2
        # Sub 3: 3x2x2
        # Sub 4: 3x2x1
        
        # For 3x2x2:
        # Split n=3 into 2, 1.
        # Sub 1: 2x2x2 (Strassen)
        # Sub 2: 1x2x2 (Schoolbook)
        
        # Let's implement a generic recursive splitter that tries to form 2x2x2 blocks.
        
        # Determine split dimension
        # If n >= 2, we can split n.
        # If m >= 2, we can split m.
        # If p >= 2, we can split p.
        
        # We prioritize splitting the dimension that allows creating 2x2x2 blocks.
        # Actually, let's just try splitting the largest dimension in half (or into 2 and rest)
        # and recursively solve.
        
        # To ensure we get the 78 rank, we need specific splits.
        # 3x4x7 -> m split 2,2 -> 3x2x7, 3x2x7
        # 3x2x7 -> p split 2,2,2,1 -> 3x2x2, 3x2x2, 3x2x2, 3x2x1
        # 3x2x2 -> n split 2,1 -> 2x2x2, 1x2x2
        
        # Let's hardcode the strategy for efficiency and correctness for this specific case
        # but keep it general enough.
        
        if m >= 4:
            m1 = 2
            m2 = m - 2
            U1, V1, W1 = solve(n, m1, p)
            U2, V2, W2 = solve(n, m2, p)
            return combine_decomps([(U1, V1, W1, n, m1, p), (U2, V2, W2, n, m2, p)], n, m, p, 0, 0, 0)
        elif p >= 4:
            p1 = 2
            p2 = p - 2
            U1, V1, W1 = solve(n, m, p1)
            U2, V2, W2 = solve(n, m, p2)
            return combine_decomps([(U1, V1, W1, n, m, p1), (U2, V2, W2, n, m, p2)], n, m, p, 0, 0, 0)
        elif n >= 3:
            n1 = 2
            n2 = n - 2
            U1, V1, W1 = solve(n1, m, p)
            U2, V2, W2 = solve(n2, m, p)
            return combine_decomps([(U1, V1, W1, n1, m, p), (U2, V2, W2, n2, m, p)], n, m, p, 0, 0, 0)
        else:
            # Small cases like 2x2x2, 2x2x1, etc.
            if n == 2 and m == 2 and p == 2:
                U, V, W = strassen_decomp()
                return U, V, W
            else:
                return schoolbook_decomp(n, m, p)

    # Execute the recursive solver
    U, V, W = solve(n, m, p)
    
    # Verify
    # The verification is expensive for large rank, but 78 is small.
    # We can skip verification in the final submission if we are confident, 
    # but for safety, let's do a quick check.
    
    # Convert to numpy for fast check
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check rank
    rank = U_np.shape[1]
    
    # Check correctness
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can check a few random points or all points if fast enough.
    # 3*4*7 = 84 points. 78 ranks. 84*78 = 6552 ops. Very fast.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    actual += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(actual - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if something went wrong
        U, V, W = schoolbook_decomp(n, m, p)
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
