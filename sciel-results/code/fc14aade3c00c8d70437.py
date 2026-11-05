import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0

    # ---------- Helper: Build a block decomposition ----------
    def build_block_decomp(strassen_rank, sub_rank_2x2x3):
        """
        Builds a 2x4x5 decomposition by splitting the 5 output cols into 2 and 3.
        2x4x2 part: two 2x2x2 blocks (Strassen).
        2x4x3 part: two 2x2x3 blocks (using provided sub_rank solution).
        """
        rank_2x2x2 = strassen_rank
        # We need to construct the actual U, V, W for the blocks.
        # This function returns the full U, V, W matrices.
        pass # Implemented below

    # ---------- Strassen 2x2x2 Decomposition ----------
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # Standard Strassen
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices: 0:11, 1:12, 2:21, 3:22
        # U (A), V (B), W (C)
        
        # Term 1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # Term 2
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] -= 1
        
        # Term 3
        U[0][2] += 1
        V[2][2] += 1; V[3][2] -= 1
        W[1][2] += 1; W[3][2] += 1
        
        # Term 4
        U[3][3] += 1
        V[1][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[2][3] += 1
        
        # Term 5
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[1][4] += 1
        
        # Term 6
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # Term 7
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1
        
        return U, V, W

    # ---------- Schoolbook 2x2x3 Decomposition ----------
    def get_schoolbook_2x2x3():
        # Rank 12
        U = [[0]*12 for _ in range(4)]
        V = [[0]*12 for _ in range(6)] # 2x3 = 6 elements
        W = [[0]*12 for _ in range(6)] # 2x3 = 6 elements
        
        # A: 0:11, 1:12, 2:21, 3:22
        # B: 0:11, 1:12, 2:13, 3:21, 4:22, 5:23
        # C: 0:11, 1:12, 2:13, 3:21, 4:22, 5:23
        
        # C11 = A11B11 + A21B21
        # C12 = A11B12 + A21B22
        # C13 = A11B13 + A21B23
        # C21 = A12B11 + A22B21
        # C22 = A12B12 + A22B22
        # C23 = A12B13 + A22B23
        
        # This is not a sum of rank-1 terms in the sense of tensor decomposition of the multiplication map?
        # The tensor is $T_{ijk}$.
        # Rank 1 term is $u \otimes v \otimes w$.
        # Schoolbook corresponds to:
        # Sum over i,j,k of A_ij B_jk C_ik.
        # Each term $A_{ij} B_{jk} C_{ik}$ is a rank 1 tensor.
        # There are 12 such terms.
        
        terms = [
            (0, 0, 0), # A11 B11 C11
            (0, 1, 1), # A11 B12 C12
            (0, 2, 2), # A11 B13 C13
            (2, 3, 0), # A21 B21 C11
            (2, 4, 1), # A21 B22 C12
            (2, 5, 2), # A21 B23 C13
            (1, 0, 3), # A12 B11 C21
            (1, 1, 4), # A12 B12 C22
            (1, 2, 5), # A12 B13 C23
            (3, 3, 3), # A22 B21 C21
            (3, 4, 4), # A22 B22 C22
            (3, 5, 5)  # A22 B23 C23
        ]
        
        for idx, (u_idx, v_idx, w_idx) in enumerate(terms):
            U[u_idx][idx] = 1
            V[v_idx][idx] = 1
            W[w_idx][idx] = 1
            
        return U, V, W

    # ---------- Greedy Extraction for 2x2x3 ----------
    def reduce_2x2x3(target_rank, time_budget):
        # Try to reduce the rank of the 2x2x3 block from 12 to target_rank
        num_u, num_v, num_w = 4, 6, 6
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        # Target tensor: A(2x2) * B(2x3) = C(2x3)
        # T[u,v,w] = 1 if u,i,v,j,w,k form valid multiplication
        # u: 0:11, 1:12, 2:21, 3:22
        # v: 0:11, 1:12, 2:13, 3:21, 4:22, 5:23
        # w: 0:11, 1:12, 2:13, 3:21, 4:22, 5:23
        
        for i in range(2):
            for k in range(3):
                c_idx = k * 2 + i # w index: k*2 + i? 
                # My indexing above: C: 0:11, 1:12, 2:13, 3:21, 4:22, 5:23
                # i=0, k=0 -> 11 -> 0
                # i=0, k=1 -> 12 -> 1
                # i=0, k=2 -> 13 -> 2
                # i=1, k=0 -> 21 -> 3
                # i=1, k=1 -> 22 -> 4
                # i=1, k=2 -> 23 -> 5
                c_idx = k * 2 + i
                for j in range(2):
                    a_idx = i * 2 + j # u index: i*2+j
                    # i=0,j=0->0, i=0,j=1->1, i=1,j=0->2, i=1,j=1->3
                    b_idx = j * 3 + k # v index: j*3+k
                    # j=0,k=0->0, j=0,k=1->1, j=0,k=2->2, j=1,k=0->3, j=1,k=1->4, j=1,k=2->5
                    T[a_idx][b_idx][c_idx] = Fraction(1)

        # Start with schoolbook
        U0, V0, W0 = get_schoolbook_2x2x3()
        U0 = [[Fraction(x) for x in row] for row in U0]
        V0 = [[Fraction(x) for x in row] for row in V0]
        W0 = [[Fraction(x) for x in row] for row in W0]
        
        R = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for r in range(12):
            for a_idx in range(num_u):
                ua = U0[a_idx][r]
                if ua:
                    for b_idx in range(num_v):
                        vb = V0[b_idx][r]
                        if vb:
                            uv = ua * vb
                            if uv:
                                for c_idx in range(num_w):
                                    wc = W0[c_idx][r]
                                    if wc:
                                        R[a_idx][b_idx][c_idx] -= uv * wc
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    rb[c_idx] = rb[c_idx] + T[a_idx][b_idx][c_idx]

        U_list, V_list, W_list = [], [], []

        def pick_candidate():
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    rb = ra[b_idx]
                    for c_idx in range(num_w):
                        if rb[c_idx] != 0:
                            return a_idx, b_idx, c_idx, rb[c_idx]
            return None

        def try_extract(a_idx, b_idx, c_idx, val, den):
            # Same logic as before, adapted for sizes
            S = [[0] * num_v for _ in range(num_u)]
            for u_i in range(num_u):
                row = S[u_i]
                for b_i in range(num_v):
                    f = R[u_i][b_i][c_idx]
                    if f:
                        row[b_i] = int(f * den)
            
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx:
                    continue
                f = R[u_i][a_idx][c_idx]
                if f:
                    u[u_i] = Fraction(-int(f * den), den)
            
            v = S[b_idx][:]
            
            Sv = [0] * num_u
            for u_i in range(num_u):
                row = S[u_i]
                s = 0
                for b_i in range(num_v):
                    if row[b_i] and v[b_i]:
                        s += row[b_i] * v[b_i]
                Sv[u_i] = s
            
            u_int = [0] * num_u
            for u_i in range(num_u):
                if u[u_i]:
                    u_int[u_i] = int(u[u_i] * den)
            
            if any(Sv[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False
            
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w[w_i] = Fraction(-int(f * den), den)
            
            S2 = [[0] * num_w for _ in range(num_u)]
            for u_i in range(num_u):
                row = S2[u_i]
                for w_i in range(num_w):
                    f = R[u_i][b_idx][w_i]
                    if f:
                        row[w_i] = int(f * den)
            
            S2w = [0] * num_u
            for u_i in range(num_u):
                row = S2[u_i]
                s = 0
                for w_i in range(num_w):
                    if row[w_i]:
                        w_int = int(w[w_i] * den)
                        if w_int:
                            s += row[w_i] * w_int
                S2w[u_i] = s
            
            if any(S2w[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False

            for u_i in range(num_u):
                Su = S[u_i]
                for b_i in range(num_v):
                    if not Su[b_i]:
                        continue
                    vv = v[b_i]
                    if not vv:
                        continue
                    base_val = Fraction(Su[b_i] * vv, den * den)
                    for w_i in range(num_w):
                        if w[w_i]:
                            R[u_i][b_idx][w_i] -= base_val * w[w_i]
            
            U_list.append(u)
            V_list.append([Fraction(x) for x in v])
            W_list.append(w)
            return True

        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            cand = pick_candidate()
            if cand is None:
                break
            a_idx, b_idx, c_idx, val = cand
            if not try_extract(a_idx, b_idx, c_idx, val, val.denominator):
                return None

        if pick_candidate() is not None:
            return None

        rank = len(U_list)
        Uf = [[0] * rank for _ in range(num_u)]
        Vf = [[0] * rank for _ in range(num_v)]
        Wf = [[0] * rank for _ in range(num_w)]
        for r in range(rank):
            for u_i in range(num_u):
                if U_list[r][u_i]:
                    Uf[u_i][r] = float(U_list[r][u_i])
            for b_i in range(num_v):
                if V_list[r][b_i]:
                    Vf[b_i][r] = float(V_list[r][b_i])
            for w_i in range(num_w):
                if W_list[r][w_i]:
                    Wf[w_i][r] = float(W_list[r][w_i])
        return {"U": Uf, "V": Vf, "W": Wf}

    # ---------- Main ----------
    # 1. Get Strassen 2x2x2
    S_U, S_V, S_W = get_strassen_2x2x2()
    
    # 2. Try to reduce 2x2x3
    best_223 = None
    best_rank_223 = 12
    for tr in range(11, 8, -1):
        if time.time() > deadline - 10: break
        res = reduce_223(tr, deadline - 10)
        if res:
            best_223 = res
            best_rank_223 = len(res["U"][0])
            break
    
    if not best_223:
        # Fallback to schoolbook
        U223, V223, W223 = get_schoolbook_2x2x3()
        U223 = [[float(x) for x in row] for row in U223]
        V223 = [[float(x) for x in row] for row in V223]
        W223 = [[float(x) for x in row] for row in W223]
        best_rank_223 = 12
    else:
        U223 = best_223["U"]
        V223 = best_223["V"]
        W223 = best_223["W"]

    # 3. Assemble 2x4x5
    # Part 1: 2x4x2 (two 2x2x2 blocks)
    # Part 2: 2x4x3 (two 2x2x3 blocks)
    
    rank_222 = 7
    rank_223 = best_rank_223
    total_rank = 2 * rank_222 + 2 * rank_223
    
    U = [[0] * total_rank for _ in range(8)] # 2x4
    V = [[0] * total_rank for _ in range(20)] # 4x5
    W = [[0] * total_rank for _ in range(10)] # 5x2
    
    col = 0
    
    # Add 2x2x2 blocks (for cols 0,1 of output)
    # Block 1: Rows 0,1 of A; Cols 0,1 of B; Cols 0,1 of C
    for r in range(rank_222):
        for i in range(4):
            if S_U[i][r]:
                U[i][col] += S_U[i][r]
            if S_V[i][r]:
                V[i][col] += S_V[i][r]
            if S_W[i][r]:
                W[i][col] += S_W[i][r]
        col += 1
        
    # Block 2: Rows 0,1 of A; Cols 2,3 of B; Cols 0,1 of C
    # Wait, A is 2x4. 
    # My 2x2x2 block uses A[0:2, 0:2] and B[0:2, 0:2] -> C[0:2, 0:2].
    # Next block uses A[0:2, 2:4] and B[2:4, 0:2] -> C[0:2, 0:2].
    # So we need to map indices carefully.
    
    # Reset col
    col = 0
    # Block 1
    for r in range(rank_222):
        for i in range(4):
            U[i][col] += S_U[i][r]
            V[i][col] += S_V[i][r]
            W[i][col] += S_W[i][r]
        col += 1
        
    # Block 2: A cols 2,3 (indices 2,3 in U), B cols 0,1 (indices 0,1 in V), C cols 0,1
    for r in range(rank_222):
        for i in range(4):
            U[i + 2][col] += S_U[i][r] # A[0:2, 2:4] -> indices 2,3
            V[i][col] += S_V[i][r] # B[0:2, 0:2] -> indices 0,1
            W[i][col] += S_W[i][r]
        col += 1
        
    # Now 2x4x3 part (Cols 2,3,4 of B; Cols 2,3 of C? No, Cols 0,1 of C? No.)
    # Output C is 2x5.
    # Cols 0,1 of C come from B cols 0,1.
    # Cols 2,3,4 of C come from B cols 2,3,4.
    # So for the 2x4x3 part, we use B cols 2,3,4 and C cols 2,3,4.
    # A is still 2x4.
    # We split A into two 2x2 blocks: A[0:2, 0:2] and A[0:2, 2:4].
    # Block 3: A[0:2, 0:2] (U 0,1), B[2:4, 2:4] (V 2,3,4,5,6,7? No, B is 4x5. Cols 2,3,4 are indices 2,3,4).
    # B indices:
    # Col 0: 0,1
    # Col 1: 2,3
    # Col 2: 4,5
    # Col 3: 6,7
    # Col 4: 8,9
    # So B cols 2,3,4 are V indices 4,5, 6,7, 8,9.
    # C indices:
    # Col 0: 0
    # Col 1: 1
    # Col 2: 2
    # Col 3: 3
    # Col 4: 4
    # So C cols 2,3,4 are W indices 2,3,4.
    
    # Block 3: A[0:2, 0:2] (U 0,1), B[2:4, 2:4] (V 4:10), C[0:2, 2:4] (W 2:5)
    for r in range(rank_223):
        for i in range(4):
            U[i][col] += U223[i][r]
            # Map V indices: i (0..5) -> 4+i
            V[4 + i][col] += V223[i][r]
            # Map W indices: i (0..5) -> 2+i
            W[2 + i][col] += W223[i][r]
        col += 1
        
    # Block 4: A[0:2, 2:4] (U 2,3), B[2:4, 2:4] (V 4:10), C[0:2, 2:4] (W 2:5)
    for r in range(rank_223):
        for i in range(4):
            U[i + 2][col] += U223[i][r]
            V[4 + i][col] += V223[i][r]
            W[2 + i][col] += W223[i][r]
        col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
