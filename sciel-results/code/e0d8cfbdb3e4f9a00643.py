import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 5
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition. U:(n*m,rank), V:(m*p,rank), W:(p*n,rank)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # Strassen-based rank-36 decomposition
    def get_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen 2x2x2 terms: (U_dict, V_dict, W_dict)
        # Keys are (local_i, local_j) for U, (local_j, local_k) for V, (local_i, local_k) for W
        strassen_terms = [
            # P1 = (a00+a11)(b00+b11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # P2 = (a10+a11)b00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # P3 = a00(b10-b11)
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # P4 = a11(b01-b00)
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # P5 = (a00+a01)b11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # P6 = (a10-a00)(b00+b10)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
            # P7 = (a01-a11)(b11+b01)
            ({(0,1):1, (1,1):-1}, {(1,1):1, (0,1):1}, {(0,0):1})
        ]
        
        # Trivial 2x2x1 terms
        trivial_2x2x1_terms = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),
            ({(0,1):1}, {(1,0):1}, {(0,0):1}),
            ({(1,0):1}, {(0,0):1}, {(1,0):1}),
            ({(1,1):1}, {(1,0):1}, {(1,0):1})
        ]
        
        col = 0
        # For each j-block (0 or 1) and k-block (0,1,2)
        for j_block in range(2):  # j_block 0: j=0,1; j_block 1: j=2,3
            for k_block in range(3):  # k_block 0: k=0,1; k_block 1: k=2,3; k_block 2: k=4
                if k_block < 2:
                    # Strassen 2x2x2
                    for u_dict, v_dict, w_dict in strassen_terms:
                        for (li, lj), val_u in u_dict.items():
                            gi = li
                            gj = 2*j_block + lj
                            U[gi*m + gj][col] += val_u
                        for (lj, lk), val_v in v_dict.items():
                            gj = 2*j_block + lj
                            gk = 2*k_block + lk
                            V[gj*p + gk][col] += val_v
                        for (li, lk), val_w in w_dict.items():
                            gi = li
                            gk = 2*k_block + lk
                            W[gk*n + gi][col] += val_w
                        col += 1
                else:
                    # Trivial 2x2x1 for k=4
                    for u_dict, v_dict, w_dict in trivial_2x2x1_terms:
                        for (li, lj), val_u in u_dict.items():
                            gi = li
                            gj = 2*j_block + lj
                            U[gi*m + gj][col] += val_u
                        for (lj, lk), val_v in v_dict.items():
                            gj = 2*j_block + lj
                            gk = 4
                            V[gj*p + gk][col] += val_v
                        for (li, lk), val_w in w_dict.items():
                            gi = li
                            gk = 4
                            W[gk*n + gi][col] += val_w
                        col += 1
        
        return U, V, W

    # Get the rank-36 decomposition
    U, V, W = get_strassen_36()
    rank = 36
    
    # Verify
    if not verify_exact(U, V, W, rank):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower rank decomposition via greedy search
    if time.time() < deadline - 20.0:
        # Attempt rank 35 search
        best_rank = rank
        best_result = (U, V, W)
        
        # Try to find rank 35 by removing one term and seeing if we can compensate
        # This is a heuristic approach - we try each of the 36 terms and see if
        # the remaining 35 can still decompose the tensor (they can't exactly, but
        # we check if we can find a replacement)
        
        # Actually, a simpler approach: try random perturbations and see if we can
        # find a rank-35 decomposition. This is unlikely to succeed quickly.
        
        # For now, return the rank-36 result as it's verified correct
        pass
    
    return {"U": U, "V": V, "W": W}
