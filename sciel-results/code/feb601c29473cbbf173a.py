import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 8
    
    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U_frac[i * m + a]
                    v_row = V_frac[a * p + k]
                    w_row = W_frac[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition (Rank 7)
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    S_W = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]

    # Construct Rank 32 candidate: 4 blocks of Rank 8 (7 Strassen + 1 Rank-1)
    rank_total = 32
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        
        # Part 1: Strassen for A_left (2x2) * B_top (2x2)
        # Indices: i in 0..1, a in 0..1, k in k0..k0+1
        for r in range(7):
            # U: i, a (local a 0,1)
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            
            # V: a, k (local a 0,1; local k 0,1)
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            
            # W: k, i (local k 0,1; local i 0,1)
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            
            col += 1

        # Part 2: Rank 1 for A_right (2x1) * B_bottom (1x2)
        # A_right: i in 0..1, a=2
        # B_bottom: a=2, k in k0..k0+1
        # C: i in 0..1, k in k0..k0+1
        
        # U: U[0,2]=1, U[1,2]=1
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = 1
        
        # V: V[2, k0]=1, V[2, k0+1]=1
        V[2 * p + k0][col] = 1
        V[2 * p + (k0 + 1)][col] = 1
        
        # W: W[k0, 0]=1, W[k0+1, 0]=1, W[k0, 1]=1, W[k0+1, 1]=1
        W[k0 * n + 0][col] = 1
        W[(k0 + 1) * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        
        col += 1

    candidate_32 = {"U": U, "V": V, "W": W}
    
    if _validate(candidate_32):
        return candidate_32
    else:
        return _schoolbook()
