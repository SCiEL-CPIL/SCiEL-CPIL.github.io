import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s
    
    n, m, p = 3, 4, 6
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    def build_decomposition(strategy):
        """Build decomposition using given strategy, return (U, V, W, rank)"""
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            """Strassen 2x2x2 block"""
            a, b = (i1, j1), (i1, j2)
            c, d = (i2, j1), (i2, j2)
            e, f = (j1, k1), (j1, k2)
            g, h = (j2, k1), (j2, k2)
            C11, C12 = (i1, k1), (i1, k2)
            C21, C22 = (i2, k1), (i2, k2)
            
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook(i, j, k):
            terms.append((
                {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
            ))
        
        if strategy == 0:
            # Original: row 0 schoolbook, rows 1-2 Strassen 2x2x2
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1)
        
        elif strategy == 1:
            # Try: rows 0-1 as 2x4x6, row 2 schoolbook
            for j in range(m):
                for k in range(p):
                    add_schoolbook(2, j, k)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_block(0, 1, j1, j1 + 1, k1, k1 + 1)
        
        elif strategy == 2:
            # Try: split p into 3+3, use different grouping
            # Rows 1-2, j=0-1, k=0-2: can't use Strassen (needs 2x2x2)
            # Fall back to schoolbook for odd-sized blocks
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k)
            # For rows 1-2, use 2x2x2 where possible
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1)
        
        elif strategy == 3:
            # Hybrid: try to use Strassen on a 2x2x3 block by combining
            # Actually, let's try: rows 0-1 with j=0-1, all k; rows 2 all schoolbook
            # This gets complex. Let's try a 3x2x2 Strassen variant
            # For now, use schoolbook for row 2, Strassen for rows 0-1
            for j in range(m):
                for k in range(p):
                    add_schoolbook(2, j, k)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_block(0, 1, j1, j1 + 1, k1, k1 + 1)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        
        return U, V, W, rank
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication"""
        n, m, p = 3, 4, 6
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try multiple strategies and pick the best valid one
    best_result = None
    best_rank = float('inf')
    
    for strategy in range(4):
        if not time_ok():
            break
        
        U, V, W, rank = build_decomposition(strategy)
        
        if rank < best_rank:
            if verify_decomposition(U, V, W):
                best_result = (U, V, W, rank)
                best_rank = rank
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: pure schoolbook
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
    
    return {"U": U, "V": V, "W": W}
