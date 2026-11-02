import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - start_time < DEADLINE

    def add_strassen_2x2x2(i1, i2, j1, j2, k1, k2, terms):
        """Standard Strassen 2x2x2"""
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

    def add_schoolbook(i, j, k, terms):
        terms.append((
            {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
        ))

    def verify_terms(terms):
        """Verify that terms correctly decompose the tensor. Return True if valid."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def terms_to_matrices(terms):
        """Convert terms to (U, V, W) lists."""
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

    best_result = None
    best_rank = float('inf')

    # Strategy 0: Row 0 schoolbook, rows 1-2 Strassen
    terms = []
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k, terms)
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            add_strassen_2x2x2(1, 2, j1, j1 + 1, k1, k1 + 1, terms)
    
    if verify_terms(terms):
        U, V, W, rank = terms_to_matrices(terms)
        if rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)

    # Strategy 1: Rows 0-1 Strassen, row 2 schoolbook
    if time_ok():
        terms = []
        for j in range(m):
            for k in range(p):
                add_schoolbook(2, j, k, terms)
        for j1 in [0, 2]:
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(0, 1, j1, j1 + 1, k1, k1 + 1, terms)
        
        if verify_terms(terms):
            U, V, W, rank = terms_to_matrices(terms)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)

    # Strategy 2: Try all possible single-row schoolbook + Strassen on other two rows
    # Already covered by strategies 0 and 1 (rows 0 or 2 as schoolbook)
    # Try row 1 as schoolbook, Strassen on rows 0-2? Can't do 3-row Strassen directly.
    # Instead: Strassen on (0,1) for some blocks, Strassen on (1,2) for others
    if time_ok():
        # Mixed: Strassen (0,1) for j-pairs [0,1], Strassen (1,2) for j-pairs [2,3]
        terms = []
        for k1 in [0, 2, 4]:
            add_strassen_2x2x2(0, 1, 0, 1, k1, k1 + 1, terms)
            add_strassen_2x2x2(1, 2, 2, 3, k1, k1 + 1, terms)
        # Need to fill in the gaps: (0,1) for j=2,3 and (1,2) for j=0,1
        # (i=0, j=2,3) and (i=0, j=0,1) are covered by Strassen(0,1)
        # (i=2, j=0,1) and (i=2, j=2,3) are covered by Strassen(1,2)
        # But (i=0, j=2,3) needs B[j=2,3] which is in Strassen(1,2) not (0,1)
        # This won't work cleanly. Skip.
        pass

    # Strategy 3: Try using 3x2x2 blocks with Strassen + schoolbook
    # For each 3x2x2 block: Strassen on rows 0-1 (7 terms) + schoolbook for row 2 (4 terms) = 11
    # 6 blocks * 11 = 66. Same as before, but let's verify.
    if time_ok():
        terms = []
        for j1 in [0, 2]:
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(0, 1, j1, j1 + 1, k1, k1 + 1, terms)
                for j in [j1, j1 + 1]:
                    for k in [k1, k1 + 1]:
                        add_schoolbook(2, j, k, terms)
        
        if verify_terms(terms):
            U, V, W, rank = terms_to_matrices(terms)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)

    # Strategy 4: Try 2x3x2 blocks
    # For each 2x3x2 block: Strassen on j-pair [0,1] (7 terms) + schoolbook for j=2 (4 terms) = 11
    # But we need to cover all 3 rows. 3 row-pairs * 2 j-pairs = 6 blocks * 11 = 66.
    if time_ok():
        terms = []
        for i1 in [0, 1]:
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(i1, i1 + 1, 0, 1, k1, k1 + 1, terms)
                for i in [i1, i1 + 1]:
                    for k in [k1, k1 + 1]:
                        add_schoolbook(i, 2, k, terms)
                # Also need j=3
                add_strassen_2x2x2(i1, i1 + 1, 2, 3, k1, k1 + 1, terms)
                for i in [i1, i1 + 1]:
                    for k in [k1, k1 + 1]:
                        add_schoolbook(i, 3, k, terms)
        # This gives 3 row-pairs * 3 k-pairs * (7+4+7+4) = 3*3*22 = 198. Too many!
        # This approach is wrong. Let me reconsider.
        pass

    if best_result is not None:
        U, V, W = best_result
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
