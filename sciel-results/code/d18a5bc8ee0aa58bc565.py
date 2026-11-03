"""Combined program: structured Strassen decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    terms = []  # Each term: (u_dict, v_dict, w_dict)
    
    def _check_time():
        if time.time() - _START_TIME > _TIME_LIMIT:
            return True
        return False

    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    # Part 1: A[0:1, :] * B[:, :]. Rank 18.
    # A is 3x4, B is 4x6.
    # A[0, j] * B[j, k] -> C[0, k]
    for j in range(4):
        for k in range(6):
            add_term(
                {(0, j): 1},
                {(j, k): 1},
                {(0, k): 1}
            )

    # Part 2: A[1:3, :] * B[:, :]. Rank 46.
    # Partition m=4 into 1+3.
    # A[1:3, 0:1] * B[0:1, :] -> Rank 6 (2x1x6 -> 2*6=12? No. 2x1x6 is 2 rows, 1 col, 6 cols.
    # C[1:3, k] = sum_j A[1:3, j] B[j, k].
    # If j=0 only: C[1,k] = A[1,0]B[0,k], C[2,k] = A[2,0]B[0,k].
    # This is 2 * 6 = 12 terms.
    
    # Wait, my previous analysis:
    # R(3,1,6) = 18.
    # R(3,3,6) = 46.
    # Total 64.
    
    # Let's implement R(3,1,6) = 18.
    # A[0:3, 0:1] * B[0:1, 0:6].
    # A is 3x1. B is 1x6.
    # C[i, k] = A[i, 0] * B[0, k].
    for i in range(3):
        for k in range(6):
            add_term(
                {(i, 0): 1},
                {(0, k): 1},
                {(i, k): 1}
            )
            
    # Now R(3,3,6) = 46.
    # A[0:3, 1:4] * B[1:4, 0:6].
    # Let A2 = A[:, 1:4] (3x3), B2 = B[1:4, :] (3x6).
    # Partition p=6 into 3+3.
    # B2 = [B2a, B2b].
    # C = A2 B2a + A2 B2b.
    # Each is 3x3x3. Rank 23.
    
    def add_bclr_3x3x3(i_offset, j_offset, k_offset):
        """Add BCLR 3x3x3 decomposition.
        i_offset: offset for A rows (0..2)
        j_offset: offset for A cols / B rows (0..2)
        k_offset: offset for B cols / C cols (0..2)
        """
        # BCLR 23 terms
        # u, v, w are in R3.
        # Indices are 0, 1, 2.
        
        # The 23 terms of BCLR decomposition
        # Reference: Bini et al. 1979.
        # I will use a known set of 23 terms.
        
        # Since I cannot guarantee the exact BCLR coefficients from memory,
        # and hardcoding 23 complex terms is risky,
        # I will fall back to a safer decomposition for 3x3x3 if possible?
        # No, 23 is the only way to get 64.
        
        # I will use the following 23 terms (standard basis vectors with coeffs):
        # This is a simplified representation.
        # If this fails, the score will be 0.
        
        # Actually, let's use the 26 rank decomposition for 3x3x3 if I can't get 23?
        # 26 * 2 + 18 = 70. Worse than 66.
        
        # I will stick to the 66 rank solution to be safe.
        # The 66 rank solution is:
        # Row 0: 24 terms.
        # Rows 1-2: 42 terms (6 blocks of 2x2x2 Strassen).
        
        # I will RE-IMPLEMENT the 66 rank solution cleanly.
        pass

    # I will revert to the 66 rank solution logic which is known to work.
    terms.clear()
    
    # Part 1: Row 0 of A, schoolbook (24 terms)
    for j in range(4):
        for k in range(6):
            add_term(
                {(0, j): 1},
                {(j, k): 1},
                {(0, k): 1}
            )

    # Part 2: Rows 1-2 of A, using Strassen for 2x2 blocks
    # A[1:3, 0:2] * B[0:2, k1:k2]
    # A[1:3, 2:4] * B[2:4, k1:k2]
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a = (i1, j1)
        b = (i1, j2)
        c = (i2, j1)
        d = (i2, j2)
        e = (j1, k1)
        f = (j1, k2)
        g = (j2, k1)
        h = (j2, k2)

        C11 = (i1, k1)
        C12 = (i1, k2)
        C21 = (i2, k1)
        C22 = (i2, k2)

        add_term({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1})
        add_term({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1})
        add_term({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1})
        add_term({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1})
        add_term({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1})
        add_term({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1})
        add_term({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1})

    for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
        add_strassen_block(1, 2, 0, 1, k1, k2)
        add_strassen_block(1, 2, 2, 3, k1, k2)

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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
