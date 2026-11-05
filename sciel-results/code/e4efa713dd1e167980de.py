"""Matrix multiplication 3x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Reshape for efficient computation
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need to check that for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Create index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        # Get the relevant entries
        u_vals = U_arr[i_idx * m + j_idx, :]  # (n*m*p, rank)
        v_vals = V_arr[j_idx * p + k_idx, :]  # (n*m*p, rank)
        w_vals = W_arr[k_idx * n + i_idx, :]  # (n*m*p, rank)
        
        # Compute the sum for each (i,j,k)
        result = np.sum(u_vals * v_vals * w_vals, axis=1)
        
        return np.all(np.abs(result - 1.0) < 1e-9)

    def strassen_terms():
        """
        Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is a tuple (U_dict, V_dict, W_dict).
        Keys are local indices (0,1).
        """
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # M2 = (A10+A11)B00
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        # M3 = A00(B01-B11)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ))
        # M4 = A11(B10-B00)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        # M5 = (A00+A01)B11
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ))
        # M6 = (A10-A00)(B00+B01)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # M7 = (A01-A11)(B10+B11)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        return terms

    def build_block_decomposition():
        """
        Build the decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
        """
        # Define groups
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        strassen_terms_list = strassen_terms()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
                for u_pat, v_pat, w_pat in strassen_terms_list:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W}

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 90."""
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

    # Try the block decomposition first
    if time.time() < deadline:
        result = build_block_decomposition()
        rank = len(result["U"][0])
        if verify(result["U"], result["V"], result["W"], rank):
            return result

    # Fallback to schoolbook
    result = schoolbook()
    return result

# EVOLVE-BLOCK-END
