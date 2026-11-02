import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Strassen-based decomposition: partition p=12 columns into 6 pairs
    # Each pair uses 7 rank-1 terms (Strassen for 2x2x2 matrix mult)
    # Total rank = 6 * 7 = 42

    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42

    # Initialize factor matrices with NumPy
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # Local A indices: 0=a, 1=b, 2=c, 3=d
    # Local B indices: 0=e, 1=f, 2=g, 3=h
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11

    # Define Strassen coefficients as lists of (index, value) for U, V, W
    # Format: [ [u_indices, u_vals], [v_indices, v_vals], [w_indices, w_vals] ]
    strassen_terms = [
        # M1
        ([0, 3], [1, 1]), ([0, 3], [1, 1]), ([], []), # Placeholder, W is filled separately
        # Actually, let's just define the coefficients for each term s in 0..6
    ]
    
    # Correct Strassen terms definition:
    # Term 0: M1. U: a+d -> {0:1, 3:1}. V: e+h -> {0:1, 3:1}. W: C00+1, C11-1 -> {0:1, 3:-1}
    # Term 1: M2. U: c+d -> {2:1, 3:1}. V: e -> {0:1}. W: C10+1, C11-1 -> {2:1, 3:-1}
    # Term 2: M3. U: a -> {0:1}. V: f-h -> {1:1, 3:-1}. W: C01+1, C11+1 -> {1:1, 3:1}
    # Term 3: M4. U: d -> {3:1}. V: g-e -> {2:1, 0:-1}. W: C00+1, C10+1 -> {0:1, 2:1}
    # Term 4: M5. U: a+b -> {0:1, 1:1}. V: h -> {3:1}. W: C00-1, C01+1 -> {0:-1, 1:1}
    # Term 5: M6. U: c-a -> {2:1, 0:-1}. V: e+f -> {0:1, 1:1}. W: C11+1 -> {3:1}
    # Term 6: M7. U: b-d -> {1:1, 3:-1}. V: g+h -> {2:1, 3:1}. W: C00+1 -> {0:1}

    terms_data = [
        # (u_idx, u_val), (v_idx, v_val), (w_idx, w_val)
        (([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, -1])),
        (([2, 3], [1, 1]), ([0], [1]), ([2, 3], [1, -1])),
        (([0], [1]), ([1, 3], [1, -1]), ([1, 3], [1, 1])),
        (([3], [1]), ([2, 0], [1, -1]), ([0, 2], [1, 1])),
        (([0, 1], [1, 1]), ([3], [1]), ([0, 1], [-1, 1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1]), ([3], [1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1]), ([0], [1]))
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_idx, u_val = terms_data[s][0]
            v_idx, v_val = terms_data[s][1]
            w_idx, w_val = terms_data[s][2]

            # Set U
            for idx, val in zip(u_idx, u_val):
                U[idx, col] = val

            # Set V
            for idx, val in zip(v_idx, v_val):
                if idx == 0:
                    global_b = 2 * t
                elif idx == 1:
                    global_b = 2 * t + 1
                elif idx == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b, col] = val

            # Set W
            for idx, val in zip(w_idx, w_val):
                if idx == 0:
                    global_w = (2 * t) * n + 0
                elif idx == 1:
                    global_w = (2 * t + 1) * n + 0
                elif idx == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w, col] = val

    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))

    if max_err > 1e-10:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, column] = 1
                    V_fb[j*p+k, column] = 1
                    W_fb[k*n+i, column] = 1
                    column += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
