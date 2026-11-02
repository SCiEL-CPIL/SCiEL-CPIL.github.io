import numpy as np
import time
from scipy.optimize import minimize

def verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition identity exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    r = U_arr.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U_arr[a * m + b] * V_arr[c * p + d] * W_arr[k * n + i])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 6

    # Schoolbook fallback (rank 36)
    def schoolbook():
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Try the rank-22 approach: two copies of a rank-11 2x3x3 decomposition
    # This is known to work and gives score 1.059
    def rank22_decomposition():
        rank = 22
        U = np.zeros((6, rank))
        V = np.zeros((18, rank))
        W = np.zeros((12, rank))

        # Rank-11 decomposition for 2x3x3
        # A: 2x3 (indices 0..5), B: 3x3 (indices 0..8), C: 2x3 (indices 0..5)
        # Using Strassen-like structure with 7 multiplications for 2x2 block
        # plus 4 additional for the third column/row

        # 2x2x2 Strassen (7 terms) for A(0,1)x(0,1), B(0,1)x(0,1) -> C(0,1)x(0,1)
        # U indices: A00=0, A01=1, A10=2, A11=3
        # V indices: B00=0, B01=1, B10=2, B11=3
        # W indices: C00=0, C10=1, C01=2, C11=3

        # P1: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # P2: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1

        # P3: A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        # P4: A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        # P5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # P6: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # P7: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1

        # Column 2 of B: A(0,1,2) x B col 2 -> C col 2 (4 terms)
        # Q1: A00*B02
        U[0, 7] = 1
        V[4, 7] = 1
        W[4, 7] = 1

        # Q2: A01*B12
        U[1, 8] = 1
        V[5, 8] = 1
        W[4, 8] = 1

        # Q3: A10*B02
        U[2, 9] = 1
        V[4, 9] = 1
        W[5, 9] = 1

        # Q4: A11*B12
        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1

        # Row 2 of B: A(0,1,2) x B row 2 -> C(0,1) (4 terms)
        # R1: A02*B20
        U[4, 11] = 1
        V[6, 11] = 1
        W[0, 11] = 1

        # R2: A02*B21
        U[4, 12] = 1
        V[7, 12] = 1
        W[2, 12] = 1

        # R3: A12*B20
        U[5, 13] = 1
        V[6, 13] = 1
        W[1, 13] = 1

        # R4: A12*B21
        U[5, 14] = 1
        V[7, 14] = 1
        W[3, 14] = 1

        # A02*B22
        U[4, 15] = 1
        V[8, 15] = 1
        W[4, 15] = 1

        # A12*B22
        U[5, 16] = 1
        V[8, 16] = 1
        W[5, 16] = 1

        # That's 17 terms for 2x3x3, not 11. Let me use the known rank-11.
        # Actually, let me just use the two-copy approach with rank-17 2x3x3 = rank-34
        # But the cell finding says rank-11 works. Let me try a different construction.

        # For 2x3x3, the rank is at most 12 (from 2x2x3 rank 6 doubled minus overlaps)
        # Let me use a known rank-12 decomposition for 2x3x3

        # Actually, let me just return the rank-34 (two copies of rank-17) which is verified
        return None  # Skip this, use verified approaches

    # Verified rank-34: two copies of rank-17 2x3x3
    def rank34_decomposition():
        rank = 34
        U = np.zeros((6, rank))
        V = np.zeros((18, rank))
        W = np.zeros((12, rank))

        # Rank-17 decomposition for 2x3x3
        # A: 2x3 (indices 0..5), B: 3x3 (indices 0..8), C: 2x3 (indices 0..5)
        # C index: k*2+i where k=col, i=row

        # 2x2x2 Strassen (7 terms) for A(0,1)x(0,1), B(0,1)x(0,1) -> C(0,1)x(0,1)
        # U: A00=0, A01=1, A10=2, A11=3
        # V: B00=0, B01=1, B10=2, B11=3
        # W: C00=0, C10=1, C01=2, C11=3

        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1

        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1

        # Column 2 of B: A(0,1,2) x B col 2 -> C col 2
        U[0, 7] = 1
        V[4, 7] = 1
        W[4, 7] = 1

        U[1, 8] = 1
        V[5, 8] = 1
        W[4, 8] = 1

        U[2, 9] = 1
        V[4, 9] = 1
        W[5, 9] = 1

        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1

        # Row 2 of B: A(0,1,2) x B row 2 -> C(0,1)
        U[4, 11] = 1
        V[6, 11] = 1
        W[0, 11] = 1

        U[4, 12] = 1
        V[7, 12] = 1
        W[2, 12] = 1

        U[5, 13] = 1
        V[6, 13] = 1
        W[1, 13] = 1

        U[5, 14] = 1
        V[7, 14] = 1
        W[3, 14] = 1

        U[4, 15] = 1
        V[8, 15] = 1
        W[4, 15] = 1

        U[5, 16] = 1
        V[8, 16] = 1
        W[5, 16] = 1

        # Verify 2x3x3
        if verify_decomposition(U[:, :17], V[:, :17], W[:, :17], 2, 3, 3):
            # Combine two 2x3x3 blocks for 2x3x6
            U_comb = np.zeros((6, 34))
            U_comb[:, :17] = U[:, :17]
            U_comb[:, 17:] = U[:, :17]

            V_comb = np.zeros((18, 34))
            for j in range(3):
                for k in range(3):
                    V_comb[j * 6 + k, :17] = V[j * 3 + k, :17]
                    V_comb[j * 6 + 3 + k, 17:] = V[j * 3 + k, :17]

            W_comb = np.zeros((12, 34))
            for k in range(3):
                for i in range(2):
                    W_comb[k * 2 + i, :17] = W[k * 2 + i, :17]
                    W_comb[(k + 3) * 2 + i, 17:] = W[k * 2 + i, :17]

            if verify_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
                return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}

        return None

    # Try rank-34 first (verified approach)
    result = rank34_decomposition()
    if result is not None:
        return result

    if time.time() > deadline - 10:
        return schoolbook()

    # Now try a structured search for lower rank
    # Ansatz: For 2x3x6, split B into two 3x3 blocks
    # C[:, 0:3] = A @ B[:, 0:3]
    # C[:, 3:6] = A @ B[:, 3:6]
    # Each 2x3x3 has rank at most 17, so total 34.
    # Can we do better by sharing structure between the two blocks?
    # If we use the same U factors for both blocks, we need V and W to differ.
    # This is already the rank-34 construction.

    # Alternative: use a different split. Split A into 1x3 and 1x3 (trivial, rank 36)
    # Or use a 2x2x3 Strassen for each half.

    # Let's try: 2x2x3 has rank 6 (Strassen for 2x2x2 + 1 extra for col 3)
    # Actually 2x2x2 rank 7, 2x2x3 rank 7+2=9? No.
    # 2x2x3: A(2x2) @ B(2x3). Rank is at most 2*2*3=12, but with Strassen on 2x2x2 part:
    # C[:, 0:2] uses 7 mults, C[:, 2] uses 4 mults (A00*B02, A01*B12, A10*B02, A11*B12)
    # Total: 11. So 2x2x3 rank 11.

    # For 2x3x3: A(2x3) @ B(3x3).
    # Split A into A[:, 0:2] (2x2) and A[:, 2] (2x1)
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, :] (2x2x3, rank 11)
    # C[:, 2] = A[:, 2] @ B[:, 2] (2x1x3, rank 6)
    # But B[0:2, :] and B[:, 2] don't cover all of B.
    # Actually: C = A @ B. C[:, k] = A @ B[:, k].
    # C[:, 0] = A[:, 0:2] @ B[0:2, 0] + A[:, 2] @ B[2, 0]
    # This is getting complicated.

    # Let me try a direct Nelder-Mead search for rank 28
    def try_rank(r):
        """Try to find a rank-r decomposition using Nelder-Mead."""
        # Initialize with schoolbook-like structure
        U_init = np.zeros((6, r))
        V_init = np.zeros((18, r))
        W_init = np.zeros((12, r))

        # Fill with first r schoolbook terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U_init[i * m + j, col] = 1
                        V_init[j * p + k, col] = 1
                        W_init[k * n + i, col] = 1
                        col += 1

        # Flatten
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        def unflatten(x):
            U = x[:6 * r].reshape(6, r)
            V = x[6 * r:6 * r + 18 * r].reshape(18, r)
            W = x[6 * r + 18 * r:].reshape(12, r)
            return U, V, W

        def objective(x):
            U, V, W = unflatten(x)
            # Compute tensor error
            error = 0.0
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            for i in range(n):
                                for k in range(p):
                                    target = 1.0 if (a == i and c == k and b == d) else 0.0
                                    s = np.sum(U[a * m + b] * V[c * p + d] * W[k * n + i])
                                    error += (s - target) ** 2
            return error

        x0 = flatten(U_init, V_init, W_init)
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        U_opt, V_opt, W_opt = unflatten(res.x)

        # Round to nearest half-integer
        U_round = np.round(2 * U_opt) / 2
        V_round = np.round(2 * V_opt) / 2
        W_round = np.round(2 * W_opt) / 2

        if verify_decomposition(U_round, V_round, W_round, n, m, p):
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}

        # Try rounding to integer
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        if verify_decomposition(U_int, V_int, W_int, n, m, p):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

        return None

    # Try ranks from 28 down to 22
    for r in range(28, 21, -1):
        if time.time() > deadline - 15:
            break
        result = try_rank(r)
        if result is not None:
            return result

    # Final fallback
    return schoolbook()
