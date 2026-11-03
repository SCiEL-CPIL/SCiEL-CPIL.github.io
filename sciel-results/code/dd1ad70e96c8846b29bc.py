import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin before 150s kill

    n, m, p = 3, 4, 7

    # Try the structured decomposition that reduces rank below schoolbook (84).
    # Based on the cell finding: splitting n=3 into 2+1 allows Strassen on
    # 2x2x2 blocks plus schoolbook for 1x2x2 or 2x2x1 blocks.
    #
    # Strategy: For the 3x4x7 multiplication, decompose as:
    #   C[0:2, :] = A[0:2, :] @ B[ :, :7]
    #   C[2, :]    = A[2, :]    @ B[ :, :7]
    #
    # For C[0:2, :]: use 2x4x7 = sum of 4 blocks of 2x1x7... but that doesn't help.
    # Better: use 2x4x7 = (2x2x7) * 2, where each 2x2x7 can use Strassen-style.
    #
    # Actually, let's use the known fact that 2x2x2 can be done in rank 7 (Strassen).
    # And 2x2x1 (or 2x1x2) can be done in rank 2 (or even 1 with the right structure).
    #
    # For 2x2x7: decompose the 7 columns into 3 pairs + 1 single.
    #   3 * rank(2x2x2) + 1 * rank(2x2x1) = 3*7 + 1*2 = 21 + 2 = 23
    # For 2x4x7: two such blocks = 2*23 = 46
    # For 1x4x7: schoolbook = 28
    # Total: 46 + 28 = 74
    #
    # But the cell finding said 78. Let me re-examine.
    # Perhaps 2x2x1 needs rank 1? Let's check:
    #   C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0}
    #   C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    #   With rank 1: U[i,j]*V[j,0]*W[i,0] = 1 for all i,j
    #   Set U=1, W=1, V[j,0]=1 -> sum over j gives 2, not 1.
    #   So rank 1 doesn't work for 2x2x1. Rank 2 works (schoolbook).
    #
    # Alternative: 2x2x1 with rank 2, so 3*7 + 2 = 23 per 2x2x7, total 46+28=74.
    # The cell finding said 78, maybe they used a different split.
    # Let me try the 74-rank construction.

    # Actually, let me just try to find any rank < 84 using a simple approach.
    # The cell finding measured 78 -> score 1.077. Let me aim for that or better.

    # Let me implement the 2x2x2 Strassen decomposition properly.
    # Strassen 2x2x2: C = A*B where A,B,C are 2x2.
    # 7 multiplications M1..M7, then combinations.

    # Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For tensor decomposition, each "multiplication" becomes a rank-1 term.
    # The U, V, W matrices encode the linear combinations.

    def add_strassen_block(U, V, W, n_off, m_off, p_off, r_start):
        """Add Strassen 2x2x2 block at offset (n_off, m_off, p_off) starting at rank r_start.
        Returns the next free rank index."""
        r = r_start
        # For each of the 7 Strassen multiplications, add a rank-1 term.
        # M1: (A00+A11)*(B00+B11)
        # U: A00+A11 -> U[n_off+m_off, r]=1, U[n_off+1+m_off, r]=1  (i=0,j=0 and i=1,j=1... wait)
        # Actually U is indexed by (i*m + j) for the A matrix.
        # For 2x2x2 at offset (n_off, m_off, p_off):
        #   A indices: (n_off+i, m_off+j) for i,j in {0,1}
        #   B indices: (m_off+j, p_off+k) for j,k in {0,1}
        #   C indices: (p_off+k, n_off+i) for k,i in {0,1}  (W convention: W[k*n+i, r])

        # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[n_off*1 + m_off, r] += 1       # A[0,0]
        U[(n_off+1)*1 + m_off+1, r] += 1 # A[1,1]
        V[(m_off)*1 + p_off, r] += 1     # B[0,0]
        V[(m_off+1)*1 + p_off+1, r] += 1 # B[1,1]
        # C00 += M1
        W[(p_off)*1 + n_off, r] += 1
        r += 1

        # M2: (A[1,0]+A[1,1]) * B[0,0]
        U[(n_off+1)*1 + m_off, r] += 1       # A[1,0]
        U[(n_off+1)*1 + m_off+1, r] += 1     # A[1,1]
        V[(m_off)*1 + p_off, r] += 1         # B[0,0]
        # C10 += M2
        W[(p_off)*1 + (n_off+1), r] += 1
        r += 1

        # M3: A[0,0] * (B[0,1]-B[1,1])
        U[n_off*1 + m_off, r] += 1           # A[0,0]
        V[(m_off)*1 + (p_off+1), r] += 1     # B[0,1]
        V[(m_off+1)*1 + (p_off+1), r] -= 1   # -B[1,1]
        # C01 += M3
        W[(p_off+1)*1 + n_off, r] += 1
        r += 1

        # M4: A[1,1] * (B[1,0]-B[0,0])
        U[(n_off+1)*1 + m_off+1, r] += 1     # A[1,1]
        V[(m_off+1)*1 + p_off, r] += 1       # B[1,0]
        V[(m_off)*1 + p_off, r] -= 1         # -B[0,0]
        # C00 += M4, C10 += M4
        W[(p_off)*1 + n_off, r] += 1
        W[(p_off)*1 + (n_off+1), r] += 1
        r += 1

        # M5: (A[0,0]+A[0,1]) * B[1,1]
        U[n_off*1 + m_off, r] += 1           # A[0,0]
        U[n_off*1 + m_off+1, r] += 1         # A[0,1]
        V[(m_off+1)*1 + (p_off+1), r] += 1   # B[1,1]
        # C01 += M5, C00 -= M5
        W[(p_off+1)*1 + n_off, r] += 1
        W[(p_off)*1 + n_off, r] -= 1
        r += 1

        # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
        U[(n_off+1)*1 + m_off, r] += 1       # A[1,0]
        U[n_off*1 + m_off, r] -= 1           # -A[0,0]
        V[(m_off)*1 + p_off, r] += 1         # B[0,0]
        V[(m_off+1)*1 + p_off, r] += 1       # B[1,0]
        # C11 += M6
        W[(p_off+1)*1 + (n_off+1), r] += 1
        r += 1

        # M7: (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
        U[n_off*1 + m_off+1, r] += 1         # A[0,1]
        U[(n_off+1)*1 + m_off+1, r] -= 1     # -A[1,1]
        V[(m_off)*1 + (p_off+1), r] += 1     # B[0,1]
        V[(m_off+1)*1 + (p_off+1), r] += 1   # B[1,1]
        # C00 += M7, C11 += M7
        W[(p_off)*1 + n_off, r] += 1
        W[(p_off+1)*1 + (n_off+1), r] += 1
        r += 1

        return r

    def add_2x2x1_block(U, V, W, n_off, m_off, p_off, r_start):
        """Add 2x2x1 schoolbook block (rank 2)."""
        r = r_start
        for j in range(2):
            for i in range(2):
                pass
        # Actually 2x2x1 schoolbook has rank 4? No, 2*2*1 = 4.
        # But we can do it in rank 2:
        # C[i,0] = sum_j A[i,j] B[j,0]
        # Term 1: A[i,0]*B[0,0] for all i -> rank 1? No, because i varies.
        # Let me just use schoolbook: rank = 2*2*1 = 4.
        for j in range(2):
            for i in range(2):
                U[(n_off+i)*1 + (m_off+j), r] = 1
                V[(m_off+j)*1 + (p_off+0), r] = 1
                W[(p_off+0)*1 + (n_off+i), r] = 1
                r += 1
        return r

    def add_1x2x2_block(U, V, W, n_off, m_off, p_off, r_start):
        """Add 1x2x2 schoolbook block (rank 4)."""
        r = r_start
        for j in range(2):
            for k in range(2):
                U[(n_off+0)*1 + (m_off+j), r] = 1
                V[(m_off+j)*1 + (p_off+k), r] = 1
                W[(p_off+k)*1 + (n_off+0), r] = 1
                r += 1
        return r

    # Build the decomposition:
    # n=3, m=4, p=7
    # Split n into 2+1: rows 0,1 and row 2.
    # For the 2x4x7 part: split m=4 into 2+2, p=7 into 2+2+2+1
    #   2x2x2 blocks: 2*3 = 6 blocks, each rank 7 -> 42
    #   2x2x1 block: 1 block, rank 4 -> 4
    #   Total for 2x4x7: 46
    # For the 1x4x7 part: split m=4 into 2+2, p=7 into 2+2+2+1
    #   1x2x2 blocks: 2*3 = 6 blocks, each rank 4 -> 24
    #   1x2x1 block: 1 block, rank 2 -> 2
    #   Total for 1x4x7: 26
    # Grand total: 46 + 26 = 72

    # Let's try rank 72.
    rank = 72
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    r = 0
    # 2x4x7 part: n_off=0, m=4, p=7
    # Split m into [0,2), [2,4), p into [0,2), [2,4), [4,6), [6,7)
    for m_start in [0, 2]:
        for p_start in [0, 2, 4]:
            if p_start < 6:
                # 2x2x2 block
                r = add_strassen_block(U, V, W, 0, m_start, p_start, r)
            else:
                # 2x2x1 block
                r = add_2x2x1_block(U, V, W, 0, m_start, p_start, r)
        # Wait, p=7, so p_start=6 gives p range [6,7) which is 1 column.
        # But I only have p_start in [0,2,4,6]. Let me fix.
    # Let me redo this more carefully.
    r = 0
    # 2x4x7: n_off=0
    for m_start in [0, 2]:
        for p_start in [0, 2, 4, 6]:
            p_end = min(p_start + 2, p)
            if p_end - p_start == 2:
                r = add_strassen_block(U, V, W, 0, m_start, p_start, r)
            else:
                r = add_2x2x1_block(U, V, W, 0, m_start, p_start, r)

    # 1x4x7: n_off=2
    for m_start in [0, 2]:
        for p_start in [0, 2, 4, 6]:
            p_end = min(p_start + 2, p)
            if p_end - p_start == 2:
                r = add_1x2x2_block(U, V, W, 2, m_start, p_start, r)
            else:
                # 1x2x1 block
                for j in range(2):
                    U[(2+0)*1 + (m_start+j), r] = 1
                    V[(m_start+j)*1 + (p_start+0), r] = 1
                    W[(p_start+0)*1 + (2+0), r] = 1
                    r += 1

    # r should be 72 now. Let's check.
    # 2x4x7: 2 m-blocks * (3 2x2x2 + 1 2x2x1) = 2*(3*7 + 4) = 2*25 = 50
    # 1x4x7: 2 m-blocks * (3 1x2x2 + 1 1x2x1) = 2*(3*4 + 2) = 2*14 = 28
    # Total: 50 + 28 = 78. That matches the cell finding!

    # Verify: rank should be 78.
    # If r != 78, something is wrong. Let's just use r as the rank.
    actual_rank = r

    # Trim to actual rank
    U = U[:, :actual_rank]
    V = V[:, :actual_rank]
    W = W[:, :actual_rank]

    # Verify the decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    # Let's do a quick check with random values
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B

    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C_computed[i, k] += U[i*m + j, :] @ (V[j*p + k, :] * W[k*n + i, :])

    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r in range(rank):
            k = r % p
            j = (r // p) % m
            i = r // (p * m)
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
        actual_rank = rank

    if time.time() > deadline - 5.0:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
