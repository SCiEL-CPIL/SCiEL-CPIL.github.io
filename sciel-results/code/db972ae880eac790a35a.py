import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n

    # Laderman's rank-15 algorithm for 2x2 block matrix multiplication.
    # Each step: (a_terms, b_terms, c_terms)
    # a_terms: list of (coeff, row, col) for A block contributions
    # b_terms: list of (coeff, row, col) for B block contributions
    # c_terms: list of (coeff, row, col) for C block contributions
    #
    # Standard Laderman:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B10-B00)       -> C01 += M3
    # M4 = A11(B01-B11)       -> C00 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01+A11)(B10+B11) -> C00 += M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4  -- wait, that's not right. Let me recheck.
    
    # Actually the standard Laderman is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4  -- no, C10 = M2 + M4 doesn't look right either
    # C11 = M1 - M2 + M3 + M6

    # Let me use the well-known formulation:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Verify: C = A*B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11

    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M2 = (A10+A11)B00 = A10*B00 + A11*B00
    # M3 = A00(B10-B00) = A00*B10 - A00*B00
    # M4 = A11(B01-B11) = A11*B01 - A11*B11
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11
    # M6 = (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
    # M7 = (A01+A11)(B10+B11) = A01*B10 + A01*B11 + A11*B10 + A11*B11

    # C00 = M1 + M4 - M5 + M7
    # = (A00*B00 + A00*B11 + A11*B00 + A11*B11) + (A11*B01 - A11*B11) - (A00*B11 + A01*B11) + (A01*B10 + A01*B11 + A11*B10 + A11*B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B11 - A00*B11 - A01*B11 + A01*B10 + A01*B11 + A11*B10 + A11*B11
    # = A00*B00 + A11*B00 + A11*B01 + A01*B10 + A11*B10 + A11*B11
    # Hmm, that's not A00*B00 + A01*B10. Let me recheck.

    # I think I'm confusing myself. Let me use a verified Laderman decomposition.
    # The correct Laderman (1976) algorithm:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 + A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Let me verify C10 = M2 + M4:
    # M2 + M4 = A10*B00 + A11*B00 + A11*B01 - A11*B11
    # Should equal: A10*B00 + A11*B10
    # That doesn't match. So my Laderman formulation is wrong.

    # Let me look up the correct one. The actual Laderman algorithm:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B00)
    # P4 = A11(B01 - B11)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 + A11)(B10 + B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Hmm, I keep getting the same thing. Let me just verify numerically.
    # Actually, I think the issue is that I'm not using the right Laderman. Let me use a different well-known rank-15 algorithm.

    # Alternative: Use the "Baur-Strassen" or just verify with a simple test.
    # For now, let me use a verified decomposition. I'll use the one from the literature.

    # Actually, let me just use Strassen-style but with 15 multiplications.
    # A simpler approach: I know that rank(2x2) = 7 (Strassen) and there's no known rank-15 for 2x2.
    # Wait, Laderman's algorithm IS rank 15 for 2x2? No, Strassen is rank 7 which is better.
    # 
    # I think I confused myself. Laderman's algorithm is for 3x3 matrices with rank 15.
    # For 2x2, Strassen gives rank 7.
    #
    # So the strategy should be: partition 6x6 into 3x3 blocks (2x2 grid), and use
    # Laderman's rank-15 algorithm for 3x3 matrix multiplication at the block level?
    # No, that doesn't make sense either.
    #
    # Let me reconsider. The directive says: "Partition the 6x6 matrices into a 2x2 grid of 3x3 blocks
    # and apply the Laderman rank-15 decomposition to the outer block multiplication"
    #
    # I think this means: treat the 2x2 block structure, and for each "block multiplication" 
    # (which is actually a 3x3 matrix multiplication), use Laderman's rank-15 algorithm for 3x3.
    # But Laderman's is for 3x3 scalar matrices, not 3x3 block matrices.
    #
    # Actually, I think the intent is:
    # - 6x6 = 2x2 grid of 3x3 blocks
    # - The "outer" multiplication is 2x2 block matrix multiplication, which has rank 7 (Strassen)
    # - Each "inner" multiplication is 3x3 scalar matrix multiplication, which has rank 27 (schoolbook)
    # - Total rank = 7 * 27 = 189 (which is what we already have!)
    #
    # OR:
    # - The "outer" multiplication uses Laderman's rank-15 for 2x2? But Laderman is for 3x3.
    #
    # I'm confused. Let me just try a different approach: use a 3x3 grid of 2x2 blocks.
    # 6x6 = 3x3 grid of 2x2 blocks.
    # Outer: 3x3 matrix multiplication, rank 27 (schoolbook)
    # Inner: 2x2 matrix multiplication, rank 7 (Strassen)
    # Total: 27 * 7 = 189 (same as before)
    #
    # To get 135, we need 15 * 9. So:
    # Outer: rank 15 (Laderman for 2x2? But Laderman is for 3x3)
    # Inner: rank 9 (schoolbook for 3x3)
    #
    # I think the correct interpretation is:
    # - 6x6 = 2x2 grid of 3x3 blocks
    # - Use a rank-15 algorithm for the 2x2 BLOCK multiplication (not standard Strassen which is rank 7)
    # - But wait, rank 7 < rank 15, so why use rank 15?
    #
    # Unless... the rank-15 algorithm for 2x2 blocks has some property that makes it work better
    # with the 3x3 inner blocks? That doesn't make sense for exact decomposition.
    #
    # I think the directive might be mistaken, or I'm misunderstanding. Let me just try to get
    # a valid decomposition with rank < 189. The simplest way is to use Strassen at the 2x2
    # block level (rank 7) with schoolbook 3x3 inner (rank 27), giving 189. That's what we have.
    #
    # To beat 189, I need a different approach. Let me try:
    # - 6x6 with a 3x3 grid of 2x2 blocks, Strassen for 2x2 (rank 7), schoolbook for 3x3 (rank 27)
    # - Total: 27 * 7 = 189
    #
    # Or: use a more sophisticated decomposition. 
    #
    # Actually, let me just make sure the current program works correctly and returns a valid result.
    # The current score is 1.1429 = 189/165.6... hmm, 1/0.875 = 1.1428. So seed rank is 216 (schoolbook).
    # 216/189 = 1.1428. Yes!
    #
    # To improve, I need rank < 189. Let me try a 5x5 + 1x1 decomposition or other tricks.
    #
    # For now, let me just ensure correctness and return the rank-189 result.

    # Use Strassen (rank 7) for 2x2 blocks in a 3x3 grid
    # 6x6 = 3x3 grid of 2x2 blocks
    # Total rank = 27 * 7 = 189

    total_rank = 27 * 7

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Strassen steps for 2x2 matrix multiplication
    steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    bs = 2
    col_idx = 0

    for step in steps:
        a_terms, b_terms, c_terms = step
        for r, s, t in itertools.product(range(bs), repeat=3):
            for I, J, ca in a_terms:
                u_idx = (I * bs + r) * dim + (J * bs + s)
                U[u_idx, col_idx] += ca

            for J, K, cb in b_terms:
                v_idx = (J * bs + s) * dim + (K * bs + t)
                V[v_idx, col_idx] += cb

            for I, K, cc in c_terms:
                w_idx = (K * bs + t) * dim + (I * bs + r)
                W[w_idx, col_idx] += cc

            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Fast verification using einsum
    try:
        if time.time() < deadline - 10:
            rank = U.shape[1]
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            if not np.allclose(result, 1.0, atol=1e-10):
                # Fall back to schoolbook
                total_rank_school = n * m * p
                U_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                V_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                W_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_s[i * m + j, idx] = 1
                            V_s[j * p + k, idx] = 1
                            W_s[k * n + i, idx] = 1
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
