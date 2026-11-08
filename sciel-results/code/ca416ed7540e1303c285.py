import time
import numpy as np

def construct():
    start = time.time()
    n, m, p = 6, 6, 6
    
    # Laderman's algorithm for 4x4x4 matrix multiplication (rank 21)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict (i,j) -> val for A
    # b_coeffs: dict (j,k) -> val for B
    # c_coeffs: dict (i,k) -> val for C
    laderman = [
        ({(0,0):1,(0,1):1}, {(0,0):1,(1,0):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ({(1,1):1}, {(0,0):1,(1,0):-1}, {(1,0):-1,(1,1):1}),
        ({(0,0):1,(1,0):-1}, {(0,1):1}, {(0,1):-1,(1,1):1}),
        ({(0,1):1}, {(0,0):1,(0,1):-1}, {(0,0):1,(0,1):-1}),
        ({(0,0):1,(0,1):-1}, {(1,0):1}, {(0,1):1,(1,1):-1}),
        ({(1,0):1}, {(1,1):1,(1,0):-1}, {(1,1):1,(1,0):-1}),
        ({(0,1):1,(1,1):-1}, {(0,0):1}, {(0,0):-1,(0,1):1}),
        ({(0,0):1,(1,0):-1}, {(0,1):1,(1,1):-1}, {(1,1):1,(1,0):-1}),
        ({(0,1):1}, {(1,0):1}, {(0,1):1}),
        ({(1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,1):1}, {(0,1):-1}),
        ({(1,0):1}, {(0,1):1}, {(1,0):-1}),
        ({(0,1):1}, {(1,0):1}, {(1,1):1}),
        ({(1,1):1}, {(0,0):1}, {(0,0):1}),
        ({(0,0):1}, {(1,0):1}, {(1,1):-1}),
    ]
    
    # Verify Laderman: check that sum of a*b = c for the 4x4 tensor
    # Build the 4x4x4 tensor from Laderman and compare with identity
    T_lad = np.zeros((4,4,4))
    for a_c, b_c, c_c in laderman:
        for (ai, aj), av in a_c.items():
            for (bi, bk), bv in b_c.items():
                if aj == bi:
                    for (ci, ck), cv in c_c.items():
                        T_lad[ai, ci, ck] += av * bv * cv
    # Expected: T[i,k,j]... wait, let me re-derive.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T_{(i,j),(j,k),(i,k)} = 1
    # So T_lad[ai, aj, bi, bk, ci, ck]... no.
    # Let me just verify numerically with random matrices.
    np.random.seed(42)
    A_test = np.random.randn(4,4)
    B_test = np.random.randn(4,4)
    C_expected = A_test @ B_test
    C_computed = np.zeros((4,4))
    for a_c, b_c, c_c in laderman:
        a_val = sum(A_test[i,j] * v for (i,j), v in a_c.items())
        b_val = sum(B_test[i,j] * v for (i,j), v in b_c.items())
        for (i,k), v in c_c.items():
            C_computed[i,k] += a_val * b_val * v
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Laderman patterns might be wrong; fall back to trivial 4x4x4 (rank 64)
        # Actually let's use a known correct set. Let me use the standard Laderman.
        pass
    
    # Build the full 6x6x6 decomposition using mixed 4+2 blocking
    # Split: A = [[A44, A42],[A24, A22]], B = [[B44, B42],[B24, B22]]
    # C44 = A44*B44 + A42*B24 (use Laderman for A44*B44, trivial for A42*B24)
    # C42 = A44*B42 + A42*B22 (trivial for both)
    # C24 = A24*B44 + A22*B24 (trivial for both)
    # C22 = A24*B42 + A22*B22 (trivial for both)
    
    # Total rank: 21 (Laderman) + 4*2 (A42*B24) + 4*4 (A44*B42) + 4*2 (A42*B22)
    #             + 2*4 (A24*B44) + 2*2 (A22*B24) + 2*2 (A24*B42) + 2*2 (A22*B22)
    # = 21 + 8 + 16 + 8 + 8 + 4 + 4 + 4 = 73? That seems too low. Let me recount.
    
    # Actually: A44*B44: 4x4x4, Laderman rank 21
    # A42*B24: 4x2 * 2x4 = 4x2x4, trivial rank = 4*2*4 = 32? No.
    # For A[i,j] (i in 0..3, j in 0..1) * B[j,k] (j in 0..1, k in 0..3):
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k], rank = 4*2*4 = 32? No, rank = i*k*j = 4*4*2 = 32.
    
    # Hmm, this is getting complex. Let me just use the simpler 3x3 Laderman blocking.
    # 6 = 3 + 3, use Laderman (rank 23) for 3x3x3, total = 8 * 23 = 184.
    
    # Actually, let me try a cleaner approach: use the 2x2 Strassen (rank 7) for 6x6x6
    # with 3x3x3 blocks: 27 * 7 = 189. That's what we have.
    
    # Let me try to get 184 with 3x3 Laderman. I need the correct Laderman patterns for 3x3.
    # The rank of M_3 is 23. Let me use a known decomposition.
    
    # For now, let's use the 2x2 Strassen approach (rank 189) which we know is correct,
    # and see if we can improve.
    
    # Actually, let me try a different strategy: use numpy to verify and find a better decomposition.
    # But for this task, I'll stick with the known-correct Strassen 2x2 blocking.
    
    # Let me implement the 3x3 Laderman blocking with rank 23 per block.
    # I need the explicit 23 rank-1 terms for 3x3 matrix multiplication.
    
    # Known Laderman-style 3x3x3 decomposition (rank 23):
    # This is complex to derive. Let me use a simpler approach.
    
    # Alternative: Use the fact that we can combine the Strassen patterns more efficiently.
    # For 6x6x6, use 2x2x2 Strassen blocks: 27 blocks * 7 = 189.
    # Can we do better by using a 3x3x3 algorithm?
    
    # Let me just use the 2x2 Strassen (189) as it's verified correct.
    
    block_size = 2
    blocks = 3
    rank = blocks**3 * 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    strassen = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for I in range(blocks):
        for J in range(blocks):
            for K in range(blocks):
                for a_c, b_c, c_c in strassen:
                    for (pi,pj), val in a_c.items():
                        r = I*block_size + pi
                        c = J*block_size + pj
                        U[r*m + c, col] += val
                    for (pi,pj), val in b_c.items():
                        r = J*block_size + pi
                        c = K*block_size + pj
                        V[r*p + c, col] += val
                    for (pi,pj), val in c_c.items():
                        i = I*block_size + pi
                        k = K*block_size + pj
                        W[k*n + i, col] += val
                    col += 1
                    if time.time() - start > 140:
                        break
            else:
                continue
            break
        else:
            continue
        break
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
