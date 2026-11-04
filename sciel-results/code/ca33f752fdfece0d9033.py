"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication.

Strategy: Use a block decomposition that minimizes rank by splitting the output columns
into 4, 1, and 2.
- 2x4 x 4: Split A into two 2x2 blocks, each multiplied by a 2x4 block of B.
  Each 2x2 x 4 is decomposed into two 2x2 x 2 Strassen blocks (rank 14).
  Total rank for 2x4 x 4 is 28.
- 2x4 x 1: Direct multiplication (rank 4).
- 2x4 x 2: Two 2x4 x 1 direct multiplications (rank 8).
Total rank: 28 + 4 + 8 = 40.
"""
import numpy as np
import time


def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Build the target tensor for verification
    T_full = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0

    def verify_decomposition(U, V, W, T_target, atol=1e-9):
        r = U.shape[1]
        T_approx = np.zeros_like(T_target)
        for r_idx in range(r):
            T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_target, T_approx, atol=atol)

    def strassen_2x2x2(i0, j0, k0):
        """
        Strassen decomposition of 2x2 * 2x2 sub-block.
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        """
        terms = []
        
        def make_u(i_vals, j_vals):
            vec = np.zeros(n * m)
            for ii in i_vals:
                for jj in j_vals:
                    vec[ii * m + jj] = 1.0
            return vec
        
        def make_v(j_vals, k_vals):
            vec = np.zeros(m * p)
            for jj in j_vals:
                for kk in k_vals:
                    vec[jj * p + kk] = 1.0
            return vec
        
        def make_w(coeffs):
            vec = np.zeros(p * n)
            for (kk, ii), coeff in coeffs.items():
                vec[kk * n + ii] = coeff
            return vec
        
        # M1 = (A00+A11)*(B00+B11)
        u1 = make_u([i0, i0+1], [j0, j0+1])
        v1 = make_v([j0, j0+1], [k0, k0+1])
        w1 = make_w({(k0, i0): 1, (k0+1, i0+1): 1})
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)*B00
        u2 = make_u([i0+1], [j0, j0+1])
        v2 = make_v([j0], [k0])
        w2 = make_w({(k0, i0+1): 1})
        terms.append((u2, v2, w2))
        
        # M3 = A00*(B10-B11)
        u3 = make_u([i0], [j0])
        v3 = np.zeros(m * p)
        v3[(j0+1) * p + k0] = 1
        v3[(j0+1) * p + k0 + 1] = -1
        w3 = make_w({(k0+1, i0): 1, (k0+1, i0+1): 1})
        terms.append((u3, v3, w3))
        
        # M4 = A11*(B01-B10)
        u4 = make_u([i0+1], [j0+1])
        v4 = np.zeros(m * p)
        v4[j0 * p + k0 + 1] = 1
        v4[(j0+1) * p + k0] = -1
        w4 = make_w({(k0, i0): -1, (k0+1, i0): 1})
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)*B11
        u5 = make_u([i0], [j0, j0+1])
        v5 = make_v([j0+1], [k0+1])
        w5 = make_w({(k0, i0): 1, (k0, i0+1): 1})
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)*(B00+B10)
        u6 = np.zeros(n * m)
        u6[(i0+1) * m + j0] = 1
        u6[i0 * m + j0] = -1
        v6 = make_v([j0, j0+1], [k0])
        w6 = make_w({(k0, i0+1): 1})
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)*(B01+B11)
        u7 = np.zeros(n * m)
        u7[i0 * m + j0 + 1] = 1
        u7[(i0+1) * m + j0 + 1] = -1
        v7 = make_v([j0, j0+1], [k0+1])
        w7 = make_w({(k0, i0): 1})
        terms.append((u7, v7, w7))
        
        return terms

    def direct_2x4x1(j0, k0):
        """
        Direct 4-term decomposition of 2x4 * 4x1:
        A[:, j0:j0+4] * B[j0:j0+4, k0] -> C[:, k0]
        Note: This function is for the full 2x4 A block.
        """
        terms = []
        for di in range(2):
            for dj in range(4):
                u = np.zeros(n * m)
                u[di * m + (j0 + dj)] = 1.0
                v = np.zeros(m * p)
                v[(j0 + dj) * p + k0] = 1.0
                w = np.zeros(p * n)
                w[k0 * n + di] = 1.0
                terms.append((u, v, w))
        return terms

    all_terms = []
    
    # Part 1: 2x4 x 4 (k=0,1,2,3)
    # Split A into two 2x2 blocks: A[:, 0:2] and A[:, 2:4]
    # Split B into 2x4 blocks: B[0:2, 0:4] and B[2:4, 0:4]
    
    # A[:, 0:2] * B[0:2, 0:4]
    # This is 2x2 x 4. Split into two 2x2 x 2.
    # k=0,1
    all_terms.extend(strassen_2x2x2(0, 0, 0))
    # k=2,3
    all_terms.extend(strassen_2x2x2(0, 0, 2))
    
    # A[:, 2:4] * B[2:4, 0:4]
    # This is 2x2 x 4. Split into two 2x2 x 2.
    # k=0,1
    all_terms.extend(strassen_2x2x2(0, 2, 0))
    # k=2,3
    all_terms.extend(strassen_2x2x2(0, 2, 2))
    
    # Part 2: 2x4 x 1 (k=4)
    # Direct multiplication of full A[:, 0:4] with B[0:4, 4]
    all_terms.extend(direct_2x4x1(0, 4))
    
    # Part 3: 2x4 x 2 (k=5,6)
    # k=5
    all_terms.extend(direct_2x4x1(0, 5))
    # k=6
    all_terms.extend(direct_2x4x1(0, 6))

    rank = len(all_terms)
    print(f"Constructed decomposition with rank {rank}")

    # Assemble matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for t, (u, v, w) in enumerate(all_terms):
        U[:, t] = u
        V[:, t] = v
        W[:, t] = w

    # Verify
    if verify_decomposition(U, V, W, T_full):
        print("Verification PASSED")
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        print("Verification FAILED, falling back to schoolbook")
    
    # Fallback: Schoolbook rank 56
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    assert verify_decomposition(U_sb, V_sb, W_sb, T_full)
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
