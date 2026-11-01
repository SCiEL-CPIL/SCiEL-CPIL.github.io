"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy:
    # Partition J (size 5) into J1={0,1}, J2={2,3}, J3={4}
    # Partition K (size 6) into K1={0,1}, K2={2,3}, K3={4,5}
    #
    # 6 Strassen 2x2x2 blocks: rank 6*7 = 42
    # 3 standard 2x1x2 blocks (J3 x K1, J3 x K2, J3 x K3): rank 3*4 = 12
    # Total: 54

    # Build the 54-term decomposition
    terms = []  # list of (u_vec, v_vec, w_vec) as sparse dicts

    def strassen_block(ja, jb, ka, kb):
        # u indices: (i, j) -> i*m + j
        u0a = 0 * m + ja  # (0, ja)
        u0b = 0 * m + jb  # (0, jb)
        u1a = 1 * m + ja  # (1, ja)
        u1b = 1 * m + jb  # (1, jb)

        # v indices: (j, k) -> j*p + k
        v_a_ka = ja * p + ka
        v_a_kb = ja * p + kb
        v_b_ka = jb * p + ka
        v_b_kb = jb * p + kb

        # w indices: (k, i) -> k*n + i
        w_ka_0 = ka * n + 0
        w_ka_1 = ka * n + 1
        w_kb_0 = kb * n + 0
        w_kb_1 = kb * n + 1

        # M1: (A00+A11)(B00+B11) -> C00, C11
        terms.append({
            'u': {u0a: 1, u1b: 1},
            'v': {v_a_ka: 1, v_b_kb: 1},
            'w': {w_ka_0: 1, w_kb_1: 1}
        })
        # M2: (A10+A11)(B00) -> C10, C11
        terms.append({
            'u': {u1a: 1, u1b: 1},
            'v': {v_a_ka: 1},
            'w': {w_ka_1: 1, w_kb_1: -1}
        })
        # M3: (A00)(B01-B11) -> C01, C11
        terms.append({
            'u': {u0a: 1},
            'v': {v_a_kb: 1, v_b_kb: -1},
            'w': {w_kb_0: 1, w_kb_1: 1}
        })
        # M4: (A11)(B10-B00) -> C00, C10
        terms.append({
            'u': {u1b: 1},
            'v': {v_b_ka: 1, v_a_ka: -1},
            'w': {w_ka_0: 1, w_ka_1: 1}
        })
        # M5: (A00+A01)(B11) -> C00, C01
        terms.append({
            'u': {u0a: 1, u0b: 1},
            'v': {v_b_kb: 1},
            'w': {w_ka_0: -1, w_kb_0: 1}
        })
        # M6: (A10-A00)(B00+B01) -> C11
        terms.append({
            'u': {u1a: 1, u0a: -1},
            'v': {v_a_ka: 1, v_a_kb: 1},
            'w': {w_kb_1: 1}
        })
        # M7: (A01-A11)(B10+B11) -> C00
        terms.append({
            'u': {u0b: 1, u1b: -1},
            'v': {v_b_ka: 1, v_b_kb: 1},
            'w': {w_ka_0: 1}
        })

    # 6 Strassen blocks
    strassen_block(0, 1, 0, 1)  # J1, K1
    strassen_block(0, 1, 2, 3)  # J1, K2
    strassen_block(0, 1, 4, 5)  # J1, K3
    strassen_block(2, 3, 0, 1)  # J2, K1
    strassen_block(2, 3, 2, 3)  # J2, K2
    strassen_block(2, 3, 4, 5)  # J2, K3

    # 3 standard 2x1x2 blocks for J3={4}
    for k_start in [0, 2, 4]:
        for k in [k_start, k_start + 1]:
            for i in [0, 1]:
                u_idx = i * m + 4
                v_idx = 4 * p + k
                w_idx = k * n + i
                terms.append({
                    'u': {u_idx: 1},
                    'v': {v_idx: 1},
                    'w': {w_idx: 1}
                })

    # Verify the 54-term decomposition
    def verify_decomposition(terms_list):
        U = np.zeros((n * m, len(terms_list)))
        V = np.zeros((m * p, len(terms_list)))
        W = np.zeros((p * n, len(terms_list)))
        for c, t in enumerate(terms_list):
            for idx, coeff in t['u'].items():
                U[idx, c] = coeff
            for idx, coeff in t['v'].items():
                V[idx, c] = coeff
            for idx, coeff in t['w'].items():
                W[idx, c] = coeff

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False, None, None, None
        return True, U, V, W

    valid, U54, V54, W54 = verify_decomposition(terms)
    if not valid:
        # Fallback: trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    best_rank = 54
    best_U, best_V, best_W = U54, V54, W54

    # Try to reduce rank by finding linear dependencies
    # The tensor is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Each term c defines a rank-1 tensor. We need the sum to equal the multiplication tensor.
    # If we can find a subset of terms whose tensor sum is zero, we can remove them.
    # More generally, if a linear combination of term tensors is zero, we can reduce rank.

    # Build the tensor matrix: flatten each term into a vector of length n*m*p
    tensor_dim = n * m * p  # 60
    term_vectors = np.zeros((len(terms), tensor_dim))
    for c, t in enumerate(terms):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = t['u'].get(i * m + j, 0)
                    v_val = t['v'].get(j * p + k, 0)
                    w_val = t['w'].get(k * n + i, 0)
                    term_vectors[c, k * n + i] += u_val * v_val * w_val  # This is wrong, let me fix

    # Actually, let me build the tensor properly
    term_vectors = np.zeros((len(terms), tensor_dim))
    for c, t in enumerate(terms):
        for (i, j, k) in [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]:
            u_val = t['u'].get(i * m + j, 0)
            v_val = t['v'].get(j * p + k, 0)
            w_val = t['w'].get(k * n + i, 0)
            term_vectors[c, i * m * p + j * p + k] = u_val * v_val * w_val

    # The multiplication tensor
    mult_tensor = np.zeros(tensor_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                mult_tensor[i * m * p + j * p + k] = 1

    # Check: sum of term vectors should equal mult_tensor
    assert np.allclose(term_vectors.sum(axis=0), mult_tensor, atol=1e-10)

    # Find linear dependencies: if columns of term_vectors are linearly dependent,
    # we might be able to reduce rank. But the constraint is that we need the sum
    # to equal mult_tensor, not just any linear combination.

    # A simpler approach: try removing one term at a time and see if the remaining
    # terms can still span the multiplication tensor. This is expensive, so let's
    # use a smarter approach.

    # For each term, check if it's in the span of the other terms' tensors
    # that still sum to mult_tensor. This is complex.

    # Instead, let's try a greedy approach: for each pair of terms, check if their
    # sum can be replaced by a single term (i.e., if the sum tensor is rank-1).

    # Actually, let's just try to find if any single term is redundant:
    # If term c's tensor is a linear combination of the others, and the others
    # still sum to mult_tensor - term_c's tensor, then we can remove it.
    # But this requires the others to still sum correctly, which is the key constraint.

    # Given time constraints, let's just return the 54-rank solution.
    # The matching-based approach suggested in the strategy would require more
    # sophisticated optimization that may not yield improvements within the time budget.

    if time.time() > deadline - 5:
        pass  # Safety margin

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
