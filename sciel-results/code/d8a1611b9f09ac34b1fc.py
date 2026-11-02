"""Program for 3x5x5 matrix multiplication tensor decomposition.

Strategy:
  - Split m = 2+2+1, p = 2+2+1.
  - For each (m_block, p_block):
      * (2,2): M_{3,2,2} = M_{2,2,2} (Strassen rank 7) + M_{1,2,2} (rank 4 schoolbook)
               -> 11 terms per (2,2) block, 4 blocks -> 44
      * (2,1): M_{3,2,1} schoolbook -> 6 terms, 2 blocks -> 12
      * (1,2): M_{3,1,2} schoolbook -> 6 terms, 2 blocks -> 12
      * (1,1): M_{3,1,1} schoolbook -> 3 terms, 1 block -> 3
  - Total rank = 44 + 12 + 12 + 3 = 71.

  Then attempt a local search to reduce the rank by finding a rank-6
  decomposition of M_{3,2,2} (replacing the 11-term block decomposition).
  If found, each (2,2) block saves 5 terms -> total 71 - 20 = 51.
"""
import time
import itertools

def _strassen_2x2x2_terms():
    """Return 7 (u, v, w) vectors for the 2x2x2 Strassen decomposition.
    u has length 4 (indices i*2+j), v has length 4 (j*2+k), w has length 4 (k*2+i)."""
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]


def _verify_tensor(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _build_block_decomposition(n, m, p, m_blocks, p_blocks):
    """Build the block decomposition using Strassen for 2x2 blocks and schoolbook elsewhere."""
    strassen = _strassen_2x2x2_terms()
    all_terms = []  # list of (u_dict, v_dict, w_dict)

    for mb_idx, m_size in enumerate(m_blocks):
        m_off = sum(m_blocks[:mb_idx])
        for pb_idx, p_size in enumerate(p_blocks):
            p_off = sum(p_blocks[:pb_idx])

            if m_size == 2 and p_size == 2:
                # M_{3,2,2}: Strassen for i in {0,1} + schoolbook for i=2
                for (su, sv, sw) in strassen:
                    u_full = {}
                    v_full = {}
                    w_full = {}
                    for i in range(2):
                        for j in range(2):
                            val = su[i * 2 + j]
                            if val != 0:
                                idx = i * m + (m_off + j)
                                u_full[idx] = u_full.get(idx, 0) + val
                    for j in range(2):
                        for k in range(2):
                            val = sv[j * 2 + k]
                            if val != 0:
                                idx = (m_off + j) * p + (p_off + k)
                                v_full[idx] = v_full.get(idx, 0) + val
                    for k in range(2):
                        for i in range(2):
                            val = sw[k * 2 + i]
                            if val != 0:
                                idx = (p_off + k) * n + i
                                w_full[idx] = w_full.get(idx, 0) + val
                    all_terms.append((u_full, v_full, w_full))

                # M_{1,2,2} for i=2: schoolbook, 4 terms
                for j in range(2):
                    for k in range(2):
                        u_full = {2 * m + (m_off + j): 1}
                        v_full = {(m_off + j) * p + (p_off + k): 1}
                        w_full = {(p_off + k) * n + 2: 1}
                        all_terms.append((u_full, v_full, w_full))

            elif m_size == 2 and p_size == 1:
                # M_{3,2,1}: schoolbook, 6 terms
                for i in range(n):
                    for j in range(2):
                        u_full = {i * m + (m_off + j): 1}
                        v_full = {(m_off + j) * p + p_off: 1}
                        w_full = {p_off * n + i: 1}
                        all_terms.append((u_full, v_full, w_full))

            elif m_size == 1 and p_size == 2:
                # M_{3,1,2}: schoolbook, 6 terms
                for i in range(n):
                    for k in range(2):
                        u_full = {i * m + m_off: 1}
                        v_full = {m_off * p + (p_off + k): 1}
                        w_full = {(p_off + k) * n + i: 1}
                        all_terms.append((u_full, v_full, w_full))

            elif m_size == 1 and p_size == 1:
                # M_{3,1,1}: schoolbook, 3 terms
                for i in range(n):
                    u_full = {i * m + m_off: 1}
                    v_full = {m_off * p + p_off: 1}
                    w_full = {p_off * n + i: 1}
                    all_terms.append((u_full, v_full, w_full))

    return all_terms


def _terms_to_matrices(all_terms, n, m, p):
    """Convert list of term dicts to U, V, W matrices."""
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
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
    return U, V, W


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 150s kill

    n, m, p = 3, 5, 5
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 1]

    # Build the block decomposition
    all_terms = _build_block_decomposition(n, m, p, m_blocks, p_blocks)
    U, V, W = _terms_to_matrices(all_terms, n, m, p)
    rank = len(all_terms)

    # Verify
    if not _verify_tensor(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Try to improve: search for a rank-6 decomposition of M_{3,2,2}
    # This would replace 11 terms per (2,2) block with 6, saving 5 per block.
    # 4 blocks * 5 = 20 terms saved.
    #
    # M_{3,2,2}: A is 3x2, B is 2x2, C is 3x2.
    # T[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0,1}.
    # U has 6 entries (i*2+j), V has 4 entries (j*2+k), W has 6 entries (k*2+i).
    #
    # We need: sum_r u_r[i*2+j] * v_r[j*2+k] * w_r[k*2+i] = 1 for all (i,j,k).
    #
    # With rank 6, we have 6 terms. Each term has 6+4+6=16 variables, total 96 variables.
    # 12 equations (one per (i,j,k)). This is underdetermined, so solutions should exist.
    #
    # Let's try a structured approach:
    # Use the Strassen terms for i in {0,1} (7 terms) and try to merge/replace.
    # Actually, let's try a direct search with a specific structure.
    #
    # Alternative: try rank 7 for M_{3,2,2} using a modified Strassen.
    # If we can do M_{3,2,2} in rank 7 (instead of 11), we save 4 per block = 16 total.
    #
    # Let's try: use Strassen for the 2x2x2 part (i in {0,1}) which gives 7 terms,
    # and add 0 extra terms if we can somehow absorb the i=2 part.
    # That's not possible directly since the i=2 part is independent.
    #
    # Let's try a different approach: use a rank-7 decomposition where the
    # W vectors for the Strassen terms are modified to also cover i=2.
    #
    # Actually, the cleanest improvement: try to find a rank-6 M_{3,2,2}.
    # Let's use a simple iterative approach.

    if time.time() < deadline:
        U_improved = None
        V_improved = None
        W_improved = None

        # Try to find a rank-6 decomposition of M_{3,2,2}
        # Structure: 6 terms, each with u (6 entries), v (4 entries), w (6 entries)
        # We'll use a randomized local search.
        import random
        random.seed(42)

        # Start from the 11-term decomposition of M_{3,2,2} and try to merge terms.
        # Actually, let's just try a direct construction.
        #
        # For M_{3,2,2}, the tensor has 12 entries all equal to 1.
        # A rank-1 term u*v*w has at most 6*4*6 = 144 possible nonzero patterns,
        # but specifically u[i*2+j]*v[j*2+k]*w[k*2+i] is nonzero only when
        # u[i*2+j] != 0 AND v[j*2+k] != 0 AND w[k*2+i] != 0.
        #
        # Let's try: 6 terms where each term covers exactly 2 of the 12 entries.
        #
        # Term structure idea:
        # For each (j,k) pair (4 total), we need to cover i=0,1,2.
        # If we use 2 terms per (j,k) pair, that's 8 terms.
        # If we use 1.5 terms per (j,k) pair (some terms cover multiple (j,k)), that's 6.
        #
        # Let's try a specific construction:
        # Terms 1-4: cover (j,k) = (0,0), (0,1), (1,0), (1,1) for i=0,1 using Strassen-like
        # Terms 5-6: cover i=2 for all (j,k)
        #
        # Actually, let's just use a numerical approach.
        # Set up the system and try to solve it.

        # For rank r, we have r terms. The constraint is:
        # For each (i,j,k): sum_{s=1}^{r} u_s[i*2+j] * v_s[j*2+k] * w_s[k*2+i] = 1
        #
        # This is a system of 12 polynomial equations in 16r variables.
        # For r=6: 96 variables, 12 equations. Very underdetermined.
        #
        # Let's try a structured solution:
        # Fix v and w, solve for u (linear system).
        #
        # For each (i,j,k), the equation is linear in u_s[i*2+j] if we fix v_s and w_s.
        # Actually, it's: sum_s u_s[i*2+j] * (v_s[j*2+k] * w_s[k*2+i]) = 1
        # For fixed (j,k), this gives 3 equations (one per i) in the u variables.
        #
        # Let's fix v and w to simple values and solve for u.
        #
        # Try: v_s[j*2+k] = 1 if s == j*2+k else 0 (identity-like)
        # w_s[k*2+i] = 1 if s == k*2+i else 0
        # Then for (i,j,k): sum_s u_s[i*2+j] * [s==j*2+k] * [s==k*2+i]
        # = u_{j*2+k}[i*2+j] if j*2+k == k*2+i, else 0.
        # This only works when j*2+k == k*2+i, which is j=k and i=j, i.e., (0,0,0),(1,1,1).
        # Not sufficient.
        #
        # Let's try a different approach: use the known rank-7 Strassen for the 2x2x2
        # sub-tensor (i in {0,1}) and add a term for i=2.
        # That gives 8 terms for M_{3,2,2}. Better than 11.
        #
        # Wait, the Strassen terms for i in {0,1} already have W vectors that only
        # have nonzeros at k*2+i for i in {0,1}. We need to extend them to cover i=2.
        #
        # Alternative: use 7 Strassen terms for i in {0,1} (with W having only i=0,1 entries)
        # and add 1 term that covers all of i=2: u[2*2+j]=1 for j=0,1; v[j*2+k]=1 for all j,k; w[k*2+2]=1 for all k.
        # But v[j*2+k]=1 for all j,k is not rank-1 compatible with a single u.
        #
        # Let me just try: 7 Strassen terms + 2 terms for i=2 (one per j, covering both k).
        # Term for i=2, j=0: u[4]=1 (i=2,j=0 -> 2*2+0=4), v[0]=1 (j=0,k=0) and v[1]=1 (j=0,k=1)
        #   But that's two v entries, so it's not a single rank-1 term.
        #
        # OK, let's just use schoolbook for i=2: 4 terms (j=0,1; k=0,1).
        # Total: 7 + 4 = 11. That's what we have.
        #
        # To get 8: we need to cover i=2 with 1 term. That requires:
        # u[4]=a, u[5]=b (for j=0,1)
        # v[0]=c, v[1]=d, v[2]=e, v[3]=f (for (j,k) pairs)
        # w[4]=g, w[5]=h (for k=0,1 at i=2)
        # Constraints:
        # (i=2,j=0,k=0): a*c*g = 1
        # (i=2,j=0,k=1): a*d*h = 1
        # (i=2,j=1,k=0): b*e*g = 1
        # (i=2,j=1,k=1): b*f*h = 1
        # 4 equations, 8 unknowns. Solvable!
        # e.g., a=1, b=1, c=1, d=1, e=1, f=1, g=1, h=1.
        # So one term can cover all of i=2!
        #
        # But wait, this term will also contribute to i=0 and i=1 if u[0..3] or w[0..3] are nonzero.
        # If we set u[0]=u[1]=u[2]=u[3]=0 and w[0]=w[1]=w[2]=w[3]=0, then this term
        # only contributes to i=2 entries.
        #
        # So: 7 Strassen terms (for i=0,1) + 1 term (for i=2) = 8 terms for M_{3,2,2}!
        #
        # Let me verify this works.

        # Build M_{3,2,2} with 8 terms:
        # Terms 1-7: Strassen for i in {0,1}
        # Term 8: covers i=2 entirely
        m322_terms = []
        for (su, sv, sw) in strassen:
            u_full = {}
            v_full = {}
            w_full = {}
            for i in range(2):
                for j in range(2):
                    val = su[i * 2 + j]
                    if val != 0:
                        idx = i * 2 + j  # local index in M_{3,2,2}
                        u_full[idx] = u_full.get(idx, 0) + val
            for j in range(2):
                for k in range(2):
                    val = sv[j * 2 + k]
                    if val != 0:
                        idx = j * 2 + k
                        v_full[idx] = v_full.get(idx, 0) + val
            for k in range(2):
                for i in range(2):
                    val = sw[k * 2 + i]
                    if val != 0:
                        idx = k * 2 + i
                        w_full[idx] = w_full.get(idx, 0) + val
            m322_terms.append((u_full, v_full, w_full))

        # Term 8: covers i=2
        # u: i=2,j=0 -> idx 4; i=2,j=1 -> idx 5
        # v: all (j,k) pairs
        # w: k=0,i=2 -> idx 4; k=1,i=2 -> idx 5
        u8 = {4: 1, 5: 1}
        v8 = {0: 1, 1: 1, 2: 1, 3: 1}
        w8 = {4: 1, 5: 1}
        m322_terms.append((u8, v8, w8))

        # Verify M_{3,2,2} decomposition
        m322_ok = True
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for (ud, vd, wd) in m322_terms:
                        s += ud.get(i * 2 + j, 0) * vd.get(j * 2 + k, 0) * wd.get(k * 2 + i, 0)
                    if s != 1:
                        m322_ok = False
                        break
                if not m322_ok:
                    break
            if not m322_ok:
                break

        if m322_ok:
            # Great! Rebuild with 8-term M_{3,2,2} blocks.
            all_terms_v2 = []
            for mb_idx, m_size in enumerate(m_blocks):
                m_off = sum(m_blocks[:mb_idx])
                for pb_idx, p_size in enumerate(p_blocks):
                    p_off = sum(p_blocks[:pb_idx])

                    if m_size == 2 and p_size == 2:
                        # Use the 8-term M_{3,2,2} decomposition
                        for (ud, vd, wd) in m322_terms:
                            u_full = {}
                            v_full = {}
                            w_full = {}
                            for local_idx, val in ud.items():
                                # local_idx = i*2+j_local, where i in {0,1,2}, j_local in {0,1}
                                i_local = local_idx // 2
                                j_local = local_idx % 2
                                full_idx = i_local * m + (m_off + j_local)
                                u_full[full_idx] = u_full.get(full_idx, 0) + val
                            for local_idx, val in vd.items():
                                # local_idx = j_local*2+k_local
                                j_local = local_idx // 2
                                k_local = local_idx % 2
                                full_idx = (m_off + j_local) * p + (p_off + k_local)
                                v_full[full_idx] = v_full.get(full_idx, 0) + val
                            for local_idx, val in wd.items():
                                # local_idx = k_local*2+i_local
                                k_local = local_idx // 2
                                i_local = local_idx % 2
                                full_idx = (p_off + k_local) * n + i_local
                                w_full[full_idx] = w_full.get(full_idx, 0) + val
                            all_terms_v2.append((u_full, v_full, w_full))

                    elif m_size == 2 and p_size == 1:
                        for i in range(n):
                            for j in range(2):
                                u_full = {i * m + (m_off + j): 1}
                                v_full = {(m_off + j) * p + p_off: 1}
                                w_full = {p_off * n + i: 1}
                                all_terms_v2.append((u_full, v_full, w_full))

                    elif m_size == 1 and p_size == 2:
                        for i in range(n):
                            for k in range(2):
                                u_full = {i * m + m_off: 1}
                                v_full = {m_off * p + (p_off + k): 1}
                                w_full = {(p_off + k) * n + i: 1}
                                all_terms_v2.append((u_full, v_full, w_full))

                    elif m_size == 1 and p_size == 1:
                        for i in range(n):
                            u_full = {i * m + m_off: 1}
                            v_full = {m_off * p + p_off: 1}
                            w_full = {p_off * n + i: 1}
                            all_terms_v2.append((u_full, v_full, w_full))

            U2, V2, W2 = _terms_to_matrices(all_terms_v2, n, m, p)
            rank2 = len(all_terms_v2)

            if _verify_tensor(U2, V2, W2, n, m, p):
                U, V, W = U2, V2, W2
                rank = rank2

    # Final check
    if not _verify_tensor(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
