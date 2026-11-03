"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen's algorithm for 2x2 blocks.
Combined approach: vectorized rank-1 term accumulation with block recursion.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = 140.0  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 term to the decomposition."""
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
            
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_block(i0, i1, j0, j1, k0, k1):
        """
        Strassen decomposition for a 2x2x2 block.
        i0,i1: row indices for A (2 rows)
        j0,j1: col indices for A / row indices for B (2 cols/rows)
        k0,k1: col indices for B (2 cols)
        """
        # M1: (A11+A22)*(B11+B22)
        add_rank1_term(
            [i0*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0, k1*n+i1], [1, 1]
        )
        
        # M2: (A21+A22)*B11
        add_rank1_term(
            [i1*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0], [1],
            [k0*n+i1, k1*n+i1], [1, -1]
        )
        
        # M3: A11*(B12-B22)
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k1, j1*p+k1], [1, -1],
            [k1*n+i0, k1*n+i1], [1, 1]
        )
        
        # M4: A22*(B21-B11)
        add_rank1_term(
            [i1*m+j1], [1],
            [j1*p+k0, j0*p+k0], [1, -1],
            [k0*n+i0, k0*n+i1], [1, 1]
        )
        
        # M5: (A11+A12)*B22
        add_rank1_term(
            [i0*m+j0, i0*m+j1], [1, 1],
            [j1*p+k1], [1],
            [k0*n+i0, k1*n+i0], [-1, 1]
        )
        
        # M6: (A21-A11)*(B11+B12)
        add_rank1_term(
            [i1*m+j0, i0*m+j0], [1, -1],
            [j0*p+k0, j0*p+k1], [1, 1],
            [k1*n+i1], [1]
        )
        
        # M7: (A12-A22)*(B21+B22)
        add_rank1_term(
            [i0*m+j1, i1*m+j1], [1, -1],
            [j1*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0], [1]
        )

    def schoolbook_block(i_start, j_start, k_start, n_l, m_l, p_l):
        """Schoolbook decomposition for a block."""
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_start + i
                    jj = j_start + j
                    kk = k_start + k
                    add_rank1_term(
                        [ii*m+jj], [1.0],
                        [jj*p+kk], [1.0],
                        [kk*n+ii], [1.0]
                    )

    def decomp_2x2x1(i0, i1, j0, j1, k0):
        """Rank 2 decomposition for 2x2 * 2x1."""
        # Term 1: j=j0
        add_rank1_term(
            [i0*m+j0, i1*m+j0], [1, 1],
            [j0*p+k0], [1],
            [k0*n+i0, k0*n+i1], [1, 1]
        )
        # Term 2: j=j1
        add_rank1_term(
            [i0*m+j1, i1*m+j1], [1, 1],
            [j1*p+k0], [1],
            [k0*n+i0, k0*n+i1], [1, 1]
        )

    def decomp_1x2x2(i0, j0, j1, k0, k1):
        """Rank 2 decomposition for 1x2 * 2x2."""
        # Term 1: j=j0
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k0, j0*p+k1], [1, 1],
            [k0*n+i0, k1*n+i0], [1, 1]
        )
        # Term 2: j=j1
        add_rank1_term(
            [i0*m+j1], [1],
            [j1*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0, k1*n+i0], [1, 1]
        )

    def decomp_1x2x1(i0, j0, j1, k0):
        """Rank 2 decomposition for 1x2 * 2x1."""
        # Term 1: j=j0
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k0], [1],
            [k0*n+i0], [1]
        )
        # Term 2: j=j1
        add_rank1_term(
            [i0*m+j1], [1],
            [j1*p+k0], [1],
            [k0*n+i0], [1]
        )

    def decomp_1x1x2(i0, j0, k0, k1):
        """Rank 2 decomposition for 1x1 * 1x2."""
        # Term 1: k=k0
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k0], [1],
            [k0*n+i0], [1]
        )
        # Term 2: k=k1
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k1], [1],
            [k1*n+i0], [1]
        )

    def decomp_2x1x2(i0, i1, j0, k0, k1):
        """Rank 2 decomposition for 2x1 * 1x2."""
        # Term 1: i=i0
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k0, j0*p+k1], [1, 1],
            [k0*n+i0, k1*n+i0], [1, 1]
        )
        # Term 2: i=i1
        add_rank1_term(
            [i1*m+j0], [1],
            [j0*p+k0, j0*p+k1], [1, 1],
            [k0*n+i1, k1*n+i1], [1, 1]
        )

    def decomp_2x1x1(i0, i1, j0, k0):
        """Rank 2 decomposition for 2x1 * 1x1."""
        # Term 1: i=i0
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k0], [1],
            [k0*n+i0], [1]
        )
        # Term 2: i=i1
        add_rank1_term(
            [i1*m+j0], [1],
            [j0*p+k0], [1],
            [k0*n+i1], [1]
        )

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        """Recursively generate rank-1 terms for a block."""
        if time.time() > deadline:
            return

        # Base case: 1x1x1
        if n_l == 1 and m_l == 1 and p_l == 1:
            add_rank1_term(
                [i_off*m+j_off], [1.0],
                [j_off*p+k_off], [1.0],
                [k_off*n+i_off], [1.0]
            )
            return

        # 2x2x2: use Strassen
        if n_l == 2 and m_l == 2 and p_l == 2:
            strassen_block(i_off, i_off+1, j_off, j_off+1, k_off, k_off+1)
            return

        # 2x2x1: use dedicated decomposition (rank 2)
        if n_l == 2 and m_l == 2 and p_l == 1:
            decomp_2x2x1(i_off, i_off+1, j_off, j_off+1, k_off)
            return

        # 1x2x2: use dedicated decomposition (rank 2)
        if n_l == 1 and m_l == 2 and p_l == 2:
            decomp_1x2x2(i_off, j_off, j_off+1, k_off, k_off+1)
            return

        # 1x2x1: use dedicated decomposition (rank 2)
        if n_l == 1 and m_l == 2 and p_l == 1:
            decomp_1x2x1(i_off, j_off, j_off+1, k_off)
            return

        # 1x1x2: use dedicated decomposition (rank 2)
        if n_l == 1 and m_l == 1 and p_l == 2:
            decomp_1x1x2(i_off, j_off, k_off, k_off+1)
            return

        # 2x1x2: use dedicated decomposition (rank 2)
        if n_l == 2 and m_l == 1 and p_l == 2:
            decomp_2x1x2(i_off, i_off+1, j_off, k_off, k_off+1)
            return

        # 2x1x1: use dedicated decomposition (rank 2)
        if n_l == 2 and m_l == 1 and p_l == 1:
            decomp_2x1x1(i_off, i_off+1, j_off, k_off)
            return

        # 1x1x1 already handled above

        # Recursive splits

        # Split along n
        if n_l >= 2:
            n1 = min(2, n_l)
            n2 = n_l - n1
            gen_terms(n1, m_l, p_l, i_off, j_off, k_off)
            gen_terms(n2, m_l, p_l, i_off + n1, j_off, k_off)
            return

        # Split along m
        if m_l >= 2:
            m1 = min(2, m_l)
            m2 = m_l - m1
            gen_terms(n_l, m1, p_l, i_off, j_off, k_off)
            gen_terms(n_l, m2, p_l, i_off, j_off + m1, k_off)
            return

        # Split along p
        if p_l >= 2:
            p1 = min(2, p_l)
            p2 = p_l - p1
            gen_terms(n_l, m_l, p1, i_off, j_off, k_off)
            gen_terms(n_l, m_l, p2, i_off, j_off, k_off + p1)
            return

        # Fallback: schoolbook
        schoolbook_block(i_off, j_off, k_off, n_l, m_l, p_l)

    # Generate the full decomposition
    gen_terms(n, m, p, 0, 0, 0)

    if not U_vecs:
        # Fallback to schoolbook if something went wrong
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

    rank = len(U_vecs)
    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Verify correctness
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B

    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = val

    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
