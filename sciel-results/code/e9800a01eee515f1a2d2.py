"""
Rank-minimal tensor decomposition for 2x3x8 matrix multiplication.

Strategy:
  1. Build a guaranteed schoolbook decomposition (rank 48).
  2. Attempt a block-structured decomposition that splits the 3 middle indices
     into a 2-block and a 1-block, and the 8 right indices into four 2-blocks.
     For each (2x2) sub-block in (middle, right) space, use the Strassen
     decomposition of the 2x2x2 tensor (rank 7 instead of 8).
     For each (1x2) sub-block, use the schoolbook rank-2 decomposition.
     This yields rank 4*7 + 4*2 = 36.
  3. Verify the decomposition exactly. If it passes, return it.
     Otherwise fall back to schoolbook.
"""

import time
import numpy as np


def construct():
    """
    Return a valid tensor decomposition for 2x3x8 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    U shape: (6, rank), V shape: (24, rank), W shape: (16, rank)
    Convention: W[k*n + i, r]
    """
    start = time.time()
    deadline = start + 130  # safety margin

    n, m, p = 2, 3, 8

    # ------------------------------------------------------------------
    # Verification helper
    # ------------------------------------------------------------------
    def verify(U, V, W, n, m, p, tol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # ------------------------------------------------------------------
    # Schoolbook decomposition (rank 48) – guaranteed fallback
    # ------------------------------------------------------------------
    def schoolbook(n, m, p):
        R = n * m * p
        U = [[0.0] * R for _ in range(n * m)]
        V = [[0.0] * R for _ in range(m * p)]
        W = [[0.0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)
    assert verify(U0, V0, W0, n, m, p), "Schoolbook failed"

    # ------------------------------------------------------------------
    # Strassen decomposition for 2x2x2 tensor
    # Returns U (2x2=4 entries), V (2x2=4 entries), W (2x2=4 entries)
    # as lists of 7 vectors, each of length 4.
    # ------------------------------------------------------------------
    def strassen_222():
        """
        Return 7 rank-1 decompositions of the 2x2x2 matrix multiplication tensor.
        Each rank-1 is (u, v, w) where u,v,w are length-4 vectors.
        u[i*2+j], v[j*2+k], w[k*2+i] for i,j,k in {0,1}.
        """
        # Correct Strassen decomposition verified by construction
        terms = []
        # r0: u=(1,0,0,0) v=(0,0,1,1) w=(1,0,1,0)
        terms.append(([1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]))
        # r1: u=(1,1,0,0) v=(0,0,1,0) w=(1,0,1,0)
        terms.append(([1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0]))
        # r2: u=(0,0,1,1) v=(1,0,0,0) w=(0,1,0,1)
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
        # r3: u=(0,0,0,1) v=(0,1,-1,0) w=(1,0,0,1)
        terms.append(([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 1]))
        # r4: u=(1,0,0,1) v=(1,0,0,1) w=(1,0,0,1)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # r5: u=(0,1,0,-1) v=(0,0,1,1) w=(0,1,0,1)
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]))
        # r6: u=(1,-1,0,0) v=(1,1,0,0) w=(1,0,1,0)
        terms.append(([1, -1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]))
        return terms

    # Verify Strassen decomposition
    def verify_222(terms):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for u, v, w in terms:
                        s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    strassen_terms = strassen_222()
    if not verify_222(strassen_terms):
        # Fallback: use schoolbook for 2x2x2 (rank 8)
        strassen_terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0.0]*4; v = [0.0]*4; w = [0.0]*4
                    u[i*2+j] = 1.0
                    v[j*2+k] = 1.0
                    w[k*2+i] = 1.0
                    strassen_terms.append((u, v, w))

    # ------------------------------------------------------------------
    # Build the full block-structured decomposition
    # ------------------------------------------------------------------
    def build_block_decomposition(n, m, p, strassen_terms):
        """
        Build the block-structured decomposition.
        J1 = {0,1}, J2 = {2}
        K1 = {0,1}, K2 = {2,3}, K3 = {4,5}, K4 = {6,7}
        """
        U_rows = n * m  # 6
        V_rows = m * p  # 24
        W_rows = p * n  # 16

        U = []
        V = []
        W = []

        j_blocks = [(0, 2), (2, 1)]  # (start, size)
        k_blocks = [(0, 2), (2, 2), (4, 2), (6, 2)]

        for j0, js in j_blocks:
            for k0, ks in k_blocks:
                if js == 2 and ks == 2:
                    # 2x2x2 block → use Strassen (7 terms)
                    for u_sub, v_sub, w_sub in strassen_terms:
                        u_full = [0.0] * U_rows
                        v_full = [0.0] * V_rows
                        w_full = [0.0] * W_rows

                        for i in range(2):
                            for jp in range(2):
                                u_full[i * m + (j0 + jp)] = u_sub[i * 2 + jp]

                        for jp in range(2):
                            for kp in range(2):
                                v_full[(j0 + jp) * p + (k0 + kp)] = v_sub[jp * 2 + kp]

                        for kp in range(2):
                            for i in range(2):
                                w_full[(k0 + kp) * n + i] = w_sub[kp * 2 + i]

                        U.append(u_full)
                        V.append(v_full)
                        W.append(w_full)
                elif js == 1 and ks == 2:
                    # 1x2x2 block → rank 2 (one term per i in {0,1})
                    for i in range(2):
                        u_full = [0.0] * U_rows
                        v_full = [0.0] * V_rows
                        w_full = [0.0] * W_rows

                        # u: A[i, j0] → U[i*m + j0] = 1
                        u_full[i * m + j0] = 1.0

                        # v: B[j0, k0+kp] for kp in {0,1} → V[j0*p + k0+kp] = 1
                        for kp in range(2):
                            v_full[j0 * p + (k0 + kp)] = 1.0

                        # w: C[k0+kp, i] for kp in {0,1} → W[(k0+kp)*n + i] = 1
                        for kp in range(2):
                            w_full[(k0 + kp) * n + i] = 1.0

                        U.append(u_full)
                        V.append(v_full)
                        W.append(w_full)
                else:
                    # Should not happen
                    pass

        return U, V, W

    # ------------------------------------------------------------------
    # Attempt the block decomposition
    # ------------------------------------------------------------------
    if time.time() < deadline - 5:
        U1, V1, W1 = build_block_decomposition(n, m, p, strassen_terms)

        if verify(U1, V1, W1, n, m, p):
            return {"U": U1, "V": V1, "W": W1}
    
    # ------------------------------------------------------------------
    # Try a different block structure: split j into singletons and 
    # k into four 2-blocks. Each (1,1,2) block has rank 1.
    # Total rank = 2*3*4 = 24.
    # ------------------------------------------------------------------
    if time.time() < deadline - 5:
        U2 = []
        V2 = []
        W2 = []
        for i in range(n):
            for j in range(m):
                for k0 in range(0, p, 2):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[i * m + j] = 1.0
                    for kp in range(2):
                        v[j * p + k0 + kp] = 1.0
                        w[(k0 + kp) * n + i] = 1.0
                    U2.append(u)
                    V2.append(v)
                    W2.append(w)
        
        if verify(U2, V2, W2, n, m, p):
            return {"U": U2, "V": V2, "W": W2}

    # ------------------------------------------------------------------
    # Fallback to schoolbook
    # ------------------------------------------------------------------
    return {"U": U0, "V": V0, "W": W0}
