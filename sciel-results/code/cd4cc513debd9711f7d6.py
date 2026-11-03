"""Matrix multiplication 3x4x8 with rank-88 decomposition using Strassen blocks."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    Order of A elements: a00, a01, a10, a11
    Order of B elements: b00, b01, b10, b11
    Order of C elements: c00, c01, c10, c11
    
    Strassen products:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b01-b11)
    m4 = a11(b10-b00)
    m5 = (a00+a01)b11
    m6 = (a10-a00)(b00+b01)
    m7 = (a01-a11)(b10+b11)
    
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00: m1+m3+m5-m6
        [0, 0, 0, 0, 1, 0, 1],    # a01: m5+m7
        [0, 1, 0, 0, 0, 1, 0],    # a10: m2+m6
        [1, 1, 0, 1, 0, 0, -1]    # a11: m1+m2+m4-m7
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00: m1+m2-m4+m6
        [0, 0, 1, 0, 0, 1, 0],    # b01: m3+m6
        [0, 0, 0, 1, 0, 0, 1],    # b10: m4+m7
        [1, 0, -1, 0, 1, 0, 1]    # b11: m1-m3+m5+m7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00: m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],    # c01: m3+m5
        [0, 1, 0, 1, 0, 0, 0],    # c10: m2+m4
        [1, -1, 1, 0, 0, 1, 0]    # c11: m1-m2+m3+m6
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification."""
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    for (btype, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            r = 7
        elif btype == 'school1':
            r = 4
        else:
            raise ValueError("Unknown block type")
        blocks.append((btype, js, je, ks, ke, r))
        total_rank += r
    
    U = [[0] * total_rank for _ in range(n * m)]   
    V = [[0] * total_rank for _ in range(m * p)]   
    W = [[0] * total_rank for _ in range(p * n)]   
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for (btype, js, je, ks, ke, r) in blocks:
        j0, j1 = js, js + 1
        k0, k1 = ks, ks + 1
        
        if btype == 'strassen2':
            u_rows = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1
            ]
            v_rows = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1
            ]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
        elif btype == 'school1':
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [
                j0 * p + k0, j1 * p + k0,
                j0 * p + k1, j1 * p + k1
            ]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _default_tiling():
    """Default tiling: 8 Strassen blocks for i=0,1 and 8 schoolbook for i=2."""
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('strassen2', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    for jp in range(2):
        for kp in range(4):
            tiling.append(('school1', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _try_3x2x2_search():
    """Try to find a rank-3 decomposition for 3x2x2 matrix multiplication.
    
    Returns (U, V, W) if found, else None.
    U shape: (6, 3) for a_{i,j} with i in 0..2, j in 0..1
    V shape: (4, 3) for b_{j,k} with j in 0..1, k in 0..1  
    W shape: (6, 3) for c_{k,i} with k in 0..1, i in 0..2 (trace convention: W[k*n+i] where n=3)
    
    Conditions: for all i in 0..2, j in 0..1, k in 0..1:
        sum_c U[i*2+j][c] * V[j*2+k][c] * W[k*3+i][c] = 1
    """
    rng = np.random.RandomState(42)
    
    # Precompute index pairs
    idx_pairs = [(i, j, k) for i in range(3) for j in range(2) for k in range(2)]
    u_idx = [i * 2 + j for (i, j, k) in idx_pairs]
    v_idx = [j * 2 + k for (i, j, k) in idx_pairs]
    w_idx = [k * 3 + i for (i, j, k) in idx_pairs]
    
    # Try structured approach first: fix V and solve
    # Strategy: use V with simple structure, then solve for U, W
    
    # Try many random V matrices and solve for U, W
    for attempt in range(5000):
        # Random V with small integer/half-integer entries
        V = rng.choice([-2, -1, 0, 1, 2], size=(4, 3))
        V = V.astype(float)
        
        # For each c, we need: U[u_idx] * V[v_idx] * W[w_idx] summed over c = 1
        # This is: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c] = 1
        
        # Let's try: for each c, define M_c = outer-like structure
        # Actually, let's just do full random search with numpy vectorization
        
        U = rng.choice([-2, -1, 0, 1, 2], size=(6, 3))
        W = rng.choice([-2, -1, 0, 1, 2], size=(6, 3))
        
        # Vectorized check
        Uf = U.astype(float)
        Vf = V.astype(float)
        Wf = W.astype(float)
        
        # For each (i,j,k): sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*3+i,c]
        vals = np.zeros(len(idx_pairs))
        for c in range(3):
            vals += Uf[u_idx, c] * Vf[v_idx, c] * Wf[w_idx, c]
        
        if np.allclose(vals, 1.0, atol=1e-9):
            # Convert to list of lists
            U_out = [[U[i][c] for c in range(3)] for i in range(6)]
            V_out = [[V[i][c] for c in range(3)] for i in range(4)]
            W_out = [[W[i][c] for c in range(3)] for i in range(6)]
            return U_out, V_out, W_out
    
    # Try half-integer values too
    for attempt in range(5000):
        U = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(6, 3))
        V = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(4, 3))
        W = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(6, 3))
        
        vals = np.zeros(len(idx_pairs))
        for c in range(3):
            vals += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
        
        if np.allclose(vals, 1.0, atol=1e-9):
            U_out = [[U[i][c] for c in range(3)] for i in range(6)]
            V_out = [[V[i][c] for c in range(3)] for i in range(4)]
            W_out = [[W[i][c] for c in range(3)] for i in range(6)]
            return U_out, V_out, W_out
    
    return None


def _construct_3x4x8():
    """Construct the decomposition."""
    tiling = _default_tiling()
    result = _build_decomposition(tiling)
    
    # Try to improve by finding a rank-3 decomposition for 3x2x2 blocks
    # If found, we can replace the 8 schoolbook blocks (rank 32) with 8 rank-3 blocks (rank 24)
    # reducing total rank from 88 to 80.
    r3 = _try_3x2x2_search()
    if r3 is not None:
        u322, v322, w322 = r3
        result2 = _build_decomposition_with_3x2x2(tiling, u322, v322, w322)
        n, m, p = 3, 4, 8
        if _verify_decomposition(result2["U"], result2["V"], result2["W"], n, m, p):
            return result2
    
    return result


def _build_decomposition_with_3x2x2(tiling, u322, v322, w322):
    """Build a decomposition using Strassen for i=0,1 and 3x2x2 blocks for i=2."""
    n, m, p = 3, 4, 8
    # 8 Strassen blocks (rank 7 each) + 8 3x2x2 blocks (rank 3 each) = 56 + 24 = 80
    total_rank = 8 * 7 + 8 * 3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    col_offset = 0
    # Strassen blocks for i=0,1
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            for i in range(2):  # i = 0, 1
                u_rows = [i * m + j0, i * m + j1]
                v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
                w_rows = [k0 * n + i, k1 * n + i]
                # Strassen U is 4x7, V is 4x7, W is 4x7
                # U rows correspond to a_{i,j} for j in {j0,j1} -> 2 rows
                # But Strassen expects 2x2, so we map:
                # U2[0] -> a_{i,j0}, U2[1] -> a_{i,j1}
                # V2[0] -> b_{j0,k0}, V2[1] -> b_{j0,k1}, V2[2] -> b_{j1,k0}, V2[3] -> b_{j1,k1}
                # W2[0] -> c_{k0,i}, W2[1] -> c_{k1,i}, W2[2] -> c_{k0,i+1}... wait
                # W2 is for 2x2x2: W2[0]=c00, W2[1]=c01, W2[2]=c10, W2[3]=c11
                # For our block: c_{k0,i}, c_{k1,i} are the outputs
                # W2[0] -> c_{k0, i}, W2[1] -> c_{k1, i}? No...
                # In 2x2x2: c_{0,0}=W2[0], c_{0,1}=W2[1], c_{1,0}=W2[2], c_{1,1}=W2[3]
                # Here: row index in W is k*n+i, so c_{k0,i} = W[k0*n+i], c_{k1,i} = W[k1*n+i]
                # But Strassen W2 has 4 rows for 2x2 output. We only need 2 outputs (k0 and k1 for fixed i).
                # This doesn't work directly. Let me reconsider.
                
                # Actually, for a 2x2x2 Strassen block covering A rows i,i+1 and B cols k0,k1:
                # It computes c_{i,k0}, c_{i,k1}, c_{i+1,k0}, c_{i+1,k1}
                # So for i=0: c_{0,k0}, c_{0,k1}, c_{1,k0}, c_{1,k1}
                # For i=1: c_{1,k0}, c_{1,k1}, c_{2,k0}, c_{2,k1} -- but we only want i=0,1 for Strassen
                
                # Let me restructure: each Strassen block handles 2 consecutive i values.
                pass
            col_offset += 7
    
    # This approach is getting complicated. Let me just use the original tiling.
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s limit
    
    result = _construct_3x4x8()
    n, m, p = DIMENSIONS
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
