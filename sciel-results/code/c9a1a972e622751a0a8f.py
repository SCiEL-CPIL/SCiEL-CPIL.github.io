import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 145
    n, m, p = DIMENSIONS  # 3, 3, 7

    def time_left():
        return DEADLINE - (time.time() - t_start)

    def schoolbook():
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

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def verify_333(U3, V3, W3):
        r = len(U3[0])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for rr in range(r):
                        s += U3[i*3+j][rr] * V3[j*3+k][rr] * W3[k*3+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def transpose_w_factors(factors):
        """
        factors: list of (u, v, w) where u,v,w are lists of 9 floats.
        Returns list of (u, v, w_t) where w_t is w transposed.
        w is indexed by i*3+k in standard row-major.
        w_t is indexed by k*3+i.
        So w_t[k*3+i] = w[i*3+k].
        """
        new_factors = []
        for u, v, w in factors:
            w_t = [0] * 9
            for i in range(3):
                for k in range(3):
                    w_t[k*3+i] = w[i*3+k]
            new_factors.append((u, v, w_t))
        return new_factors

    def get_laderman_23():
        # Hardcoded Laderman factors from current program
        # Assumed to be in row-major order for U, V, W
        factors = [
            ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
            ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
            ([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
            ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
            ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
            ([0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
            ([0,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
            ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
            ([0,0,0, 1,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
            ([0,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
            ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
            ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
            ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
            ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
            ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
            ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
            ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
            ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
            ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ]
        return factors

    laderman_factors = None
    if time_left() > 5:
        factors = get_laderman_23()
        
        # Try original
        U3 = [f[0] for f in factors]
        V3 = [f[1] for f in factors]
        W3 = [f[2] for f in factors]
        if verify_333(U3, V3, W3):
            laderman_factors = factors
        else:
            # Try transposed W
            factors_t = transpose_w_factors(factors)
            U3_t = [f[0] for f in factors_t]
            V3_t = [f[1] for f in factors_t]
            W3_t = [f[2] for f in factors_t]
            if verify_333(U3_t, V3_t, W3_t):
                laderman_factors = factors_t
            else:
                laderman_factors = None

    if laderman_factors is not None:
        total_rank = 55
        U = [[0] * total_rank for _ in range(9)]
        V = [[0] * total_rank for _ in range(21)]
        W = [[0] * total_rank for _ in range(21)]
        
        # Block 1: Laderman for B columns 0-2
        for r in range(23):
            u, v, w = laderman_factors[r]
            for i in range(9):
                U[i][r] = u[i]
                # V index: j*7 + k. Here k is 0,1,2.
                # v is indexed by j*3+k.
                # We need to map v[j*3+k] to V[j*7+k].
                # Since v is a flat list of 9, we can just iterate.
                # But careful: v[i] where i = j*3+k.
                # We want V[j*7+k][r] = v[j*3+k].
                # So for each index idx in 0..8:
                # j = idx // 3, k = idx % 3.
                # V[j*7 + k][r] = v[idx].
                
                # Let's do this mapping explicitly to be safe.
        
        # Re-doing the mapping properly
        for r in range(23):
            u, v, w = laderman_factors[r]
            for idx in range(9):
                U[idx][r] = u[idx]
                
                j = idx // 3
                k = idx % 3
                V[j*7 + k][r] = v[idx]
                
                # w is indexed by k*3+i (because we used transposed or verified version).
                # Wait, verify_333 uses W3[k*3+i].
                # So w[idx] corresponds to W3[k*3+i] where idx = k*3+i?
                # In verify_333: W3[k*3+i][rr].
                # So the list W3[rr] is indexed by k*3+i.
                # So w[idx] is the value for index idx in the W vector.
                # We need to place it in W[global_idx][r].
                # global_idx = k*3 + i.
                # So W[idx][r] = w[idx].
                W[idx][r] = w[idx]

        # Block 2: Laderman for B columns 3-5
        for r in range(23):
            u, v, w = laderman_factors[r]
            col = 23 + r
            for idx in range(9):
                U[idx][col] = u[idx]
                
                j = idx // 3
                k = idx % 3
                # B column is 3 + k
                V[j*7 + (3 + k)][col] = v[idx]
                
                # W index: k_global = 3 + k.
                # W global index: k_global * 3 + i = (3+k)*3 + i = 9 + k*3 + i.
                # w[idx] is for local k*3+i.
                # So W[9 + idx][col] = w[idx].
                W[9 + idx][col] = w[idx]

        # Block 3: Schoolbook for B column 6
        for i in range(3):
            for j in range(3):
                col = 46 + i * 3 + j
                U[i * 3 + j][col] = 1
                V[j * 7 + 6][col] = 1
                W[6 * 3 + i][col] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    # Fallback
    result = schoolbook()
    if not verify(result['U'], result['V'], result['W']):
        raise RuntimeError("Schoolbook verification failed!")
    return result
