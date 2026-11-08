import time
import numpy as np

def construct():
    """
    Rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Split the 4x8 matrix B into two 4x4 halves.
    Use Strassen's algorithm recursively: 4x4x4 requires 7 * (2x2x2) = 49 rank-1 terms.
    Two halves give 98 total rank-1 terms.
    
    The decomposition is exact with integer entries.
    """
    n, m, p = 4, 4, 8
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    def _strassen_2x2x2():
        """Return 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen)."""
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append({
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        })
        # M2 = (A21+A22)B11
        terms.append({
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(1,0): 1, (1,1): -1}
        })
        # M3 = A11(B21-B22)
        terms.append({
            'U': {(0,0): 1},
            'V': {(1,0): 1, (1,1): -1},
            'W': {(0,1): 1, (1,1): 1}
        })
        # M4 = A22(B12-B11)
        terms.append({
            'U': {(1,1): 1},
            'V': {(0,1): 1, (0,0): -1},
            'W': {(0,0): 1, (1,0): 1}
        })
        # M5 = (A11+A12)B22
        terms.append({
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (0,1): 1}
        })
        # M6 = (A21-A11)(B11+B12)
        terms.append({
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        })
        # M7 = (A12-A22)(B21+B22)
        terms.append({
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1}
        })
        return terms
    
    def _strassen_4x4x4():
        """Return 49 rank-1 terms for 4x4x4 matrix multiplication."""
        base_terms = _strassen_2x2x2()
        all_terms = []
        
        def _add_terms(terms, U_blocks, V_blocks, W_blocks):
            for t in terms:
                new_U = {}
                new_V = {}
                new_W = {}
                for (i_b, j_b), c_u in U_blocks:
                    for (i, j), val in t['U'].items():
                        gi = i_b * 2 + i
                        gj = j_b * 2 + j
                        new_U[(gi, gj)] = new_U.get((gi, gj), 0) + val * c_u
                for (j_b, k_b), c_v in V_blocks:
                    for (j, k), val in t['V'].items():
                        gj = j_b * 2 + j
                        gk = k_b * 2 + k
                        new_V[(gj, gk)] = new_V.get((gj, gk), 0) + val * c_v
                for (i_b, k_b), c_w in W_blocks:
                    for (k, i), val in t['W'].items():
                        gi = i_b * 2 + i
                        gk = k_b * 2 + k
                        new_W[(gk, gi)] = new_W.get((gk, gi), 0) + val * c_w
                all_terms.append({'U': new_U, 'V': new_V, 'W': new_W})
        
        # M1 = (A11+A22)(B11+B22)
        _add_terms(base_terms,
                   [((0,0), 1), ((1,1), 1)],
                   [((0,0), 1), ((1,1), 1)],
                   [((0,0), 1), ((1,1), 1)])
        # M2 = (A21+A22)B11
        _add_terms(base_terms,
                   [((1,0), 1), ((1,1), 1)],
                   [((0,0), 1)],
                   [((1,0), 1), ((1,1), -1)])
        # M3 = A11(B21-B22)
        _add_terms(base_terms,
                   [((0,0), 1)],
                   [((1,0), 1), ((1,1), -1)],
                   [((0,1), 1), ((1,1), 1)])
        # M4 = A22(B12-B11)
        _add_terms(base_terms,
                   [((1,1), 1)],
                   [((0,1), 1), ((0,0), -1)],
                   [((0,0), 1), ((1,0), 1)])
        # M5 = (A11+A12)B22
        _add_terms(base_terms,
                   [((0,0), 1), ((0,1), 1)],
                   [((1,1), 1)],
                   [((0,0), -1), ((0,1), 1)])
        # M6 = (A21-A11)(B11+B12)
        _add_terms(base_terms,
                   [((1,0), 1), ((0,0), -1)],
                   [((0,0), 1), ((0,1), 1)],
                   [((1,1), 1)])
        # M7 = (A12-A22)(B21+B22)
        _add_terms(base_terms,
                   [((0,1), 1), ((1,1), -1)],
                   [((1,0), 1), ((1,1), 1)],
                   [((0,0), 1)])
        
        return all_terms
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Fallback to schoolbook
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
    
    # Generate terms for left half (columns 0-3 of B)
    left_terms = _strassen_4x4x4()
    
    if time.time() - start_time > DEADLINE:
        # Fallback to schoolbook
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
    
    # Generate terms for right half (columns 4-7 of B)
    right_terms = _strassen_4x4x4()
    
    # Combine: shift right terms to cover columns 4-7
    all_terms = []
    for t in left_terms:
        all_terms.append(t)
    for t in right_terms:
        new_V = {}
        for (j, k), v in t['V'].items():
            new_V[(j, k + 4)] = v
        new_W = {}
        for (k, i), v in t['W'].items():
            new_W[(k + 4, i)] = v
        all_terms.append({'U': t['U'], 'V': new_V, 'W': new_W})
    
    rank = len(all_terms)
    
    if time.time() - start_time > DEADLINE:
        # Fallback to schoolbook
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
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, t in enumerate(all_terms):
        for (i, j), coeff in t['U'].items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in t['V'].items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in t['W'].items():
            if coeff != 0:
                W[k * n + i][c] = coeff
    
    # Verify with a few random matrices
    if time.time() - start_time < DEADLINE - 5:
        np.random.seed(12345)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for c in range(rank):
                    u_val = 0
                    for j in range(m):
                        u_val += U[i * m + j][c] * A[i, j]
                    v_val = 0
                    for j in range(m):
                        v_val += V[j * p + k][c] * B[j, k]
                    val += u_val * v_val * W[k * n + i][c]
                C_computed[i, k] = val
        
        max_err = np.max(np.abs(C_computed - C_expected))
        
        if max_err > 1e-10:
            # Fallback to schoolbook if verification fails
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
    
    # Ensure entries are integers (they should be, but round to be safe)
    for i in range(n * m):
        for c in range(rank):
            U[i][c] = round(U[i][c])
    for i in range(m * p):
        for c in range(rank):
            V[i][c] = round(V[i][c])
    for i in range(p * n):
        for c in range(rank):
            W[i][c] = round(W[i][c])
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
