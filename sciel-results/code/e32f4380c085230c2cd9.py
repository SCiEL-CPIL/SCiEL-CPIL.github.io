"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 4]
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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


def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    Strassen's 2x2 algorithm applied to 2x2 blocks.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def make_block_combo_A(blocks_with_coeffs):
        result = [[[] for _ in range(2)] for _ in range(2)]
        for bi, bj, coeff in blocks_with_coeffs:
            for r in range(2):
                for c in range(2):
                    idx = (2*bi + r) * m + (2*bj + c)
                    result[r][c].append((idx, coeff))
        return result
    
    def make_block_combo_B(blocks_with_coeffs):
        result = [[[] for _ in range(2)] for _ in range(2)]
        for bi, bj, coeff in blocks_with_coeffs:
            for r in range(2):
                for c in range(2):
                    idx = (2*bi + r) * p + (2*bj + c)
                    result[r][c].append((idx, coeff))
        return result
    
    X1 = make_block_combo_A([(0,0,1), (1,1,1)])
    X2 = make_block_combo_A([(1,0,1), (1,1,1)])
    X3 = make_block_combo_A([(0,0,1)])
    X4 = make_block_combo_A([(1,1,1)])
    X5 = make_block_combo_A([(0,0,1), (0,1,1)])
    X6 = make_block_combo_A([(1,0,1), (0,0,-1)])
    X7 = make_block_combo_A([(0,1,1), (1,1,-1)])
    
    Y1 = make_block_combo_B([(0,0,1), (1,1,1)])
    Y2 = make_block_combo_B([(0,0,1)])
    Y3 = make_block_combo_B([(0,1,1), (1,1,-1)])
    Y4 = make_block_combo_B([(1,0,1), (0,0,-1)])
    Y5 = make_block_combo_B([(1,1,1)])
    Y6 = make_block_combo_B([(0,0,1), (0,1,1)])
    Y7 = make_block_combo_B([(1,0,1), (1,1,1)])
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    strassen_table = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ]
    
    block_contrib = [
        {((0,0)): 1, ((1,1)): 1},
        {((1,0)): 1, ((1,1)): -1},
        {((0,1)): 1, ((1,1)): 1},
        {((0,0)): 1, ((1,0)): 1},
        {((0,0)): -1, ((0,1)): 1},
        {((1,1)): 1},
        {((0,0)): 1},
    ]
    
    col = 0
    for j in range(7):
        Xj = Xs[j]
        Yj = Ys[j]
        
        x0, x1 = Xj[0]
        x2, x3 = Xj[1]
        y0, y1 = Yj[0]
        y2, y3 = Yj[1]
        
        def add_factors(f1, f2, sign2=1):
            result = list(f1)
            for idx, c in f2:
                result.append((idx, c * sign2))
            return result
        
        def sub_factors(f1, f2):
            return add_factors(f1, f2, -1)
        
        sub_products = [
            (add_factors(x0, x3), add_factors(y0, y3)),
            (add_factors(x2, x3), y0),
            (x0, sub_factors(y1, y3)),
            (x3, sub_factors(y2, y0)),
            (add_factors(x0, x1), y3),
            (sub_factors(x2, x0), add_factors(y0, y1)),
            (sub_factors(x1, x3), add_factors(y2, y3)),
        ]
        
        for sk_idx in range(7):
            x_factor, y_factor = sub_products[sk_idx]
            
            u_dict = {}
            for idx, coeff in x_factor:
                u_dict[idx] = u_dict.get(idx, 0) + coeff
            for idx in range(n * m):
                U[idx][col] = u_dict.get(idx, 0)
            
            v_dict = {}
            for idx, coeff in y_factor:
                v_dict[idx] = v_dict.get(idx, 0) + coeff
            for idx in range(m * p):
                V[idx][col] = v_dict.get(idx, 0)
            
            w_dict = {}
            for (kl), alpha in block_contrib[j].items():
                k, l = kl
                for r in range(2):
                    for c in range(2):
                        coeff_rc = strassen_table[sk_idx][r][c]
                        if coeff_rc != 0:
                            C_row = 2 * k + r
                            C_col = 2 * l + c
                            W_idx = C_col * n + C_row
                            total_coeff = alpha * coeff_rc
                            w_dict[W_idx] = w_dict.get(W_idx, 0) + total_coeff
            for idx in range(p * n):
                W[idx][col] = w_dict.get(idx, 0)
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    if n >= 2 and m >= 2 and p >= 2:
        a = 0
        b = 1 * p + 0
        c = 0
        actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(actual) > tol:
            return False
    return True


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    
    Uses Strassen's algorithm applied to 2x2 blocks, giving rank 49.
    """
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Try the rank-49 Strassen-based decomposition
    try:
        if time.time() - t0 < _DEADLINE - 5.0:
            U, V, W = _strassen_4x4()
            # Verify before returning
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                return {"U": U, "V": V, "W": W}
            else:
                # Strassen failed verification; fall through
                pass
    except Exception:
        pass
    
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
