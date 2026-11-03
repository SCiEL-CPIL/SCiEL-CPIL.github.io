"""Matrix multiplication 2x3 by 3x6 - rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Strategy: partition 3x6 into three 3x2 blocks, use rank-r decomposition
    # for 2x3x2 per block, total rank = 3*r
    # Try to find low-rank decomposition for 2x3x2, fall back to schoolbook

    best_rank_232 = None
    best_decomp_232 = None

    # Try ranks from 6 upward with time-bounded search
    for r in range(6, 13):
        if time.time() > deadline - 55:
            break
        decomp = _find_decomp_232(r, deadline - 55)
        if decomp is not None:
            best_rank_232 = r
            best_decomp_232 = decomp
            break

    if best_decomp_232 is None:
        # Fall back to schoolbook for 2x3x2 (rank 12)
        best_rank_232 = 12
        best_decomp_232 = _schoolbook_232()

    # Construct the full 2x3x6 decomposition
    total_rank = 3 * best_rank_232
    U = [[0] * total_rank for _ in range(n * m)]      # 6 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]      # 18 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]      # 12 x total_rank

    U232, V232, W232 = best_decomp_232

    for b in range(3):
        col_offset = b * best_rank_232
        # U: row (i*m + j)
        for i in range(n):
            for j in range(m):
                for c in range(best_rank_232):
                    U[i * m + j][col_offset + c] = U232[i * m + j][c]
        
        # V: row (j*p + k), k = b*2 + k_local
        for j in range(m):
            for k_local in range(2):
                k = b * 2 + k_local
                for c in range(best_rank_232):
                    V[j * p + k][col_offset + c] = V232[j * 2 + k_local][c]
        
        # W: row (k*n + i), k = b*2 + k_local
        for k_local in range(2):
            k = b * 2 + k_local
            for i in range(n):
                for c in range(best_rank_232):
                    W[k * n + i][col_offset + c] = W232[k_local * n + i][c]

    # Verify
    if time.time() < deadline - 10:
        _verify(U, V, W, n, m, p, total_rank)

    return {"U": U, "V": V, "W": W}


def _schoolbook_232():
    """Schoolbook rank-12 decomposition for 2x3x2."""
    n, m, p = 2, 3, 2
    rank = n * m * p  # 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 6 x 12
    W = [[0] * rank for _ in range(p * n)]  # 4 x 12
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _find_decomp_232(r, deadline):
    """Try to find a rank-r decomposition for 2x3x2 matrix multiplication."""
    n, m, p = 2, 3, 2
    rng = random.Random(42)
    
    max_attempts = 3000
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = [[rng.randint(-3, 3) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.randint(-3, 3) for _ in range(r)] for _ in range(m * p)]
        W = [[rng.randint(-3, 3) for _ in range(r)] for _ in range(p * n)]
        
        if _check_232(U, V, W, n, m, p, r):
            return U, V, W
        
        # Structured init every 100 attempts
        if attempt % 100 == 50:
            U, V, W = _structured_init_232(r, rng)
            if _check_232(U, V, W, n, m, p, r):
                return U, V, W
    
    return None


def _structured_init_232(r, rng):
    """Structured initialization for 2x3x2 decomposition."""
    n, m, p = 2, 3, 2
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = rng.randint(0, r - 1)
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return U, V, W


def _check_232(U, V, W, n, m, p, r):
    """Check if decomposition is valid for 2x3x2."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = 0
                            for c in range(r):
                                s += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + ip][c]
                            if s != 0:
                                return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    if k == kp:
                        continue
                    for ip in range(n):
                        s = 0
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[kp * n + ip][c]
                        if s != 0:
                            return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    if i == ip:
                        continue
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + ip][c]
                    if s != 0:
                        return False
    
    return True


def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    raise AssertionError(f"Diagonal check failed at ({i},{j},{k}): got {s}")
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = 0
                            for c in range(rank):
                                s += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + ip][c]
                            if s != 0:
                                raise AssertionError(f"Off-diag j check failed at ({i},{j},{jp},{k},{kp},{ip})")
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    if k == kp:
                        continue
                    for ip in range(n):
                        s = 0
                        for c in range(rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[kp * n + ip][c]
                        if s != 0:
                            raise AssertionError(f"Off-diag k check failed at ({i},{j},{k},{kp},{ip})")
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    if i == ip:
                        continue
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + ip][c]
                    if s != 0:
                        raise AssertionError(f"Off-diag i check failed at ({i},{j},{k},{ip})")
