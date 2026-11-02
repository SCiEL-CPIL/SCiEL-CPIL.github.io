import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 140.0  # seconds, with 10s safety margin

# EVOLVE-BLOCK-START

def _build_rank11():
    """Build the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 2: (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 3: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 4: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 5: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 6: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 7: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 8: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 9: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 10: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 11: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=2, p=3, trials=10):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    for _ in range(trials):
        A = np.random.randint(-10, 11, (n, m))
        B = np.random.randint(-10, 11, (m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p), dtype=int)
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(rank):
                    for j in range(m):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_computed[i, k] = total
        if not np.array_equal(C_expected, C_computed):
            return False
    return True


def _check_rank10_over_gf(mod, max_attempts=50000, deadline=None):
    """
    Search for a rank-10 decomposition over a finite field.
    Uses the approach: fix U and V, then solve for W.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    rng = random.Random(12345)
    attempts = 0
    
    while (deadline is None or time.time() < deadline) and attempts < max_attempts:
        attempts += 1
        
        # Generate random U and V over the field
        U = [[rng.randint(0, mod-1) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.randint(0, mod-1) for _ in range(rank)] for _ in range(m*p)]
        
        # Compute the "partial product" tensor: C[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
        C = [[[0]*rank for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    total = 0
                    for j in range(m):
                        total = (total + U[i*m+j][r] * V[j*p+k][r]) % mod
                    C[i][k][r] = total
        
        # For each (i,k), we need to find r such that C[i,k,r] != 0
        # and set W[k*n+i, r] = inv(C[i,k,r])
        W = [[0]*rank for _ in range(p*n)]
        valid = True
        
        for i in range(n):
            for k in range(p):
                non_zero_rs = [r for r in range(rank) if C[i][k][r] != 0]
                if not non_zero_rs:
                    valid = False
                    break
                # Pick one non-zero r
                r = rng.choice(non_zero_rs)
                W[k*n + i][r] = pow(C[i][k][r], mod-2, mod)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify: for all (i,k), sum_r W[k*n+i, r] * C[i,k,r] should equal 1
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total = (total + W[k*n+i][r] * C[i][k][r]) % mod
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Convert to integers (they're already in [0, mod-1])
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Get the known rank-11 solution
    sol11 = _build_rank11()
    
    # Verify it works
    if not _verify_decomposition(sol11["U"], sol11["V"], sol11["W"], trials=5):
        # If verification fails (shouldn't happen), return as-is
        return sol11
    
    # Try to find rank-10 over small finite fields
    remaining = deadline - time.time()
    if remaining > 10:
        for mod in [7, 11, 13, 17]:
            if time.time() > deadline - 5:
                break
            sol10 = _check_rank10_over_gf(mod, max_attempts=20000, deadline=deadline - 5)
            if sol10 is not None:
                # Verify over integers
                if _verify_decomposition(sol10["U"], sol10["V"], sol10["W"], trials=5):
                    return sol10
    
    return sol11

# EVOLVE-BLOCK-END
