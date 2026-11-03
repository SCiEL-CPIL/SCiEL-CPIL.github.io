"""Matrix multiplication 3x5x6 rank-minimal decomposition search."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_deadline = 120.0
    start_time = time.time()
    safety_margin = 10.0  # seconds before deadline to stop searching

    def remaining_time():
        return total_deadline - safety_margin - (time.time() - start_time)

    # Always return a valid schoolbook decomposition as fallback
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

    # Verify decomposition validity using exact Fraction arithmetic
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        if len(U) != n * m: return False
        if len(V) != m * p: return False
        if len(W) != p * n: return False
        for r in range(rank):
            if len(U[r]) != rank or len(V[r]) != rank or len(W[r]) != rank:
                return False
        # Check tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = Fraction(1)
                    actual = Fraction(0)
                    for r in range(rank):
                        actual += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if actual != expected:
                        return False
        return True

    def verify_fast(U, V, W):
        """Fast float verification for candidate solutions"""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(actual - 1.0) > 1e-6:
                        return False
        return True

    # GF(2) support pattern search
    def gf2_search(rank, max_restarts=50):
        """
        Search for a support pattern over GF(2) that satisfies the tensor
        equations modulo 2, then attempt to lift to exact rational factors.
        """
        if remaining_time() < 5.0:
            return None

        # Target: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] ≡ 1 (mod 2)
        # We search for binary supports (non-zero positions) that work mod 2,
        # then try to find integer/half-integer values that work exactly.

        rng = np.random.RandomState(12345)

        for restart in range(max_restarts):
            if remaining_time() < 3.0:
                break

            # Initialize random binary support patterns
            U_bin = np.zeros((n*m, rank), dtype=np.uint8)
            V_bin = np.zeros((m*p, rank), dtype=np.uint8)
            W_bin = np.zeros((p*n, rank), dtype=np.uint8)

            # Each column r has some number of non-zeros in each factor
            # Start with moderate density
            for r in range(rank):
                # Randomly place 1-3 non-zeros in each factor
                for _ in range(rng.randint(1, 4)):
                    U_bin[rng.randint(0, n*m), r] = 1
                for _ in range(rng.randint(1, 4)):
                    V_bin[rng.randint(0, m*p), r] = 1
                for _ in range(rng.randint(1, 4)):
                    W_bin[rng.randint(0, p*n), r] = 1

            # Check if this support pattern works over GF(2)
            # For each (i,j,k): XOR of (U[i,j,r] & V[j,k,r] & W[k,i,r]) over r should be 1
            gf2_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        parity = 0
                        for r in range(rank):
                            if U_bin[i*m+j, r] & V_bin[j*p+k, r] & W_bin[k*n+i, r]:
                                parity ^= 1
                        if parity != 1:
                            gf2_valid = False
                            break
                    if not gf2_valid:
                        break
                if not gf2_valid:
                    break

            if not gf2_valid:
                # Try to repair: for each failing (i,j,k), flip a random bit
                for _ in range(100):  # max repair attempts
                    if remaining_time() < 3.0:
                        break

                    # Find a failing triple
                    fail_found = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                parity = 0
                                for r in range(rank):
                                    if U_bin[i*m+j, r] & V_bin[j*p+k, r] & W_bin[k*n+i, r]:
                                        parity ^= 1
                                if parity != 1:
                                    # Flip a random bit in one of the three factors
                                    factor = rng.randint(0, 3)
                                    r_flip = rng.randint(0, rank)
                                    if factor == 0:
                                        U_bin[i*m+j, r_flip] ^= 1
                                    elif factor == 1:
                                        V_bin[j*p+k, r_flip] ^= 1
                                    else:
                                        W_bin[k*n+i, r_flip] ^= 1
                                    fail_found = True
                                    break
                            if fail_found:
                                break
                        if fail_found:
                            break

                    if not fail_found:
                        break

                    # Re-check
                    gf2_valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                parity = 0
                                for r in range(rank):
                                    if U_bin[i*m+j, r] & V_bin[j*p+k, r] & W_bin[k*n+i, r]:
                                        parity ^= 1
                                if parity != 1:
                                    gf2_valid = False
                                    break
                            if not gf2_valid:
                                break
                        if not gf2_valid:
                            break

                    if gf2_valid:
                        break

            if not gf2_valid:
                continue

            # GF(2) pattern found! Now try to lift to exact rational factors.
            # The GF(2) pattern tells us where non-zeros can be.
            # We need to find integer/half-integer values such that the
            # tensor equations hold exactly.

            # Strategy: use the support pattern to set up a system of equations.
            # For each (i,j,k): sum_r u[i,j,r]*v[j,k,r]*w[k,i,r] = 1
            # where u,v,w are 0 or 1/2 or -1/2 or 1 or -1 based on support.

            # Simpler approach: try all-ones on the support first
            U_try = U_bin.astype(int)
            V_try = V_bin.astype(int)
            W_try = W_bin.astype(int)

            if verify_fast(U_try.tolist(), V_try.tolist(), W_try.tolist()):
                return (U_try.tolist(), V_try.tolist(), W_try.tolist())

            # Try with some -1 values to fix sign issues
            # For each column, try different sign assignments
            best_solution = None
            for sign_attempt in range(20):
                if remaining_time() < 3.0:
                    break

                U_s = U_bin.copy().astype(int)
                V_s = V_bin.copy().astype(int)
                W_s = W_bin.copy().astype(int)

                # Randomly flip signs of some entries
                for r in range(rank):
                    for idx in range(n*m):
                        if U_bin[idx, r] and rng.random() < 0.3:
                            U_s[idx, r] = -1
                    for idx in range(m*p):
                        if V_bin[idx, r] and rng.random() < 0.3:
                            V_s[idx, r] = -1
                    for idx in range(p*n):
                        if W_bin[idx, r] and rng.random() < 0.3:
                            W_s[idx, r] = -1

                if verify_fast(U_s.tolist(), V_s.tolist(), W_s.tolist()):
                    return (U_s.tolist(), V_s.tolist(), W_s.tolist())

                # Try half-integer values
                U_h = U_bin.copy().astype(float)
                V_h = V_bin.copy().astype(float)
                W_h = W_bin.copy().astype(float)

                for r in range(rank):
                    for idx in range(n*m):
                        if U_bin[idx, r]:
                            if rng.random() < 0.5:
                                U_h[idx, r] = 0.5
                            else:
                                U_h[idx, r] = 1.0
                    for idx in range(m*p):
                        if V_bin[idx, r]:
                            if rng.random() < 0.5:
                                V_h[idx, r] = 0.5
                            else:
                                V_h[idx, r] = 1.0
                    for idx in range(p*n):
                        if W_bin[idx, r]:
                            if rng.random() < 0.5:
                                W_h[idx, r] = 0.5
                            else:
                                W_h[idx, r] = 1.0

                if verify_fast(U_h.tolist(), V_h.tolist(), W_h.tolist()):
                    return (U_h.tolist(), V_h.tolist(), W_h.tolist())

        return None

    # Block decomposition approach: exploit structure of 3x5x6
    def block_decomposition():
        """
        Try to decompose 3x5x6 by splitting dimensions.
        6 = 2*3, so split k into k1 (size 2) and k2 (size 3).
        C[i, k1, k2] = sum_j A[i,j] * B[j, k1, k2]
        This is equivalent to multiplying a 3x5 matrix by a 5x(2*3) matrix.
        We can view this as a block structure.
        """
        if remaining_time() < 10.0:
            return None

        # Approach: treat the 5x6 matrix B as having structure
        # B[j, k] where k = 2*k1 + k2, k1 in {0,1}, k2 in {0,1,2}
        # C[i, k1, k2] = sum_j A[i,j] * B[j, 2*k1+k2]
        #
        # If we can find a decomposition where the k2 dimension uses a
        # 3x3x3 Winograd-like trick and k1 adds 2 more terms...
        #
        # Actually, let's try a direct structured approach:
        # Split j into blocks and use Kronecker products.

        # For now, try a rank-45 decomposition (half of schoolbook)
        # by exploiting that 5*6 = 30 and we might save factors.

        # Try: rank = 3*5*6 / 2 = 45 (if we can pair up terms)
        # This is speculative; let's try a more principled approach.

        # Use the fact that we can share computation across the k dimension.
        # If B has structure B[j,k] = sum_l X[j,l] * Y[l,k], then
        # C[i,k] = sum_j A[i,j] * sum_l X[j,l] * Y[l,k]
        #         = sum_l (sum_j A[i,j]*X[j,l]) * Y[l,k]
        # This is like rank-l factorization of B, giving rank = n*l*p.
        # For l=1: rank = 3*1*6 = 18 (but only works if B has rank 1)
        # For general B, we need l=5, giving rank = 3*5*6 = 90 (schoolbook).

        # So the block approach doesn't directly help unless B has structure.
        # Let's try a different angle: use Strassen-like tricks on sub-blocks.

        return None

    # Main search: try ranks from lower bound upward
    def search_rank_minimal():
        """Search for minimal rank using GF(2) support search and structured methods"""
        best_rank = n * m * p  # schoolbook
        best_solution = None

        # Lower bound for matrix multiplication tensor rank:
        # R >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
        # But also R >= n*p = 18 (from considering C as n*p matrix)
        # And R >= m*p = 30 (from considering B as m*p matrix)
        # So R >= 30

        lower_bound = 30

        # Try structured approaches first (faster)
        if remaining_time() > 30:
            sol = block_decomposition()
            if sol is not None:
                return sol

        # Try GF(2) search for various ranks
        for rank in range(lower_bound, best_rank + 1):
            if remaining_time() < 15.0:
                break

            sol = gf2_search(rank, max_restarts=20)
            if sol is not None:
                U, V, W = sol
                # Final exact verification
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
                else:
                    # Float check passed but exact failed; skip
                    continue

        return None

    # Try the search
    result = search_rank_minimal()
    if result is not None:
        return result

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
