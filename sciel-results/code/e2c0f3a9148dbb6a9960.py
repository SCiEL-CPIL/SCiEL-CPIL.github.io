"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy (keeping the Strassen-on-2x2-sub-blocks design and refining within its family):
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10).
    Partition the output columns (10) into five 2-wide blocks.
    For each 2x2 x 2x2 sub-multiplication, use the classic rank-7 Strassen algorithm.
    Subtotal: 5 * 7 = 35.
  - Part 2: row 2 of A (2x1) times row 2 of B (1x10).
    Schoolbook: 2 * 1 * 10 = 20.
  Total rank: 55.

Refinements within this family:
  1.  The Strassen construction is applied uniformly to all five 2-wide output blocks.
  2.  We verify the decomposition numerically (exact integer/half-integer arithmetic via
      random integer inputs) before returning, to guard against indexing errors.
  3.  We check the deadline and, if exceeded, still return the constructed result
      (construction is O(1) and completes in microseconds, so this is a safety net).
  4.  Vectorized verification for speed.
  5.  Attempt to reduce rank by checking if any term is linearly dependent on others.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145.0  # Safety margin before 150s kill


def _verify_decomposition(U, V, W, n, m, p, num_trials=10, seed=42):
    """Verify decomposition with random integer inputs using vectorized ops."""
    rng = np.random.RandomState(seed)
    rank = U.shape[1]
    for _ in range(num_trials):
        A = rng.randint(-10, 11, size=(n, m)).astype(float)
        B = rng.randint(-10, 11, size=(m, p)).astype(float)
        C_direct = A @ B

        # Vectorized decomposition evaluation
        # For each rank term r: scalar_r = sum(U[:,r] * A_flat) * sum(V[:,r] * B_flat)
        # C = sum_r scalar_r * W[:,r] reshaped
        A_flat = A.flatten()  # shape (n*m,)
        B_flat = B.flatten()  # shape (m*p,)
        # U is (n*m, rank), so U.T @ A_flat gives (rank,)
        u_sums = U.T @ A_flat  # (rank,)
        v_sums = V.T @ B_flat  # (rank,)
        scalars = u_sums * v_sums  # (rank,)
        # W is (p*n, rank), so W @ scalars gives (p*n,)
        C_flat = W @ scalars  # (p*n,)
        C_decomp = C_flat.reshape(p, n).T  # (n, p)

        err = np.max(np.abs(C_decomp - C_direct))
        if err > 1e-9:
            return False, err
    return True, 0.0


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by checking if any term is linearly dependent on others.
    Returns the reduced matrices if successful.
    """
    rank = U.shape[1]
    
    # For each column r, check if removing it still gives correct tensor
    def check_valid_without(U_m, V_m, W_m, r_to_remove):
        """Check if decomposition without column r is still valid."""
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_m.shape[1]):
                        if r == r_to_remove:
                            continue
                        s += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try removing each column
    for r in range(rank):
        if time.time() > deadline:
            break
        if check_valid_without(U, V, W, r):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            # Try one more removal
            if time.time() < deadline - 5:
                for r2 in range(rank):
                    if time.time() > deadline:
                        break
                    if check_valid_without(U, V, W, r2):
                        U = np.delete(U, r2, axis=1)
                        V = np.delete(V, r2, axis=1)
                        W = np.delete(W, r2, axis=1)
                        rank -= 1
                        break
            break
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Check deadline early - construction is O(1) but guard anyway
    if time.time() - start_time > DEADLINE:
        # Fallback: return a trivial (schoolbook) decomposition
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Partition middle dim: 3 = 2 + 1
    # Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10) -> 5 Strassen blocks of 2x2x2
    # Part 2: row 2 of A (2x1) times row 2 of B (1x10) -> schoolbook rank 20

    rank = 35 + 20  # 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        # Contributes to C[0,k0] and C[1,k1]
        U[0 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        # Contributes to C[0,k1] and C[1,k1]
        U[1 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        W[k1 * 2 + 1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        # Contributes to C[0,k1] and C[1,k1]
        U[0 * 3 + 0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        V[1 * 10 + k1, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        # Contributes to C[0,k0] and C[1,k0]
        U[1 * 3 + 1, idx] = 1.0
        V[1 * 10 + k0, idx] = 1.0
        V[0 * 10 + k0, idx] = -1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        # Contributes to C[0,k0] and C[0,k1]
        U[0 * 3 + 0, idx] = 1.0
        U[0 * 3 + 1, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        # Contributes to C[1,k1]
        U[1 * 3 + 0, idx] = 1.0
        U[0 * 3 + 0, idx] = -1.0
        V[0 * 10 + k0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        # Contributes to C[0,k0]
        U[0 * 3 + 1, idx] = 1.0
        U[1 * 3 + 1, idx] = -1.0
        V[1 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    # Verify correctness before attempting reduction
    ok, err = _verify_decomposition(U, V, W, n, m, p, num_trials=3, seed=42)
    if not ok:
        # Fallback to schoolbook if Strassen decomposition is incorrect
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, DEADLINE)

    # Final verification
    ok, err = _verify_decomposition(U, V, W, n, m, p, num_trials=3, seed=42)
    if not ok:
        # If reduction broke something, fall back to original rank-55
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for block in range(5):
            k0 = 2 * block
            k1 = 2 * block + 1
            U[0, idx] = 1.0; U[4, idx] = 1.0
            V[k0, idx] = 1.0; V[10+k1, idx] = 1.0
            W[k0, idx] = 1.0; W[2*k1+1, idx] = 1.0
            idx += 1
            U[3, idx] = 1.0; U[4, idx] = 1.0
            V[k0, idx] = 1.0
            W[k0+1, idx] = 1.0; W[2*k1+1, idx] = -1.0
            idx += 1
            U[0, idx] = 1.0
            V[10+k1, idx] = 1.0; V[20+k1, idx] = -1.0
            W[2*k1, idx] = 1.0; W[2*k1+1, idx] = 1.0
            idx += 1
            U[4, idx] = 1.0
            V[20+k0, idx] = 1.0; V[10+k0, idx] = -1.0
            W[k0, idx] = 1.0; W[k0+1, idx] = 1.0
            idx += 1
            U[0, idx] = 1.0; U[1, idx] = 1.0
            V[20+k1, idx] = 1.0
            W[k0, idx] = -1.0; W[2*k1, idx] = 1.0
            idx += 1
            U[3, idx] = 1.0; U[0, idx] = -1.0
            V[k0, idx] = 1.0; V[10+k1, idx] = 1.0
            W[2*k1+1, idx] = 1.0
            idx += 1
            U[1, idx] = 1.0; U[4, idx] = -1.0
            V[20+k0, idx] = 1.0; V[20+k1, idx] = 1.0
            W[k0, idx] = 1.0
            idx += 1
        for i in range(2):
            for k in range(10):
                U[i*3+2, idx] = 1.0
                V[20+k, idx] = 1.0
                W[k*2+i, idx] = 1.0
                idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    # Verify correctness with random integer inputs (exact arithmetic)
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]

        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
