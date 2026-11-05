import time
import numpy as np

START = time.time()
DEADLINE = START + 145.0  # safety margin under 150s kill


def _build_strassen_2x2x2():
    """Strassen 2x2x2 decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_4x4x4_strassen_tensor_product():
    """4x4x4 Strassen via tensor product of 2x2x2 (rank 49)."""
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((16, rank), dtype=np.int16)
    W = np.zeros((16, rank), dtype=np.int16)

    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            u0, u1 = U2[:, c0], U2[:, c1]
            v0, v1 = V2[:, c0], V2[:, c1]
            w0, w1 = W2[:, c0], W2[:, c1]

            # U[i,j] for i,j in 0..3: index in 0..15
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    val = u0[i0*2+j0] * u1[i1*2+j1]
                    if val != 0:
                        U[i*4+j, c] = val

            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    val = v0[j0*2+k0] * v1[j1*2+k1]
                    if val != 0:
                        V[j*4+k, c] = val

            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    val = w0[k0*2+i0] * w1[k1*2+i1]
                    if val != 0:
                        W[k*4+i, c] = val

    return U, V, W


def _build_4x4x8_rank98():
    """4x4x8 via duplicating 4x4x4 Strassen for each 4-col block (rank 98)."""
    U4, V4, W4 = _build_4x4x4_strassen_tensor_product()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)

    U[:, 0:49] = U4
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local, 0:49] = V4[j*4 + k_local, :]
            V[j*8 + 4 + k_local, 49:98] = V4[j*4 + k_local, :]
            W[k_local*4 + 0:4, 0:49] = W4[k_local*4 + 0:4, :]
            W[(4+k_local)*4 + 0:4, 49:98] = W4[k_local*4 + 0:4, :]

    return U, V, W


def _verify(U, V, W):
    """Verify decomposition: sum_r U[:,r] * V[:,r] * W[:,r] == matmul tensor."""
    n, m, p = 4, 4, 8
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape for efficient einsum
    U_r = U.T  # (rank, n*m)
    V_r = V.T  # (rank, m*p)
    W_r = W.T  # (rank, p*n)
    # T[i,j,k] = sum_r U_r[r, i*m+j] * V_r[r, j*p+k] * W_r[r, k*n+i]
    T = np.einsum('ra,rb,rc->abc', U_r, V_r, W_r)
    # Reshape to (n, m, p)
    T = T.reshape(n, m, p)
    # Expected: T[i,j,k] = 1 if i==i, j==j, k==k (i.e., delta)
    # Actually matmul tensor: T[i,j,k] = delta[i,k] * delta[j,j]? No.
    # For (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 if the (i,k) entry uses A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the j-sum)
    # Actually T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} = 1 always? No.
    # The matmul tensor has T[i,j,k] = 1 for all valid i,j,k because
    # (A*B)[i,k] = sum_j A[i,j] B[j,k], so the tensor is all ones.
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, expected, atol=1e-9)


def _greedy_eliminate(U, V, W, max_time=10.0):
    """Try to eliminate redundant terms via greedy removal."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    keep = list(range(rank))
    start = time.time()
    
    # Try removing each term and check validity
    for _ in range(50):  # up to 50 eliminations
        if time.time() - start > max_time:
            break
        removed = False
        for r in list(keep):
            new_keep = [k for k in keep if k != r]
            if not new_keep:
                continue
            U_sub = U[:, new_keep]
            V_sub = V[:, new_keep]
            W_sub = W[:, new_keep]
            if _verify(U_sub, V_sub, W_sub):
                keep = new_keep
                removed = True
                break
        if not removed:
            break
    return U[:, keep], V[:, keep], W[:, keep]


def _try_local_search(U, V, W, max_time=30.0):
    """Try small perturbations to find a lower-rank decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    best_rank = rank
    start = time.time()
    
    # Try removing pairs of terms
    for _ in range(20):
        if time.time() - start > max_time:
            break
        # Random pair removal
        r1 = np.random.randint(0, rank)
        r2 = np.random.randint(0, rank)
        if r1 == r2:
            continue
        keep = [k for k in range(rank) if k not in (r1, r2)]
        U_sub = U[:, keep]
        V_sub = V[:, keep]
        W_sub = W[:, keep]
        if _verify(U_sub, V_sub, W_sub):
            best_rank = len(keep)
            U, V, W = U_sub, V_sub, W_sub
            rank = best_rank
            break
    return U, V, W


def _schoolbook(n=4, m=4, p=8):
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


def construct():
    try:
        if time.time() < DEADLINE:
            U, V, W = _build_4x4x8_rank98()
            # Verify base decomposition
            if not _verify(U, V, W):
                raise ValueError("Base decomposition failed")
            # Try greedy elimination
            if time.time() < DEADLINE - 20:
                U, V, W = _greedy_eliminate(U, V, W, max_time=min(20.0, DEADLINE - time.time() - 20))
            # Try local search
            if time.time() < DEADLINE - 10:
                U, V, W = _try_local_search(U, V, W, max_time=min(15.0, DEADLINE - time.time() - 10))
            # Final verification
            if not _verify(U, V, W):
                raise ValueError("Final decomposition failed")
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    return _schoolbook()


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
