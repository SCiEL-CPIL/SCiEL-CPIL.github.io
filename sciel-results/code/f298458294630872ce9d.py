"""Matrix multiplication (2,3,9) rank search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _verify_block(U_block, V_block, W_block, r):
    """Verify a 2x3*3x3 decomposition."""
    Ur = U_block.reshape(2, 3, r)
    Vr = V_block.reshape(3, 3, r)
    Wr = W_block.reshape(3, 2, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _kron_decomposition():
    """
    Decompose 2x3 * 3x9 into 3 independent 2x3 * 3x3 blocks.
    Each block has schoolbook rank 18. Total rank = 54.
    """
    n, m, p = DIMENSIONS
    block_size = 3
    num_blocks = p // block_size
    
    rank = num_blocks * n * m * block_size  # 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for kp in range(block_size):
                    U[i * m + j, col] = 1
                    V[j * p + (b * block_size + kp), col] = 1
                    W[(b * block_size + kp) * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_2x3_3x3(target_rank, num_trials=2000, seed=42):
    """
    Search for a low-rank decomposition of 2x3 * 3x3.
    Returns (U, V, W) or None.
    """
    random.seed(seed)
    vals = [0, 1, -1, 0.5, -0.5]
    for _ in range(num_trials):
        U = np.zeros((6, target_rank))
        V = np.zeros((9, target_rank))
        W = np.zeros((6, target_rank))
        for c in range(target_rank):
            for idx in range(6):
                if random.random() < 0.45:
                    U[idx, c] = random.choice(vals)
            for idx in range(9):
                if random.random() < 0.45:
                    V[idx, c] = random.choice(vals)
            for idx in range(6):
                if random.random() < 0.45:
                    W[idx, c] = random.choice(vals)
        if _verify_block(U, V, W, target_rank):
            return U, V, W
    return None

def _build_from_block(U_block, V_block, W_block, block_rank):
    """
    Build the full 2x3*3x9 decomposition from a 2x3*3x3 block decomposition.
    """
    n, m, p = DIMENSIONS
    block_size = 3
    num_blocks = p // block_size
    total_rank = num_blocks * block_rank
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    for b in range(num_blocks):
        col_offset = b * block_rank
        for c in range(block_rank):
            U[:, col_offset + c] = U_block[:, c]
            for j in range(m):
                for kp in range(block_size):
                    V[j * p + (b * block_size + kp), col_offset + c] = V_block[j * block_size + kp, c]
            for kp in range(block_size):
                for i in range(n):
                    W[(b * block_size + kp) * n + i, col_offset + c] = W_block[kp * n + i, c]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try Kronecker decomposition
    kron_result = _kron_decomposition()
    if _verify(np.array(kron_result['U']), np.array(kron_result['V']), np.array(kron_result['W']), n, m, p, len(kron_result['U'][0])):
        kron_rank = len(kron_result['U'][0])
        if kron_rank < best_rank:
            best = kron_result
            best_rank = kron_rank
    
    # Try to find low-rank 2x3*3x3 block and use it
    for block_rank in range(12, 18):
        if time.time() > deadline - 30:
            break
        result = _search_2x3_3x3(block_rank, num_trials=1000, seed=1000 + block_rank)
        if result is not None:
            U_block, V_block, W_block = result
            full = _build_from_block(U_block, V_block, W_block, block_rank)
            if _verify(np.array(full['U']), np.array(full['V']), np.array(full['W']), n, m, p, len(full['U'][0])):
                full_rank = len(full['U'][0])
                if full_rank < best_rank:
                    best = full
                    best_rank = full_rank
                break  # Found the best block rank, stop
    
    # Random search for full problem
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
    ]

    random.seed(12345)
    low_bound = max(n * m, m * p, p * n)
    rmax = best_rank

    for r in range(low_bound, rmax + 1):
        if time.time() > deadline:
            break
        trials = 2000
        for _ in range(trials):
            if time.time() > deadline:
                break
            vs = random.choice(val_sets)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                for idx in range(n * m):
                    if random.random() < 0.5:
                        U[idx, c] = random.choice(vs)
                for idx in range(m * p):
                    if random.random() < 0.5:
                        V[idx, c] = random.choice(vs)
                for idx in range(p * n):
                    if random.random() < 0.5:
                        W[idx, c] = random.choice(vs)
            if _verify(U, V, W, n, m, p, r):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                return best
    return best
# EVOLVE-BLOCK-END
