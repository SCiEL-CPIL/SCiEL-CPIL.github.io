"""Matrix multiplication 2x5x6 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _to_halfint(x):
    """Convert to integer or half-integer, or None if not representable."""
    if isinstance(x, int):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float):
        # Check if it's an integer or half-integer
        rounded_2x = round(2 * x)
        if abs(2 * x - rounded_2x) < 1e-9:
            result = rounded_2x / 2
            if abs(result - round(result)) < 1e-9:
                return int(round(result))
            else:
                return result
        return None
    if hasattr(x, 'numerator') and hasattr(x, 'denominator'):
        # Fraction
        if x.denominator == 1:
            return x.numerator
        if x.denominator == 2:
            return x.numerator / 2
        return None
    return None

def _verify(U, V, W, n, m, p):
    """Verify decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    # Check entries
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if _to_halfint(val) is None:
                    return False
    # Verify tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Also check that off-diagonal terms are zero
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                continue
                            s = 0.0
                            for c in range(rank):
                                s += U[i1 * m + j1][c] * V[j2 * p + k2][c] * W[k2 * n + i2][c]
                            if abs(s) > 1e-9:
                                return False
    return True

def _strassen_222():
    """
    Strassen's 7-term decomposition for 2x2x2.
    U, V, W are lists of 4 lists of 7 values.
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([-1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, -1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]),
    ]
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    for c in range(7):
        u, v, w = terms[c]
        for idx in range(4):
            U[idx][c] = u[idx]
            V[idx][c] = v[idx]
            W[idx][c] = w[idx]
    return U, V, W

def _schoolbook_block(n_b, m_b, p_b):
    """Schoolbook decomposition for n_b x m_b x p_b block."""
    rank = n_b * m_b * p_b
    U = [[0] * rank for _ in range(n_b * m_b)]
    V = [[0] * rank for _ in range(m_b * p_b)]
    W = [[0] * rank for _ in range(p_b * n_b)]
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i * m_b + j][col] = 1
                V[j * p_b + k][col] = 1
                W[k * n_b + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Partition J into blocks and K into blocks
    # To minimize rank, we want to use Strassen for 2x2 blocks where possible.
    # J has 5 elements, K has 6 elements.
    # Best partition: J = [2, 2, 1], K = [2, 2, 2]
    # This gives 3*3 = 9 blocks:
    #   (2,2) -> Strassen rank 7
    #   (2,2) -> Strassen rank 7
    #   (2,2) -> Strassen rank 7
    #   (2,2) -> Strassen rank 7
    #   (2,2) -> Strassen rank 7
    #   (2,2) -> Strassen rank 7
    #   (1,2) -> schoolbook rank 2
    #   (1,2) -> schoolbook rank 2
    #   (1,2) -> schoolbook rank 2
    # Total: 6*7 + 3*2 = 42 + 6 = 48
    
    # Wait, let me reconsider. J has 5 elements, so partitions of 5:
    # [2,2,1] -> 3 blocks
    # K has 6 elements, partitions of 6:
    # [2,2,2] -> 3 blocks
    # So 3*3 = 9 blocks total.
    
    # Block (m_b, p_b) -> rank:
    # (2,2) -> 7 (Strassen)
    # (2,1) -> 2 (schoolbook)
    # (1,2) -> 2 (schoolbook)
    # (1,1) -> 1 (schoolbook)
    
    # With J=[2,2,1], K=[2,2,2]:
    # (2,2)*3 + (2,2)*3 + (2,2)*3... wait no.
    # Block (j_idx, k_idx) has size (len(J[j_idx]), len(K[k_idx]))
    # J blocks: [2, 2, 1]
    # K blocks: [2, 2, 2]
    # Block (0,0): (2,2) -> 7
    # Block (0,1): (2,2) -> 7
    # Block (0,2): (2,2) -> 7
    # Block (1,0): (2,2) -> 7
    # Block (1,1): (2,2) -> 7
    # Block (1,2): (2,2) -> 7
    # Block (2,0): (1,2) -> 2
    # Block (2,1): (1,2) -> 2
    # Block (2,2): (1,2) -> 2
    # Total: 6*7 + 3*2 = 42 + 6 = 48
    
    # Alternative: J=[3,2], K=[2,2,2]
    # (3,2) -> schoolbook 6
    # (3,2) -> schoolbook 6
    # (3,2) -> schoolbook 6
    # (2,2) -> 7
    # (2,2) -> 7
    # (2,2) -> 7
    # Total: 3*6 + 3*7 = 18 + 21 = 39? No wait, that's not right.
    # Block (0,0): (3,2) -> 3*2 = 6
    # Block (0,1): (3,2) -> 6
    # Block (0,2): (3,2) -> 6
    # Block (1,0): (2,2) -> 7
    # Block (1,1): (2,2) -> 7
    # Block (1,2): (2,2) -> 7
    # Total: 18 + 21 = 39? That seems too good. Let me verify.
    
    # Actually wait - for a 3x2 block, can we do better than schoolbook?
    # 3x2x2 tensor... The minimum rank for 3x2x2 is not trivially known.
    # Schoolbook gives 3*2*2 = 12? No wait, the block is n_b x m_b x p_b where n_b=2 (rows of A), m_b=3 (cols of A = rows of B), p_b=2 (cols of B).
    # So it's a 2x3x2 tensor. Schoolbook rank = 2*3*2 = 12.
    # But we can partition the 3 dimension... Actually no, within a block we use a single decomposition.
    
    # Let me reconsider. The block decomposition is:
    # A is 2x5, B is 5x6, C = A*B is 2x6.
    # We partition the 5 columns of A / 5 rows of B into J blocks.
    # We partition the 6 columns of B into K blocks.
    # For each pair (J_block, K_block), we compute a sub-product of size 2 x len(J_block) times len(J_block) x len(K_block) = 2 x len(K_block).
    # The rank of this sub-decomposition depends on (2, len(J_block), len(K_block)).
    
    # For (2, 2, 2): Strassen gives rank 7.
    # For (2, 3, 2): This is a 2x3x2 tensor. Can we do better than 2*3*2=12?
    #   We could partition the 3 into [2,1] and use Strassen for (2,2,2) and schoolbook for (2,1,2).
    #   But that's a further partition, not a single block decomposition.
    #   In our framework, each (J_block, K_block) pair gets one decomposition.
    
    # So the options for (2, m_b, p_b) blocks:
    # If m_b=2, p_b=2: use Strassen, rank 7
    # Otherwise: use schoolbook, rank 2*m_b*p_b
    
    # Let me enumerate all partitions of 5 and 6:
    # Partitions of 5: [5], [4,1], [3,2], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]
    # Partitions of 6: [6], [5,1], [4,2], [4,1,1], [3,3], [3,2,1], [3,1,1,1], [2,2,2], [2,2,1,1], [2,1,1,1,1], [1,1,1,1,1,1]
    
    # For each partition pair, compute total rank:
    # rank(J_block, K_block) = 7 if (m_b==2 and p_b==2) else 2*m_b*p_b
    
    # Let me write this out:
    
    def _compute_rank(partition_j, partition_k):
        total = 0
        for jb in partition_j:
            for kb in partition_k:
                if jb == 2 and kb == 2:
                    total += 7
                else:
                    total += 2 * jb * kb
        return total
    
    # Try all partitions:
    best_rank = 999
    best_partition_j = None
    best_partition_k = None
    
    def _get_partitions(n, max_parts=None):
        """Generate all integer partitions of n (as lists, sorted descending)."""
        if max_parts is None:
            max_parts = n
        if n == 0:
            yield []
            return
        if len([]) >= max_parts:
            return
        for first in range(n, 0, -1):
            for rest in _get_partitions(n - first, max_parts - 1):
                yield [first] + rest
    
    # To limit computation, let's just try reasonable partitions
    # Partitions of 5 (max 3 parts for efficiency):
    parts5 = [
        [5],
        [4, 1],
        [3, 2],
        [3, 1, 1],
        [2, 2, 1],
        [2, 1, 1, 1],
    ]
    # Partitions of 6 (max 3 parts):
    parts6 = [
        [6],
        [5, 1],
        [4, 2],
        [4, 1, 1],
        [3, 3],
        [3, 2, 1],
        [3, 1, 1, 1],
        [2, 2, 2],
        [2, 2, 1, 1],
    ]
    
    for pj in parts5:
        for pk in parts6:
            r = _compute_rank(pj, pk)
            if r < best_rank:
                best_rank = r
                best_partition_j = pj
                best_partition_k = pk
    
    # Now build the decomposition
    J_blocks = []
    pos = 0
    for sz in best_partition_j:
        J_blocks.append(list(range(pos, pos + sz)))
        pos += sz
    
    K_blocks = []
    pos = 0
    for sz in best_partition_k:
        K_blocks.append(list(range(pos, pos + sz)))
        pos += sz
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    total_rank = 0
    
    for J in J_blocks:
        m_b = len(J)
        for K in K_blocks:
            p_b = len(K)
            n_b = 2
            if m_b == 2 and p_b == 2:
                U_b, V_b, W_b = _strassen_222()
            else:
                U_b, V_b, W_b = _schoolbook_block(n_b, m_b, p_b)
            U_blocks.append(U_b)
            V_blocks.append(V_b)
            W_blocks.append(W_b)
            total_rank += len(U_b[0])
    
    U_final = [[0] * total_rank for _ in range(n * m)]
    V_final = [[0] * total_rank for _ in range(m * p)]
    W_final = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for idx, J in enumerate(J_blocks):
        m_b = len(J)
        for idx_k, K in enumerate(K_blocks):
            p_b = len(K)
            n_b = 2
            U_b = U_blocks[idx * len(K_blocks) + idx_k]
            V_b = V_blocks[idx * len(K_blocks) + idx_k]
            W_b = W_blocks[idx * len(K_blocks) + idx_k]
            block_rank = len(U_b[0])
            
            for i in range(n_b):
                for j_local in range(m_b):
                    global_j = J[j_local]
                    u_global_idx = i * m + global_j
                    u_local_idx = i * m_b + j_local
                    for c in range(block_rank):
                        U_final[u_global_idx][col_offset + c] = U_b[u_local_idx][c]
            
            for j_local in range(m_b):
                global_j = J[j_local]
                for k_local in range(p_b):
                    global_k = K[k_local]
                    v_global_idx = global_j * p + global_k
                    v_local_idx = j_local * p_b + k_local
                    for c in range(block_rank):
                        V_final[v_global_idx][col_offset + c] = V_b[v_local_idx][c]
            
            for k_local in range(p_b):
                global_k = K[k_local]
                for i in range(n_b):
                    w_global_idx = global_k * n + i
                    w_local_idx = k_local * n_b + i
                    for c in range(block_rank):
                        W_final[w_global_idx][col_offset + c] = W_b[w_local_idx][c]
            
            col_offset += block_rank
    
    result = {"U": U_final, "V": V_final, "W": W_final}
    
    if _verify(U_final, V_final, W_final, n, m, p):
        return result
    else:
        # Fallback to full schoolbook
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

# EVOLVE-BLOCK-END
