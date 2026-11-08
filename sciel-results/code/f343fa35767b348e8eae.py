import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 blocks
    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]

    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]

    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    # 8 blocks: 2 horizontal splits of A, 4 vertical splits of B
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=2, m=4, p=8):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        expected = 1 if (i == ip and k == kp) else 0
                        actual = 0
                        for r in range(rank):
                            actual += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        if actual != expected:
                            return False
    return True


def _try_eliminate(U, V, W, n=2, m=4, p=8, deadline=None):
    """Try to eliminate one rank term by solving for dependencies."""
    if deadline is not None and time.time() > deadline:
        return None
    
    R = len(U[0])
    
    # Try to eliminate each rank term
    for elim_r in range(R):
        if deadline is not None and time.time() > deadline:
            break
        
        # For each other rank term, check if we can adjust it to compensate
        # This is a simplified approach - try to see if removing term elim_r
        # can be compensated by scaling other terms
        
        new_rank = R - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        # Map old indices to new indices
        idx_map = {}
        new_i = 0
        for old_r in range(R):
            if old_r != elim_r:
                idx_map[old_r] = new_i
                new_i += 1
        
        # Try simple scaling: scale each remaining term by factor
        # to try to compensate for the removed term
        # This is heuristic - try various scaling patterns
        
        # First, try uniform scaling
        for scale in [1, 2, -1, -2]:
            for row in range(n * m):
                for old_r in range(R):
                    if old_r == elim_r:
                        continue
                    new_r = idx_map[old_r]
                    new_U[row][new_r] = scale * U[row][old_r]
            for row in range(m * p):
                for old_r in range(R):
                    if old_r == elim_r:
                        continue
                    new_r = idx_map[old_r]
                    new_V[row][new_r] = scale * V[row][old_r]
            for row in range(p * n):
                for old_r in range(R):
                    if old_r == elim_r:
                        continue
                    new_r = idx_map[old_r]
                    new_W[row][new_r] = scale * W[row][old_r]
            
            if _verify(new_U, new_V, new_W, n, m, p):
                return {"U": new_U, "V": new_V, "W": new_W}
        
        # Try to find a linear combination that works
        # This is more complex - skip for now
        
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with the known rank-56 solution
    base = _build_rank56()
    
    # Verify the base solution
    assert _verify(base["U"], base["V"], base["W"]), "Base rank-56 solution failed verification"
    
    # Try to reduce the rank
    if time.time() < deadline - 5:
        reduced = _try_eliminate(base["U"], base["V"], base["W"], deadline=deadline - 5)
        if reduced is not None:
            return reduced
    
    # Fallback to rank-56
    return base
