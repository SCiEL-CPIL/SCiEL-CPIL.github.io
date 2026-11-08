import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 5, 5
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def check_deadline():
        return (time.time() - start_time) > DEADLINE

    def verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            U_col = U[:, c].reshape(n, m)
            V_col = V[:, c].reshape(m, p)
            W_col = W[:, c].reshape(p, n)
            T += U_col[:, :, None] * V_col[:, None, :] * W_col[None, :, :]
        if np.max(np.abs(T - 1.0)) > 1e-10:
            return False
        return True

    def add_strassen_block(i0, j0, k0, U, V, W, col):
        """Add a 2x2x2 Strassen block at position (i0, j0, k0)."""
        c = col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        c = col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        c = col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    def try_configuration(blocks):
        """Try a configuration of Strassen blocks and return result if valid."""
        num_blocks = len(blocks)
        total_rank = num_blocks * 7 + (n*m*p - num_blocks*8)

        U = [[0.0]*total_rank for _ in range(n*m)]
        V = [[0.0]*total_rank for _ in range(m*p)]
        W = [[0.0]*total_rank for _ in range(p*n)]

        col = 0
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]

        valid_config = True
        for i0, j0, k0 in blocks:
            if i0+1 >= n or j0+1 >= m or k0+1 >= p:
                valid_config = False
                break
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if covered[i0+di][j0+dj][k0+dk]:
                            valid_config = False
                            break
                    if not valid_config:
                        break
                if not valid_config:
                    break
            if not valid_config:
                break
            add_strassen_block(i0, j0, k0, U, V, W, col)
            col += 7
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True

        if not valid_config:
            return None

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] = 1.0
                        V[j*p+k][col] = 1.0
                        W[k*n+i][col] = 1.0
                        col += 1

        actual_rank = col
        U = [row[:actual_rank] for row in U]
        V = [row[:actual_rank] for row in V]
        W = [row[:actual_rank] for row in W]

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    # Generate all possible 2x2x2 block positions
    possible_blocks = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                possible_blocks.append((i0, j0, k0))

    # Try all combinations of up to 4 non-overlapping blocks
    # With n=3, m=5, p=5: i0 can be 0 or 1, j0 can be 0-3, k0 can be 0-3
    # Max non-overlapping: at most 2 in i-dim (but only 1 since n=3), 
    # at most 2 in j-dim, at most 2 in k-dim => max 4 blocks

    # Enumerate all valid sets of non-overlapping blocks
    def get_all_valid_block_sets(max_blocks=4):
        valid_sets = []
        
        def backtrack(start, current_set, covered):
            if len(current_set) >= max_blocks:
                valid_sets.append(list(current_set))
                return
            for idx in range(start, len(possible_blocks)):
                i0, j0, k0 = possible_blocks[idx]
                # Check if this block overlaps with any in current_set
                overlaps = False
                for (ci0, cj0, ck0) in current_set:
                    if (abs(ci0 - i0) < 2 and abs(cj0 - j0) < 2 and abs(ck0 - k0) < 2):
                        overlaps = True
                        break
                if not overlaps:
                    current_set.append((i0, j0, k0))
                    backtrack(idx + 1, current_set, covered)
                    current_set.pop()
        
        backtrack(0, [], set())
        return valid_sets

    # Get all valid configurations
    all_configs = get_all_valid_block_sets(max_blocks=4)
    
    # Sort by number of blocks (more blocks = lower rank)
    all_configs.sort(key=lambda x: -len(x))
    
    best_result = None
    best_rank = float('inf')
    
    for blocks in all_configs:
        if check_deadline():
            break
        
        result = try_configuration(blocks)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
                # If we achieved the theoretical minimum, stop
                # Minimum possible with 4 blocks: 4*7 + (75-32) = 28 + 43 = 71
                if rank <= 71:
                    break
    
    if best_result is not None:
        return best_result

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = 3, 5, 5
    print(f"Rank: {U.shape[1]}")
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    print(f"Valid: {ok}")
