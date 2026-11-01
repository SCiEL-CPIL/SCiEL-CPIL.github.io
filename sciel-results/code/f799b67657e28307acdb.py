import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p, num_tests=10):
    """Check that the decomposition is exact for random inputs."""
    rank = len(U[0])
    for seed in range(num_tests):
        random.seed(seed * 7 + 13)
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        X = [[0] * n for _ in range(rank)]
        for r in range(rank):
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j][r] * A[i][j]
                X[r][i] = s
        
        Y = [[0] * p for _ in range(rank)]
        for r in range(rank):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k][r] * B[j][k]
                Y[r][k] = s
        
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += W[k * n + i][r] * X[r][i] * Y[r][k]
                C[i][k] = s
        
        if C != AB:
            return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2x6():
    """
    Strassen-based decomposition for 2x2x6 matrix multiplication.
    Splits B into 3 blocks of 2 columns and applies Strassen to each.
    Rank = 3 * 7 = 21.
    """
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_s[s][ij] where ij = i*2+j
    U_s = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [-1, 0, 1, 0],  # M6: (A10-A00)
        [0, 1, 0, -1],  # M7: (A01-A11)
    ]
    
    # V_s[s][jk] where jk = j*2+k
    V_s = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01-B11)
        [-1, 0, 1, 0],  # M4: (B10-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1],   # M7: (B10+B11)
    ]
    
    # W_s[s][ki] where ki = k*2+i
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    W_s = [
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, 0],   # M2: C10+1
        [0, 1, 0, -1],  # M3: C01+1, C11-1
        [-1, 0, 1, 0],  # M4: C00-1, C10+1
        [0, 1, 0, 0],   # M5: C01+1
        [0, 0, 0, 1],   # M6: C11+1
        [1, 0, 0, 0],   # M7: C00+1
    ]
    
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            # U: same for all blocks
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            # V: offset by block*2 in the k-dimension
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    v_s_idx = j * 2 + k
                    V[v_idx][r] = V_s[s][v_s_idx]
            # W: offset by block*2 in the k-dimension
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    w_s_idx = k * 2 + i
                    W[w_idx][r] = W_s[s][w_s_idx]
                    
    return {"U": U, "V": V, "W": W}


def _reduce_rank_numpy(U, V, W, n, m, p, max_iterations=100):
    """
    Use NumPy to find linear dependencies and reduce rank.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # We want to find columns that are linearly dependent in U and V simultaneously.
    # If U[:, r1] and U[:, r2] are dependent, and V[:, r1] and V[:, r2] are dependent,
    # we can merge them.
    
    # Simple check: if U[:, r1] == U[:, r2] and V[:, r1] == V[:, r2], merge.
    # Or if U[:, r2] = a * U[:, r1] and V[:, r2] = b * V[:, r1].
    
    for _ in range(max_iterations):
        merged = False
        for r1 in range(rank):
            for r2 in range(r1 + 1, rank):
                # Check if U[:, r2] is proportional to U[:, r1]
                if np.allclose(U[:, r1], 0):
                    continue
                # Find scale
                # Use the first non-zero element
                idx = np.argmax(U[:, r1] != 0)
                if U[idx, r1] == 0:
                    continue
                scale_u = U[idx, r2] / U[idx, r1]
                if np.allclose(U[:, r2], scale_u * U[:, r1]):
                    # Check V
                    if np.allclose(V[:, r1], 0):
                        continue
                    idx_v = np.argmax(V[:, r1] != 0)
                    if V[idx_v, r1] == 0:
                        continue
                    scale_v = V[idx_v, r2] / V[idx_v, r1]
                    if np.allclose(V[:, r2], scale_v * V[:, r1]):
                        # Merge
                        # W[:, r1] += scale_u * scale_v * W[:, r2]
                        W[:, r1] += scale_u * scale_v * W[:, r2]
                        # Remove r2
                        U = np.delete(U, r2, axis=1)
                        V = np.delete(V, r2, axis=1)
                        W = np.delete(W, r2, axis=1)
                        rank -= 1
                        merged = True
                        break
            if merged:
                break
        if not merged:
            break
            
    # Convert back to list of lists
    U_out = [[int(round(x)) for x in row] for row in U.T]
    V_out = [[int(round(x)) for x in row] for row in V.T]
    W_out = [[int(round(x)) for x in row] for row in W.T]
    
    return {"U": U_out, "V": V_out, "W": W_out}


def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Try Strassen-based decomposition first (rank 21)
    result = _strassen_2x2x6()
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Try to reduce rank using NumPy
        if time.time() < deadline - 10:
            reduced = _reduce_rank_numpy(result["U"], result["V"], result["W"], n, m, p)
            if _check_decomposition(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                return reduced
        return result
    
    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    return result

# EVOLVE-BLOCK-END
