"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) lists of lists, rank n*m*p."""
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
    return U, V, W


def _strassen_2x2():
    """Strassen's 7-product algorithm for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists (integers).
    A is 2x2 with rows [a b; c d] flattened row-major: (a,b,c,d)
    B is 2x2 with rows [e f; g h] flattened row-major: (e,f,g,h)
    C = A*B has rows [a*e+b*g, a*f+b*h; c*e+d*g, c*f+d*h]
    W index is k*n + i = k*2 + i (since n=2, p=2).
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, -1, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]    # d
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, 0, 0, 1, 0, 1]    # h
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],   # C[0][0]
        [0, 0, 1, 0, 1, 0, 0],     # C[0][1]
        [0, 1, 0, 1, 0, 0, 0],     # C[1][0]
        [1, -1, -1, 0, 0, 1, 0]    # C[1][1]
    ]
    return U, V, W


def _embed_subproblem(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_offset, j_offset, k_offset, n, m, p):
    """Embed a subproblem decomposition into the full 4x5x5 space."""
    rank_sub = len(U_sub[0])
    U_full = [[0] * rank_sub for _ in range(n * m)]
    V_full = [[0] * rank_sub for _ in range(m * p)]
    W_full = [[0] * rank_sub for _ in range(p * n)]
    
    for local_i in range(n_sub):
        for local_j in range(m_sub):
            full_i = i_offset + local_i
            full_j = j_offset + local_j
            local_idx = local_i * m_sub + local_j
            full_idx = full_i * m + full_j
            U_full[full_idx] = U_sub[local_i * m_sub + local_j]
    
    for local_j in range(m_sub):
        for local_k in range(p_sub):
            full_j = j_offset + local_j
            full_k = k_offset + local_k
            local_idx = local_j * p_sub + local_k
            full_idx = full_j * p + full_k
            V_full[full_idx] = V_sub[local_j * p_sub + local_k]
    
    for local_k in range(p_sub):
        for local_i in range(n_sub):
            full_k = k_offset + local_k
            full_i = i_offset + local_i
            local_idx = local_k * n_sub + local_i
            full_idx = full_k * n + full_i
            W_full[full_idx] = W_sub[local_k * n_sub + local_i]
    
    return U_full, V_full, W_full


def _concatenate_decompositions(U1, V1, W1, U2, V2, W2, n, m, p):
    """Concatenate two decompositions horizontally (independent terms)."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    U = []
    for i in range(n * m):
        U.append(U1[i] + U2[i])
    V = []
    for i in range(m * p):
        V.append(V1[i] + V2[i])
    W = []
    for i in range(p * n):
        W.append(W1[i] + W2[i])
    return U, V, W


def _construct_4x5x5_optimized():
    """Build 4x5x5 decomposition with optimized block splits.
    
    Part 1: A[:, 0:2] (4x2) * B[0:2, :] (2x5)
    Split 4x2 into two 2x2. Split 2x5 into 2x2 and 2x3.
    Top-Left (2x2*2x2): Strassen (7)
    Top-Right (2x2*2x3): Split B into 2x2 and 2x1.
        2x2*2x2: Strassen (7)
        2x2*2x1: Schoolbook (4)
        Total: 11
    Bottom-Left (2x2*2x2): Strassen (7)
    Bottom-Right (2x2*2x3): Split B into 2x2 and 2x1.
        2x2*2x2: Strassen (7)
        2x2*2x1: Schoolbook (4)
        Total: 11
    Part 1 Total: 36
    
    Part 2: A[:, 2:5] (4x3) * B[2:5, :] (3x5)
    Split 4x3 into 4x1 and 4x2.
    Sub-part 2a: 4x1 * 1x5: Schoolbook (20)
    Sub-part 2b: 4x2 * 2x5: Same as Part 1 (36)
    Part 2 Total: 56
    
    Grand Total: 92
    """
    n, m, p = 4, 5, 5
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    # Top-Left: A[0:2, 0:2] * B[0:2, 0:2]
    U_1a, V_1a, W_1a = _strassen_2x2()
    U_1a_full, V_1a_full, W_1a_full = _embed_subproblem(
        U_1a, V_1a, W_1a, 2, 2, 2, i_offset=0, j_offset=0, k_offset=0, n=n, m=m, p=p)
    
    # Top-Right: A[0:2, 0:2] * B[0:2, 2:5]
    # Split B[0:2, 2:5] into B[0:2, 2:4] (2x2) and B[0:2, 4:5] (2x1)
    U_1b1, V_1b1, W_1b1 = _strassen_2x2()
    U_1b1_full, V_1b1_full, W_1b1_full = _embed_subproblem(
        U_1b1, V_1b1, W_1b1, 2, 2, 2, i_offset=0, j_offset=0, k_offset=2, n=n, m=m, p=p)
        
    U_1b2, V_1b2, W_1b2 = _schoolbook(2, 2, 1)
    U_1b2_full, V_1b2_full, W_1b2_full = _embed_subproblem(
        U_1b2, V_1b2, W_1b2, 2, 2, 1, i_offset=0, j_offset=0, k_offset=4, n=n, m=m, p=p)
        
    U_1b_full, V_1b_full, W_1b_full = _concatenate_decompositions(
        U_1b1_full, V_1b1_full, W_1b1_full, U_1b2_full, V_1b2_full, W_1b2_full, n, m, p)

    # Bottom-Left: A[2:4, 0:2] * B[0:2, 0:2]
    U_1c, V_1c, W_1c = _strassen_2x2()
    U_1c_full, V_1c_full, W_1c_full = _embed_subproblem(
        U_1c, V_1c, W_1c, 2, 2, 2, i_offset=2, j_offset=0, k_offset=0, n=n, m=m, p=p)

    # Bottom-Right: A[2:4, 0:2] * B[0:2, 2:5]
    U_1d1, V_1d1, W_1d1 = _strassen_2x2()
    U_1d1_full, V_1d1_full, W_1d1_full = _embed_subproblem(
        U_1d1, V_1d1, W_1d1, 2, 2, 2, i_offset=2, j_offset=0, k_offset=2, n=n, m=m, p=p)
        
    U_1d2, V_1d2, W_1d2 = _schoolbook(2, 2, 1)
    U_1d2_full, V_1d2_full, W_1d2_full = _embed_subproblem(
        U_1d2, V_1d2, W_1d2, 2, 2, 1, i_offset=2, j_offset=0, k_offset=4, n=n, m=m, p=p)
        
    U_1d_full, V_1d_full, W_1d_full = _concatenate_decompositions(
        U_1d1_full, V_1d1_full, W_1d1_full, U_1d2_full, V_1d2_full, W_1d2_full, n, m, p)

    # Concatenate Part 1
    U_part1, V_part1, W_part1 = U_1a_full, V_1a_full, W_1a_full
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1b_full, V_1b_full, W_1b_full, n, m, p)
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1c_full, V_1c_full, W_1c_full, n, m, p)
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1d_full, V_1d_full, W_1d_full, n, m, p)

    # Part 2: A[:, 2:5] * B[2:5, :]
    # Sub-part 2a: A[:, 2] (4x1) * B[2, :] (1x5)
    U_2a, V_2a, W_2a = _schoolbook(4, 1, 5)
    U_2a_full, V_2a_full, W_2a_full = _embed_subproblem(
        U_2a, V_2a, W_2a, 4, 1, 5, i_offset=0, j_offset=2, k_offset=0, n=n, m=m, p=p)

    # Sub-part 2b: A[:, 3:5] (4x2) * B[3:5, :] (2x5)
    # Same structure as Part 1
    # Top-Left: A[0:2, 3:5] * B[3:5, 0:2]
    U_2b1a, V_2b1a, W_2b1a = _strassen_2x2()
    U_2b1a_full, V_2b1a_full, W_2b1a_full = _embed_subproblem(
        U_2b1a, V_2b1a, W_2b1a, 2, 2, 2, i_offset=0, j_offset=3, k_offset=0, n=n, m=m, p=p)
        
    # Top-Right: A[0:2, 3:5] * B[3:5, 2:5]
    U_2b1b1, V_2b1b1, W_2b1b1 = _strassen_2x2()
    U_2b1b1_full, V_2b1b1_full, W_2b1b1_full = _embed_subproblem(
        U_2b1b1, V_2b1b1, W_2b1b1, 2, 2, 2, i_offset=0, j_offset=3, k_offset=2, n=n, m=m, p=p)
        
    U_2b1b2, V_2b1b2, W_2b1b2 = _schoolbook(2, 2, 1)
    U_2b1b2_full, V_2b1b2_full, W_2b1b2_full = _embed_subproblem(
        U_2b1b2, V_2b1b2, W_2b1b2, 2, 2, 1, i_offset=0, j_offset=3, k_offset=4, n=n, m=m, p=p)
        
    U_2b1b_full, V_2b1b_full, W_2b1b_full = _concatenate_decompositions(
        U_2b1b1_full, V_2b1b1_full, W_2b1b1_full, U_2b1b2_full, V_2b1b2_full, W_2b1b2_full, n, m, p)

    # Bottom-Left: A[2:4, 3:5] * B[3:5, 0:2]
    U_2b2a, V_2b2a, W_2b2a = _strassen_2x2()
    U_2b2a_full, V_2b2a_full, W_2b2a_full = _embed_subproblem(
        U_2b2a, V_2b2a, W_2b2a, 2, 2, 2, i_offset=2, j_offset=3, k_offset=0, n=n, m=m, p=p)

    # Bottom-Right: A[2:4, 3:5] * B[3:5, 2:5]
    U_2b2b1, V_2b2b1, W_2b2b1 = _strassen_2x2()
    U_2b2b1_full, V_2b2b1_full, W_2b2b1_full = _embed_subproblem(
        U_2b2b1, V_2b2b1, W_2b2b1, 2, 2, 2, i_offset=2, j_offset=3, k_offset=2, n=n, m=m, p=p)
        
    U_2b2b2, V_2b2b2, W_2b2b2 = _schoolbook(2, 2, 1)
    U_2b2b2_full, V_2b2b2_full, W_2b2b2_full = _embed_subproblem(
        U_2b2b2, V_2b2b2, W_2b2b2, 2, 2, 1, i_offset=2, j_offset=3, k_offset=4, n=n, m=m, p=p)
        
    U_2b2b_full, V_2b2b_full, W_2b2b_full = _concatenate_decompositions(
        U_2b2b1_full, V_2b2b1_full, W_2b2b1_full, U_2b2b2_full, V_2b2b2_full, W_2b2b2_full, n, m, p)

    U_2b_full, V_2b_full, W_2b_full = U_2b1a_full, V_2b1a_full, W_2b1a_full
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b1b_full, V_2b1b_full, W_2b1b_full, n, m, p)
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b2a_full, V_2b2a_full, W_2b2a_full, n, m, p)
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b2b_full, V_2b2b_full, W_2b2b_full, n, m, p)

    # Concatenate Part 2
    U_part2, V_part2, W_part2 = _concatenate_decompositions(
        U_2a_full, V_2a_full, W_2a_full, U_2b_full, V_2b_full, W_2b_full, n, m, p)

    # Concatenate Part 1 and Part 2
    U, V, W = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_part2, V_part2, W_part2, n, m, p)
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+r]))
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def construct():
    """Construct the rank-minimal (as far as we can determine) exact tensor
    decomposition for multiplying a 4x5 matrix by a 5x5 matrix."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try the improved block decomposition
    try:
        U, V, W = _construct_4x5x5_optimized()
        rank = len(W[0])
        
        # Verify correctness
        if _verify_decomposition(U, V, W, n, m, p):
            # Check deadline
            if time.time() - t0 < _DEADLINE_SECONDS - 5:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
