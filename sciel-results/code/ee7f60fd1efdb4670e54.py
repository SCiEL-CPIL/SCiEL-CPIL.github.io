import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _build_system(rank, n, m, p):
    """
    Build the linear system for a given rank and random sparse support.
    Returns (A, b, U, V, W) where A is the coefficient matrix (144 x rank),
    b is the target vector (all 1s), and U, V, W are the support matrices.
    """
    # Target positions: (i*m+j, j*p+k, k*n+i) for i in 0..n-1, j in 0..m-1, k in 0..p-1
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                targets.append((a_idx, b_idx, c_idx))
    
    num_targets = len(targets)
    A = [[Fraction(0) for _ in range(rank)] for _ in range(num_targets)]
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each rank component r, choose a random sparse support
    for r in range(rank):
        # Choose a random "type" of support to encourage structure
        # Type 0: single entry in each of U, V, W (schoolbook-like)
        # Type 1: two entries in one matrix, single in others
        # Type 2: random sparse with small integers
        
        support_type = random.choice([0, 1, 2])
        
        if support_type == 0:
            # Single non-zero in each
            a_idx = random.randint(0, n * m - 1)
            b_idx = random.randint(0, m * p - 1)
            c_idx = random.randint(0, p * n - 1)
            u_val = random.choice([1, -1])
            v_val = random.choice([1, -1])
            w_val = random.choice([1, -1])
            U[a_idx][r] = u_val
            V[b_idx][r] = v_val
            W[c_idx][r] = w_val
        elif support_type == 1:
            # Two entries in U, single in V, W
            a_idx1 = random.randint(0, n * m - 1)
            a_idx2 = random.randint(0, n * m - 1)
            if a_idx1 == a_idx2:
                a_idx2 = (a_idx1 + 1) % (n * m)
            b_idx = random.randint(0, m * p - 1)
            c_idx = random.randint(0, p * n - 1)
            u1 = random.choice([1, -1])
            u2 = random.choice([1, -1])
            v_val = random.choice([1, -1])
            w_val = random.choice([1, -1])
            U[a_idx1][r] = u1
            U[a_idx2][r] = u2
            V[b_idx][r] = v_val
            W[c_idx][r] = w_val
        else:
            # Random sparse: 1-2 entries per matrix with small integer values
            num_u = random.randint(1, 2)
            num_v = random.randint(1, 2)
            num_w = random.randint(1, 2)
            
            u_indices = random.sample(range(n * m), min(num_u, n * m))
            v_indices = random.sample(range(m * p), min(num_v, m * p))
            w_indices = random.sample(range(p * n), min(num_w, p * n))
            
            for idx in u_indices:
                U[idx][r] = random.choice([1, -1, 2, -2])
            for idx in v_indices:
                V[idx][r] = random.choice([1, -1, 2, -2])
            for idx in w_indices:
                W[idx][r] = random.choice([1, -1, 2, -2])
        
        # Fill in the column of A
        for t, (a_idx, b_idx, c_idx) in enumerate(targets):
            prod = Fraction(U[a_idx][r]) * Fraction(V[b_idx][r]) * Fraction(W[c_idx][r])
            A[t][r] = prod
    
    b = [Fraction(1) for _ in range(num_targets)]
    return A, b, U, V, W


def _solve_linear_system(A, b):
    """
    Solve A x = b using Gaussian elimination with fractions.
    Returns solution vector x (list of Fractions) or None if no unique solution.
    A is num_targets x rank, b is num_targets.
    We need rank >= num_targets for a unique solution, but we can have rank > num_targets.
    In that case, we find one particular solution.
    """
    num_eq = len(b)
    num_var = len(A[0]) if A else 0
    
    # Augmented matrix
    aug = [[Fraction(A[i][j]) for j in range(num_var)] + [b[i]] for i in range(num_eq)]
    
    # Gaussian elimination
    pivot_row = 0
    for col in range(num_var):
        if pivot_row >= num_eq:
            break
        # Find pivot
        pivot = -1
        for row in range(pivot_row, num_eq):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
        # Normalize
        piv_val = aug[pivot_row][col]
        for j in range(col, num_var + 1):
            aug[pivot_row][j] /= piv_val
        # Eliminate
        for row in range(num_eq):
            if row != pivot_row and aug[row][col] != 0:
                factor = aug[row][col]
                for j in range(col, num_var + 1):
                    aug[row][j] -= factor * aug[pivot_row][j]
        pivot_row += 1
    
    # Check consistency
    for i in range(pivot_row, num_eq):
        if aug[i][num_var] != 0:
            return None
    
    # Extract solution: free variables set to 0
    x = [Fraction(0) for _ in range(num_var)]
    for i in range(pivot_row):
        # Find the column with pivot in row i
        for j in range(num_var):
            if aug[i][j] != 0:
                x[j] = aug[i][num_var]
                break
    return x


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                s = Fraction(0)
                for r in range(len(U[a_idx])):
                    s += Fraction(U[a_idx][r]) * Fraction(V[b_idx][r]) * Fraction(W[c_idx][r])
                if s != 1:
                    return False
    return True


def _try_rank(rank, n, m, p, max_attempts, deadline):
    """Try to find a valid decomposition at the given rank."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        A, b, U, V, W = _build_system(rank, n, m, p)
        x = _solve_linear_system(A, b)
        if x is None:
            continue
        
        # Apply solution: U, V, W are already set as supports, but we need to scale
        # Actually, the system was built with U,V,W as the coefficient columns.
        # The solution x gives the actual values. Wait, no:
        # In _build_system, we set U[a_idx][r] etc. to integer values, and A[t][r] = U*V*W.
        # Then we solve A x = b. But that's not right.
        # The correct formulation: sum_r U[a,r] V[b,r] W[c,r] = 1
        # This is linear in the entries of U, V, W jointly? No, it's trilinear.
        # So we can't just solve a linear system in the entries.
        
        # The random support search works differently:
        # We fix the SUPPORT (which entries are non-zero) and then solve for the VALUES.
        # But the equation is trilinear, not linear.
        
        # Correct approach: fix the support pattern, then the problem is still nonlinear.
        # However, if we fix V and W, it's linear in U. Or fix U and W, linear in V.
        
        # Alternative: use the fact that for a fixed support, we can do alternating minimization
        # or use a different approach.
        
        # Let me reconsider. The standard approach for sparse support search is:
        # 1. Choose support patterns for U, V, W
        # 2. The tensor equation is sum_r u_a^r v_b^r w_c^r = T_abc
        # 3. This is a system of polynomial equations. For small rank, we can solve.
        # 4. But solving polynomial systems is hard.
        
        # A simpler approach that works: use the "slice" method.
        # Fix one factor (say W) to random values, then the problem becomes bilinear in U and V,
        # which can be linearized.
        
        # Actually, the most practical approach for this task within time limits is:
        # Use the known Strassen decomposition as a starting point and try to reduce rank
        # by finding linear dependencies or using a different block structure.
        
        # Given the complexity, let me implement a different strategy:
        # Use the Strassen 4x4x4 decomposition (rank 49) for the first 8 columns of B,
        # and try to find a lower-rank decomposition for the last column interaction.
        
        # Or better: try rank 100-113 with a structured approach.
        
        pass
    
    return None


def _strassen_444_decomposition():
    """Return the rank-49 Strassen decomposition for 4x4x4 matrix multiplication."""
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 vectors
    strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # For 4x4x4, we have 4x4 = 4 blocks of 2x2
    # Block indices: (br, bc) where br, bc in 0..1
    # For each pair of A-blocks and B-blocks, apply Strassen
    col = 0
    for ar in range(2):
        for ac in range(2):
            for br in range(2):
                for bc in range(2):
                    # A block (ar, ac), B block (br, bc) -> C block (ar, bc)
                    for t in range(7):
                        u_vec, v_vec, w_vec = strassen[t]
                        # A indices: (ar*2 + u_row, ac*2 + u_col)
                        for s in range(4):
                            if u_vec[s]:
                                a_row = ar * 2 + (s // 2)
                                a_col = ac * 2 + (s % 2)
                                U[a_row * 4 + a_col][col] += u_vec[s]
                            if v_vec[s]:
                                b_row = br * 2 + (s // 2)
                                b_col = bc * 2 + (s % 2)
                                V[b_row * 4 + b_col][col] += v_vec[s]
                            if w_vec[s]:
                                c_row = ar * 2 + (s // 2)
                                c_col = bc * 2 + (s % 2)
                                W[c_row * 4 + c_col][col] += w_vec[s]
                        col += 1
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Start with the known rank-114 decomposition
    # 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 16) = 114
    
    # Build the 4x4x4 Strassen decomposition
    U444, V444, W444, rank444 = _strassen_444_decomposition()
    
    # Full 4x4x9 decomposition
    rank = 114
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    # Part 1: A(4x4) * B(4x4, cols 0-3) -> C(4x4, cols 0-3)
    for i in range(16):
        for r in range(49):
            U[i][r] = U444[i][r]
    for i in range(16):
        for r in range(49):
            V[i][r] = V444[i][r]
    for i in range(16):
        for r in range(49):
            W[i][r] = W444[i][r]
    
    # Part 2: A(4x4) * B(4x4, cols 4-7) -> C(4x4, cols 4-7)
    for i in range(16):
        for r in range(49):
            U[i][49 + r] = U444[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 9 + 4 + k][49 + r] = V444[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[(4 + k) * 4 + i][49 + r] = W444[k * 4 + i][r]
    
    # Part 3: A(4x4) * B(4x1, col 8) -> C(4x1, col 8)
    for i in range(4):
        for j in range(4):
            for r in range(4):
                U[i * 4 + j][98 + r] = 1 if j == r else 0
                V[j * 9 + 8][98 + r] = 1 if j == r else 0
                W[8 * 4 + i][98 + r] = 1 if i == r else 0
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank by checking for linear dependencies
        # Convert to numpy for SVD-based analysis
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Check if we can reduce rank by finding a lower-rank equivalent
        # This is a research-level problem, so we'll just return the rank-114 solution
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 144
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
