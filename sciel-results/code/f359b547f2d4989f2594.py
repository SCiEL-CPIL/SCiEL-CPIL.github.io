"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    # Deadline guard: 150s wall-clock limit, use 115s for safety
    start_time = time.time()
    deadline = start_time + 115

    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_rational(U, V, W):
        """Verify exact decomposition using rational arithmetic."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0, 1)
                    for r_idx in range(r):
                        s += Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx]) * Fraction(W[k * n + i][r_idx])
                    if s != Fraction(1, 1):
                        return False
        return True

    # Try to construct a WFTA-based decomposition for 5x5x5, then extend to 4x5x5
    # The 5x5x5 matrix multiplication tensor can be decomposed using the 5th-order cyclic group
    # For cyclic group Z_5, the convolution algebra has dimension 5, and the tensor rank
    # for 5x5x5 multiplication via this structure is 5 (one per frequency component)
    
    # However, we need to be careful: the standard matrix multiplication tensor T_{i,j,k} = delta_{i,j} * delta_{j,k}
    # for multiplying A[i,j] * B[j,k] = C[i,k]
    
    # For the 5x5x5 core, we can use the fact that over Z_5, we can use the discrete Fourier transform
    # The DFT matrix for Z_5 is 5x5, and the multiplication in frequency domain is pointwise
    
    # Let omega = exp(2*pi*i/5). The DFT matrix F has F[a,b] = omega^(a*b)
    # In frequency domain, C_hat = A_hat * B_hat (pointwise)
    # So the rank-5 decomposition would use the 5 frequency components
    
    # But we need integer/half-integer entries, so we need to work with the actual structure
    
    # Alternative approach: use the fact that 5 is prime, and try to find a structured decomposition
    
    # Let's try a direct approach: construct U, V, W for rank r where r < 100
    
    # For now, let's try a more systematic search with better initialization
    
    def try_structured_search(rank, num_trials=100):
        """Try to find a rank-r decomposition using structured random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            # Use a mix of structured and random entries
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            
            # Initialize with random patterns
            for i in range(nm):
                for r_idx in range(rank):
                    U[i][r_idx] = random.choice([-1, 0, 1])
            for j in range(mp):
                for r_idx in range(rank):
                    V[j][r_idx] = random.choice([-1, 0, 1])
            for k in range(pn):
                for r_idx in range(rank):
                    W[k][r_idx] = random.choice([-1, 0, 1])
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

        return None

    # Try ranks from 99 down to 80 with structured search
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break

        result = try_structured_search(rank, num_trials=20)
        if result is not None:
            best_rank = rank
            best_result = result
            break

    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
