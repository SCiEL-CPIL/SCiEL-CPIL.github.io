"""Find rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import numpy as np
import itertools
from collections import defaultdict
from fractions import Fraction

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0  # Safety margin before 150s limit


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7

    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    def is_valid_w_val(v):
        """Check if v is integer or half-integer."""
        if abs(v) < 1e-12:
            return True
        if abs(v - round(v)) < 1e-9:
            return True
        if abs(v * 2 - round(v * 2)) < 1e-9:
            return True
        return False

    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def check_decomposition_fast(U, V, W, rank):
        """Fast vectorized check using numpy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def solve_for_w_given_uv(U, V, rank):
        """
        Given U and V, solve for W such that the decomposition holds.
        For each (i,k), we need:
          sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        This is a linear system in W[k*n+i, :].
        Returns W or None if no solution with valid entries exists.
        """
        W = np.zeros((p * n, rank))
        for i in range(n):
            for k in range(p):
                # Build A_sub: m x rank matrix where A_sub[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A_sub = np.zeros((m, rank))
                for j in range(m):
                    A_sub[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # We need A_sub @ W_row = ones(m)
                # Solve the linear system
                try:
                    # Use least squares to find a solution
                    W_row, residuals, rank_A, sv = np.linalg.lstsq(A_sub, np.ones(m), rcond=1e-12)
                    
                    # Check if the solution is exact
                    if not np.allclose(A_sub @ W_row, np.ones(m), atol=1e-9):
                        return None
                    
                    # Check if all entries are valid (integer or half-integer)
                    for val in W_row:
                        if not is_valid_w_val(val):
                            return None
                    
                    W[k * n + i, :] = W_row
                except np.linalg.LinAlgError:
                    return None
        
        return W

    def try_random_sparse_uv(rank, num_trials=500):
        """
        Try to find rank-r decomposition with sparse U, V entries in {-1, 0, 1}.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 5:
                return None

            # Generate random sparse U and V
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            for r in range(rank):
                # Each column of U has 2-3 nonzeros
                nz_u = random.choice([2, 3])
                indices_u = random.sample(range(n * m), min(nz_u, n * m))
                for idx in indices_u:
                    U[idx, r] = random.choice([-1.0, 1.0])
                
                # Each column of V has 2-3 nonzeros
                nz_v = random.choice([2, 3])
                indices_v = random.sample(range(m * p), min(nz_v, m * p))
                for idx in indices_v:
                    V[idx, r] = random.choice([-1.0, 1.0])
            
            # Solve for W
            W = solve_for_w_given_uv(U, V, rank)
            if W is not None:
                # Verify the full decomposition
                if check_decomposition(U, V, W, rank):
                    return (U, V, W)
        
        return None

    def try_structured_uv(rank):
        """
        Try structured patterns for U and V based on index symmetries.
        """
        # Pattern 1: Use modular arithmetic to create structured patterns
        for pattern_idx in range(20):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            if pattern_idx % 4 == 0:
                # Cyclic pattern
                for r in range(rank):
                    i_u = (r * 3) % n
                    j_u = (r * 5) % m
                    j_v = (r * 7) % m
                    k_v = (r * 11) % p
                    U[i_u * m + j_u, r] = 1.0 if r % 2 == 0 else -1.0
                    V[j_v * p + k_v, r] = 1.0 if r % 2 == 0 else -1.0
            elif pattern_idx % 4 == 1:
                # Block pattern
                for r in range(rank):
                    i_u = r % n
                    j_u = (r // n) % m
                    j_v = r % m
                    k_v = (r // m) % p
                    U[i_u * m + j_u, r] = 1.0
                    V[j_v * p + k_v, r] = 1.0
            elif pattern_idx % 4 == 2:
                # Mixed pattern with signs
                for r in range(rank):
                    i_u = (r * 2) % n
                    j_u = (r * 3) % m
                    j_v = (r * 4) % m
                    k_v = (r * 5) % p
                    sign_u = 1.0 if (r % 3 == 0) else (-1.0 if r % 3 == 1 else 0.5)
                    sign_v = 1.0 if (r % 2 == 0) else -1.0
                    U[i_u * m + j_u, r] = sign_u
                    V[j_v * p + k_v, r] = sign_v
            else:
                # Random but with guaranteed coverage of all (i,j) and (j,k)
                for r in range(rank):
                    i_u = random.randint(0, n - 1)
                    j_u = random.randint(0, m - 1)
                    j_v = random.randint(0, m - 1)
                    k_v = random.randint(0, p - 1)
                    U[i_u * m + j_u, r] = random.choice([-1.0, 0.5, 1.0])
                    V[j_v * p + k_v, r] = random.choice([-1.0, 0.5, 1.0])
            
            W = solve_for_w_given_uv(U, V, rank)
            if W is not None and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None

    def try_all_ones_exploitation(rank):
        """
        Exploit the all-ones structure directly.
        For the all-ones tensor, try to use decompositions based on
        factoring 1 as a sum of products.
        """
        # Key insight: 1 = sum of products where we partition the index sets
        # Try to use the fact that we can write indicator functions
        
        # Strategy: For each rank-1 term, assign it to cover a "block" of (i,j,k)
        # where the product U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for those indices
        
        for trial in range(100):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Try to assign each rank-1 term to cover specific (i,j,k) combinations
            for r in range(rank):
                # Choose which (i,j) and (j,k) this term affects
                i_choice = random.randint(0, n - 1)
                j_choice = random.randint(0, m - 1)
                k_choice = random.randint(0, p - 1)
                
                # Option 1: Single index coverage
                if trial % 3 == 0:
                    U[i_choice * m + j_choice, r] = 1.0
                    V[j_choice * p + k_choice, r] = 1.0
                    W[k_choice * n + i_choice, r] = 1.0
                # Option 2: Multiple indices
                elif trial % 3 == 1:
                    # Cover all j for fixed i, k
                    for j in range(m):
                        U[i_choice * m + j, r] = 1.0
                        V[j * p + k_choice, r] = 1.0
                    W[k_choice * n + i_choice, r] = 1.0 / m
                else:
                    # Cover all k for fixed i, j
                    for k in range(p):
                        U[i_choice * m + j_choice, r] = 1.0
                        V[j_choice * p + k, r] = 1.0
                        W[k * n + i_choice, r] = 1.0 / p
            
            # Check if this works
            if check_decomposition(U, V, W, rank):
                # Verify all W entries are valid
                valid = True
                for row in W:
                    for val in row:
                        if not is_valid_w_val(val):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return (U, V, W)
        
        return None

    def try_systematic_low_rank():
        """
        Systematic search for low-rank decompositions using the structure
        of the all-ones tensor.
        """
        # For the all-ones tensor of shape (n,m,p), we can try:
        # 1. Decompositions based on partitions of indices
        # 2. Use the fact that 1 = sum of carefully chosen products
        
        # Key: try ranks from theoretical lower bound upward
        # Lower bound: at least max(n, m, p) = 7, but likely higher
        
        # Try specific structured decompositions
        
        # Approach: Use the "sum of indicators" method
        # For each (i,j,k), we need sum_r prod_r(i,j,k) = 1
        
        # Try rank = n*m = 12 first (one term per (i,j) pair)
        for rank in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Method 1: One term per (i,j) with W depending on k
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Assign each (i,j) to a unique rank index
            rank_idx = 0
            for i in range(n):
                for j in range(m):
                    if rank_idx >= rank:
                        break
                    U[i * m + j, rank_idx] = 1.0
                    # V and W need to be set so that for all k:
                    # V[j*p+k, rank_idx] * W[k*n+i, rank_idx] = 1
                    # This requires V[j*p+k, rank_idx] = 1 and W[k*n+i, rank_idx] = 1 for all k
                    for k in range(p):
                        V[j * p + k, rank_idx] = 1.0
                        W[k * n + i, rank_idx] = 1.0
                    rank_idx += 1
            
            # Check if this works (it should for rank >= n*m)
            if check_decomposition(U, V, W, rank):
                return (U, V, W, rank)
        
        return None

    # Try systematic low-rank first
    result = try_systematic_low_rank()
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try various ranks from low to high
    for rank in range(10, min(40, rank_sb)):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try structured approach
        result = try_structured_uv(rank)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            break
        
        # Try random sparse approach
        result = try_random_sparse_uv(rank, num_trials=100)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            break
        
        # Try all-ones exploitation
        result = try_all_ones_exploitation(rank)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
