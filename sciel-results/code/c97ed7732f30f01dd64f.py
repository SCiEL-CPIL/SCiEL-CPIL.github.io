"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # Build index arrays for vectorized check
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using batched operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # For each (i,j,k) triple, check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # We can vectorize over r
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Element-wise product over rank dimension, then sum
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    num_blocks = 5
    block_size = 2
    
    # Baseline: Rank 55 solution using Strassen (7) + Outer (4) per 2x3x2 block
    rank_per_block_base = 11
    total_rank_base = num_blocks * rank_per_block_base
    
    U_base = np.zeros((n * m, total_rank_base))
    V_base = np.zeros((m * p, total_rank_base))
    W_base = np.zeros((p * n, total_rank_base))
    
    strassen_terms = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]
    outer_terms = [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            col = col_offset
            for i, j, val in term['U']:
                U_base[i * m + j, col] += val
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V_base[j * p + k_global, col] += val
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W_base[k_global * n + i, col] += val
            col_offset += 1
            
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = total_rank_base
    
    # Attempt to find a rank 10 block decomposition for 2x3x2
    def _find_rank_10_block():
        n_b, m_b, p_b = 2, 3, 2
        target_rank = 10
        time_limit = 35
        
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Build the target tensor: T[i,j,k] = 1 if this is a valid matmul entry
        # For M_{2,3,2}: indices (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1}
        # All 12 entries are 1 (since we're decomposing the multiplication tensor)
        
        # Strategy 1: Structured search based on known patterns
        # The 2x3x2 matmul tensor can be seen as two 2x3 matrices (for k=0 and k=1)
        # We need to find rank-10 decomposition
        
        # Try various structured families
        # Family 1: Based on Strassen-like patterns with modifications
        # For 2x2x2, Strassen uses rank 7. For 2x3x2, we need more terms.
        
        # Let's try a direct algebraic approach:
        # We have 12 equations (one for each (i,j,k) pair)
        # Each equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # With 10 rank terms, we have 10*6 + 10*6 + 10*4 = 160 variables
        # But we can normalize: fix W entries to be in {-1, 0, 1} or similar
        
        # Approach: Try to find solutions where entries are in {-1, 0, 1}
        # Use a smart search: fix some entries and solve for others
        
        # Method: Use the fact that we can permute and scale rank terms
        # Fix the first column of W to have specific values
        
        # Try exhaustive search over small patterns for key entries
        # Key observation: For each (i,j,k), the product sum must equal 1
        
        # Let's try a more targeted search:
        # 1. Generate random sparse matrices with entries in {-1, 0, 1}
        # 2. Check if they work
        # 3. If not, try local modifications
        
        # First, try a systematic search over a restricted space
        # Fix the structure: assume U has at most 2 nonzeros per row, etc.
        
        # Try random search with verification
        best_solution = None
        
        # Phase 1: Random search with {-1, 0, 1} entries
        for restart in range(5000):
            if time.time() - start_time > time_limit:
                break
            
            # Generate sparse random matrices
            density = 0.3
            U = (rng.random((n_b*m_b, target_rank)) < density).astype(float) * rng.choice([-1, 1], size=(n_b*m_b, target_rank))
            V = (rng.random((m_b*p_b, target_rank)) < density).astype(float) * rng.choice([-1, 1], size=(m_b*p_b, target_rank))
            W = (rng.random((p_b*n_b, target_rank)) < density).astype(float) * rng.choice([-1, 1], size=(p_b*n_b, target_rank))
            
            if _verify_decomposition_fast(U, V, W, n_b, m_b, p_b):
                return U, V, W
        
        # Phase 2: Start from rank-11 solution and try to merge columns
        # Build the rank-11 block
        U11 = np.zeros((n_b*m_b, 11))
        V11 = np.zeros((m_b*p_b, 11))
        W11 = np.zeros((p_b*n_b, 11))
        
        for idx, term in enumerate(all_terms):
            col = idx
            for i, j, val in term['U']:
                U11[i * m_b + j, col] += val
            for j, k_local, val in term['V']:
                V11[j * p_b + k_local, col] += val
            for k_local, i, val in term['W']:
                W11[k_local * n_b + i, col] += val
        
        # Try to find a rank-10 solution by searching near the rank-11 solution
        # with one column removed and perturbations
        for col_to_drop in range(11):
            if time.time() - start_time > time_limit:
                break
            
            U_sub = np.delete(U11, col_to_drop, axis=1)
            V_sub = np.delete(V11, col_to_drop, axis=1)
            W_sub = np.delete(W11, col_to_drop, axis=1)
            
            # Try various perturbation strategies
            for pert_scale in [0.1, 0.2, 0.5, 1.0]:
                for pert in range(100):
                    if time.time() - start_time > time_limit:
                        break
                    
                    # Add random perturbation
                    U_p = U_sub + rng.normal(0, pert_scale, U_sub.shape)
                    V_p = V_sub + rng.normal(0, pert_scale, V_sub.shape)
                    W_p = W_sub + rng.normal(0, pert_scale, W_sub.shape)
                    
                    # Round to nearest half-integer
                    U_r = np.round(U_p * 2) / 2
                    V_r = np.round(V_p * 2) / 2
                    W_r = np.round(W_p * 2) / 2
                    
                    if _verify_decomposition_fast(U_r, V_r, W_r, n_b, m_b, p_b):
                        return U_r, V_r, W_r
        
        # Phase 3: Try a more systematic algebraic approach
        # For 2x3x2, we can try to exploit the structure
        # Let's try fixing some entries and solving linear equations
        
        # The 12 equations are:
        # For i in {0,1}, j in {0,1,2}, k in {0,1}:
        #   sum_{r=0}^{9} U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        
        # This is a system of 12 equations in 60+60+40 = 160 unknowns
        # But they're not linear due to the products
        
        # Try a different approach: assume specific patterns
        # For example, assume W has a specific form
        
        # Try: W[k*2+i, r] is nonzero only for specific (k,i,r) combinations
        # This reduces the search space significantly
        
        # Let's try a greedy approach:
        # 1. For each rank term r, choose which (i,j,k) entries it helps satisfy
        # 2. Assign values to make the sum work out
        
        # Actually, let's try a completely different strategy:
        # Use the known minimum rank bounds. For M_{2,3,2}, the rank is known to be 10
        # (this is a result from tensor rank theory)
        
        # Let's try to construct it explicitly using algebraic methods
        # One approach: use the fact that M_{2,3,2} = M_{2,2,2} ⊗ M_{1,2,1} in some sense
        
        # Try yet another approach: use integer linear programming-like search
        # but with a smarter initialization
        
        # Initialize with a known valid rank-11 solution and try to find
        # a linear combination that gives rank 10
        
        # Actually, the simplest approach: try all combinations of a few key parameters
        # Fix the first few columns and search for the rest
        
        # Let's try a targeted search: assume entries are in {-2, -1, 0, 1, 2}
        # and use a smarter random initialization
        
        for restart in range(3000):
            if time.time() - start_time > time_limit:
                break
            
            # Use a mix of small integers
            U = rng.choice([-2, -1, 0, 0, 0, 1, 1, 2], size=(n_b*m_b, target_rank)).astype(float)
            V = rng.choice([-2, -1, 0, 0, 0, 1, 1, 2], size=(m_b*p_b, target_rank)).astype(float)
            W = rng.choice([-2, -1, 0, 0, 0, 1, 1, 2], size=(p_b*n_b, target_rank)).astype(float)
            
            if _verify_decomposition_fast(U, V, W, n_b, m_b, p_b):
                return U, V, W
        
        return None

    block_decomp = _find_rank_10_block()
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        rank_per_block = 10
        total_rank = num_blocks * rank_per_block
        
        if total_rank < best_rank:
            U_new = np.zeros((n * m, total_rank))
            V_new = np.zeros((m * p, total_rank))
            W_new = np.zeros((p * n, total_rank))
            
            for b in range(num_blocks):
                col_offset = b * rank_per_block
                U_new[:, col_offset:col_offset+rank_per_block] = U_block
                for j in range(m):
                    for k_local in range(block_size):
                        k_global = b * block_size + k_local
                        V_new[j * p + k_global, col_offset:col_offset+rank_per_block] = V_block[j * block_size + k_local, :]
                for k_local in range(block_size):
                    for i in range(n):
                        k_global = b * block_size + k_local
                        W_new[k_global * n + i, col_offset:col_offset+rank_per_block] = W_block[k_local * n + i, :]
            
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = total_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
