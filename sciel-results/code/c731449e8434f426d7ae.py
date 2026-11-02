import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 30
    W_dim = p * n  # 12
    
    # Build target tensor
    target = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        rank = U.shape[1]
        T_rec = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
        for r in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_rec, target, atol=1e-10)
    
    def verify_sparse(U, V, W):
        """Faster verification using sparse structure."""
        rank = U.shape[1]
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Check a sample of non-target entries
        import random
        random.seed(42)
        for _ in range(200):
            u = random.randint(0, U_dim - 1)
            v = random.randint(0, V_dim - 1)
            w = random.randint(0, W_dim - 1)
            # Check if this is a target entry
            is_target = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i * m + j == u and j * p + k == v and k * n + i == w):
                            is_target = True
                            break
                    if is_target:
                        break
                if is_target:
                    break
            if not is_target:
                val = 0.0
                for r in range(rank):
                    val += U[u, r] * V[v, r] * W[w, r]
                if abs(val) > 1e-10:
                    return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=np.float64)
        V = np.zeros((V_dim, rank), dtype=np.float64)
        W = np.zeros((W_dim, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = schoolbook()
    
    # Strassen-based decomposition for 2x5x6
    # Split A (2x5) into blocks, B (5x6) into blocks
    # We can use Strassen for 2x2x2 subproblems
    
    def strassen_2x5x6():
        """
        Use Strassen's algorithm structure.
        A is 2x5, B is 5x6, C is 2x6.
        
        Split A into A1 (row 0), A2 (row 1), each 1x5
        Split B into B1 (cols 0-2), B2 (cols 3-5), each 5x3
        Or better: think of it as multiple 1x5 x 5x1 products
        
        Actually, let's use a different approach:
        Group the computation into blocks where Strassen applies.
        
        For 2x5x6:
        - We have 2 rows in A, 5 intermediate, 6 columns in B
        - Think of it as 2 * 5 * 6 = 60 scalar multiplications
        
        Strassen reduces 2x2x2 from 8 to 7 multiplications.
        We can find 2x2x2 substructures.
        
        Let me try a different decomposition:
        Split the 5 intermediate dimension into groups.
        Split the 6 output columns into groups of 2 (3 groups).
        
        For each group of 2 output columns, we compute a 2x2x5 subproblem
        (2 rows of A, 5 intermediates, 2 columns of B).
        
        For 2x2x5: use Strassen on the 2x2 part (rows x cols), treating
        the 5 intermediates as the "depth".
        
        Actually, let me just try to find a good decomposition by
        systematic search over structured patterns.
        """
        # Start with schoolbook and try to merge terms
        U, V, W, rank = schoolbook()
        
        # Try to find linear dependencies and merge
        # This is complex, so let's try a direct approach
        
        # Alternative: use known results for small matrix multiplication
        # For 2x5x6, the tensor rank is at most 54 (Strassen-based)
        
        # Let me construct a Strassen-based decomposition:
        # Group the 6 output columns into 3 pairs: (0,1), (2,3), (4,5)
        # For each pair, we compute C[:, pair] = A @ B[:, pair]
        # This is a 2x5 x 5x2 = 2x5x2 subproblem
        
        # For 2x5x2: use Strassen on the 2x2 (rows x cols) part
        # A is 2x5, B is 5x2
        # Let A = [A0; A1] where A0, A1 are 1x5
        # Let B = [B0, B1] where B0, B1 are 5x1
        # C00 = A0*B0, C01 = A0*B1, C10 = A1*B0, C11 = A1*B1
        
        # Strassen:
        # M1 = A0 * B0
        # M2 = (A0+A1) * B0
        # M3 = A0 * (B0-B1)
        # M4 = A1 * (B0+B1)
        # M5 = (A0+A1) * (B0-B1)
        # M6 = (A0-A1) * B1
        # M7 = A1 * B1
        # C00 = M1, C01 = M1+M3, C10 = M2-M1, C11 = M4-M2
        
        # But each "multiplication" here is a 1x5 x 5x1 = dot product
        # which requires 5 scalar multiplications.
        # So 7 dot products * 5 = 35 scalar mults per 2x5x2 block.
        # 3 blocks * 35 = 105? That's worse!
        
        # The issue: Strassen saves on the 2x2 structure, but each
        # "entry" is a vector, so we need to be more careful.
        
        # Let me try a different approach: just do systematic local search
        # from the schoolbook decomposition.
        
        return U, V, W, rank
    
    # Systematic local search: try to reduce rank by finding
    # linear combinations that can be merged
    
    start_time = time.time()
    time_limit = 140  # Leave 10s margin
    
    def try_reduce_rank(U, V, W, rank, max_iterations=10000):
        """Try to reduce rank by finding mergeable terms."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Try to find two terms that can be merged
            # Look for terms that have overlapping support
            found_merge = False
            
            for r1 in range(rank):
                if time.time() - start_time > time_limit:
                    break
                for r2 in range(r1 + 1, rank):
                    # Check if U[:, r1] and U[:, r2] are linearly dependent
                    # or if we can find a linear combination that simplifies
                    u1 = U[:, r1]
                    u2 = U[:, r2]
                    v1 = V[:, r1]
                    v2 = V[:, r2]
                    w1 = W[:, r1]
                    w2 = W[:, r2]
                    
                    # Check if u1 * v1 * w1 + u2 * v2 * w2 can be
                    # written as a single term or fewer terms
                    # This is generally hard, so let's try specific patterns
                    
                    # Pattern 1: u1 == u2, v1 == v2, w1 == w2 (exact duplicate)
                    if np.allclose(u1, u2) and np.allclose(v1, v2) and np.allclose(w1, w2):
                        # Merge: add the contributions
                        # But they're both 1, so this would give 2, not 1
                        # This shouldn't happen in a valid decomposition
                        continue
                    
                    # Pattern 2: u1 == u2, v1 == v2, but w1 != w2
                    # Then u1*v1*(w1+w2) is one term
                    if np.allclose(u1, u2) and np.allclose(v1, v2):
                        w_new = w1 + w2
                        if np.count_nonzero(w_new) <= 1:
                            # Can merge into one term
                            U[:, r1] = u1
                            V[:, r1] = v1
                            W[:, r1] = w_new
                            # Remove r2
                            U = np.delete(U, r2, axis=1)
                            V = np.delete(V, r2, axis=1)
                            W = np.delete(W, r2, axis=1)
                            rank -= 1
                            found_merge = True
                            break
                    
                    # Pattern 3: u1 == u2, w1 == w2, but v1 != v2
                    if np.allclose(u1, u2) and np.allclose(w1, w2):
                        v_new = v1 + v2
                        if np.count_nonzero(v_new) <= 1:
                            U[:, r1] = u1
                            V[:, r1] = v_new
                            W[:, r1] = w1
                            U = np.delete(U, r2, axis=1)
                            V = np.delete(V, r2, axis=1)
                            W = np.delete(W, r2, axis=1)
                            rank -= 1
                            found_merge = True
                            break
                    
                    # Pattern 4: v1 == v2, w1 == w2, but u1 != u2
                    if np.allclose(v1, v2) and np.allclose(w1, w2):
                        u_new = u1 + u2
                        if np.count_nonzero(u_new) <= 1:
                            U[:, r1] = u_new
                            V[:, r1] = v1
                            W[:, r1] = w1
                            U = np.delete(U, r2, axis=1)
                            V = np.delete(V, r2, axis=1)
                            W = np.delete(W, r2, axis=1)
                            rank -= 1
                            found_merge = True
                            break
                    
                    # Pattern 5: More general linear combinations
                    # Try U[:, r1] + U[:, r2], etc.
                    # This is getting complex, let's stop here for now
                    
                if found_merge:
                    break
            
            if not found_merge:
                # No more simple merges, try random perturbations
                import random
                random.seed(iteration)
                r = random.randint(0, rank - 1)
                # Try scaling one term
                scale = random.choice([0.5, -0.5, 2, -2])
                U[:, r] *= scale
                # Check if still valid
                if not verify_sparse(U, V, W):
                    U[:, r] /= scale
                else:
                    # Try to absorb into another term
                    for r2 in range(rank):
                        if r2 != r:
                            # Try U[:, r2] += U[:, r]
                            U[:, r2] += U[:, r]
                            if np.count_nonzero(U[:, r2]) <= 2:
                                if verify_sparse(U, V, W):
                                    # Remove r
                                    U = np.delete(U, r, axis=1)
                                    V = np.delete(V, r, axis=1)
                                    W = np.delete(W, r, axis=1)
                                    rank -= 1
                                    break
                            else:
                                U[:, r2] -= U[:, r]
            
            if rank <= 54:
                break
        
        return U, V, W, rank
    
    # Try to reduce from schoolbook
    U_try, V_try, W_try, rank_try = try_reduce_rank(U_best, V_best, W_best, rank_best)
    
    if verify(U_try, V_try, W_try) and rank_try < rank_best:
        U_best, V_best, W_best, rank_best = U_try, V_try, W_try, rank_try
    
    # Try more aggressive search: start from lower rank and build up
    # Use random initialization with structured patterns
    
    import random
    random.seed(42)
    
    for target_rank in range(54, rank_best + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try random search for this rank
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            U = np.zeros((U_dim, target_rank), dtype=np.float64)
            V = np.zeros((V_dim, target_rank), dtype=np.float64)
            W = np.zeros((W_dim, target_rank), dtype=np.float64)
            
            # Initialize with structured patterns
            for r in range(target_rank):
                # Random sparse initialization
                nz_u = random.randint(1, 3)
                nz_v = random.randint(1, 3)
                nz_w = random.randint(1, 2)
                
                u_indices = random.sample(range(U_dim), min(nz_u, U_dim))
                v_indices = random.sample(range(V_dim), min(nz_v, V_dim))
                w_indices = random.sample(range(W_dim), min(nz_w, W_dim))
                
                for idx in u_indices:
                    U[idx, r] = random.choice([-1, 1])
                for idx in v_indices:
                    V[idx, r] = random.choice([-1, 1])
                for idx in w_indices:
                    W[idx, r] = random.choice([-1, 1])
            
            if verify_sparse(U, V, W):
                U_best, V_best, W_best, rank_best = U, V, W, target_rank
                break
        
        if rank_best <= target_rank:
            break
    
    # Final verification
    if not verify(U_best, V_best, W_best):
        # Fallback to schoolbook
        U_best, V_best, W_best, rank_best = schoolbook()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

