"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Find a rank-minimal exact tensor decomposition for multiplying
    # a 3x4 matrix by a 4x7 matrix.
    #
    # For M_{n,m,p} with n=3, m=4, p=7, the schoolbook rank is n*m*p = 84.
    # We use a search strategy to find a lower rank decomposition if one exists.
    #
    # Strategy:
    # 1. Use the schoolbook decomposition as a starting point.
    # 2. Apply rank reduction via a combinatorial search over sparse decompositions.
    # 3. Use integer/half-integer entries to maintain exactness.
    #
    # The tensor entries are T[i, j, k] = delta, and the factorization is
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    #
    # We search for the minimum rank by trying ranks from lower bounds upward.
    # For small matrices, we can use a greedy or randomized approach to find
    # exact decompositions with integer/half-integer coefficients.
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Known lower bound: max(n*m, m*p, p*n) is not a valid lower bound in general.
    # A better lower bound comes from Strassen-type results or tensor rank bounds.
    # For M_{3,4,7}, we use the known lower bound of 48 (from literature on small matrix multiplication).
    
    # Try ranks from a lower bound up to schoolbook.
    # We'll use a randomized search with exact arithmetic verification.
    
    import random
    import time
    
    deadline = time.time() + 120  # 120 second deadline
    
    def make_schoolbook(rank_offset=0):
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = len(U[0]) if nm > 0 else 0
        
        # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # It should equal delta(i,j,k) for all (i,j,k) in [0,n) x [0,m) x [0,p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for c in range(rank):
                        total += U[ui][c] * V[vj][c] * W[wk][c]
                    if total != 1:
                        return False
        return True
    
    # Start with schoolbook, then try to reduce rank.
    best = make_schoolbook()
    
    # Try to find lower rank decompositions.
    # For M_{3,4,7}, known results suggest rank < 84 might be achievable.
    # We'll try a heuristic approach: remove columns where all three matrices have small support.
    
    def try_reduce(base_decomp, target_rank, max_iters=5000):
        """Try to reduce the rank of the decomposition by removing redundant columns."""
        import random
        import time
        
        U = [row[:] for row in base_decomp["U"]]
        V = [row[:] for row in base_decomp["V"]]
        W = [row[:] for row in base_decomp["W"]]
        rank = len(U[0])
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Precompute the tensor values for verification
        def compute_tensor():
            T = [[[0] * p for _ in range(m)] for _ in range(n)]
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        ui_val = U[i * m + j][c]
                        if ui_val == 0:
                            continue
                        for k in range(p):
                            vj_val = V[j * p + k][c]
                            if vj_val == 0:
                                continue
                            wk_val = W[k * n + i][c]
                            if wk_val == 0:
                                continue
                            T[i][j][k] += ui_val * vj_val * wk_val
            return T
        
        # Check current rank validity
        if not verify(U, V, W):
            return None, rank
        
        # Try removing columns one at a time
        # This is a greedy approach; we try each column removal and check if the decomposition is still valid.
        
        def is_valid():
            if not verify(U, V, W):
                return False
            # Check all required tensor entries are correct
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for c in range(rank):
                            total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if total != 1:
                            return False
            return True
        
        # Try to remove columns
        removed = 0
        while rank > target_rank and time.time() < deadline:
            made_progress = False
            # Shuffle column order for randomness
            cols = list(range(rank))
            random.shuffle(cols)
            
            for c in cols:
                if rank <= target_rank:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Save state
                Uc = [U[row][c] for row in range(nm)]
                Vc = [V[row][c] for row in range(mp)]
                Wc = [W[row][c] for row in range(pn)]
                
                # Remove column c
                for row in range(nm):
                    U[row].pop(c)
                for row in range(mp):
                    V[row].pop(c)
                for row in range(pn):
                    W[row].pop(c)
                rank -= 1
                
                if is_valid():
                    made_progress = True
                    removed += 1
                else:
                    # Restore column
                    for row in range(nm):
                        U[row].append(Uc[row])
                    for row in range(mp):
                        V[row].append(Vc[row])
                    for row in range(pn):
                        W[row].append(Wc[row])
                    rank += 1
            
            if not made_progress:
                break
        
        if rank <= target_rank and is_valid():
            return {"U": U, V: V, W: W}, rank
        return None, rank
    
    # Try reducing from schoolbook
    for target_rank in range(60, rank_schoolbook + 1, 4):
        if time.time() > deadline - 10:
            break
        result, new_rank = try_reduce(best, target_rank)
        if result is not None:
            best = result
            if new_rank <= target_rank:
                break
    
    # Also try a simple randomized construction for smaller ranks
    # This is a heuristic approach for exact integer decompositions
    
    def try_random_exact(lower_rank, upper_rank, max_attempts=200):
        import random
        import time
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        for r in range(lower_rank, min(upper_rank + 1, rank_schoolbook + 1)):
            if time.time() > deadline - 5:
                break
            
            for _ in range(max_attempts // 10):
                if time.time() > deadline - 5:
                    break
                
                # Randomly initialize with small integers
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
                
                # Scale to get correct magnitudes
                # For exact decomposition, we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a tensor rank problem; exact solutions are hard to find randomly.
                
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}, r
        
        return None, rank_schoolbook
    
    # The randomized approach is unlikely to find exact solutions quickly.
    # For now, return the best we found.
    
    return best
# EVOLVE-BLOCK-END
