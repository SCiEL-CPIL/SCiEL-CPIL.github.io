# EVOLVE-BLOCK-START
import time
import random
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use exact integer local search minimizing the number of violated constraints.
    3. Efficiently update errors only for affected triples.
    4. Multiple random restarts with different sparse initializations.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Active triples
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    all_triples = list(itertools.product(range(nm), range(mp), range(pn)))
    active_set = set(active_triples)
    
    # Precompute indices for fast lookup
    # For each a, list of (b, c) triples
    # For each b, list of (a, c) triples
    # For each c, list of (a, b) triples
    # Actually, we just need to iterate.
    
    def count_errors(U, V, W, r):
        """
        Count number of violated triples.
        Returns (error_count, error_dict) where error_dict maps triple -> error_value
        """
        errors = {}
        err_count = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0
                    for t in range(r):
                        s += U[a][t] * V[b][t] * W[c][t]
                    expected = 1 if (a, b, c) in active_set else 0
                    if s != expected:
                        err_count += 1
                        errors[(a,b,c)] = s - expected
        return err_count, errors

    def local_search_exact(U, V, W, r, max_iters=20000, time_limit=5.0):
        """
        Local search minimizing error count.
        """
        start_ls = time.time()
        curr_err_count, curr_errors = count_errors(U, V, W, r)
        best_err_count = curr_err_count
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        # If already perfect, return
        if curr_err_count == 0:
            return U, V, W, 0
        
        for it in range(max_iters):
            if time.time() > time_limit or time.time() > deadline:
                break
            
            # Pick a random move
            # 1. Choose a matrix (U, V, W)
            # 2. Choose a row
            # 3. Choose a column
            # 4. Choose a delta
            
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                nrows = nm
            elif mat_idx == 1:
                mat = V
                nrows = mp
            else:
                mat = W
                nrows = pn
            
            row = random.randint(0, nrows - 1)
            col = random.randint(0, r - 1)
            
            # Try deltas -1, 0, 1, 2, -2
            # We want to minimize error count.
            # Changing mat[row][col] affects many triples.
            # Let's just try a small set of deltas.
            old_val = mat[row][col]
            
            # Determine affected triples
            # If mat is U (row=a), affected triples are (a, b, c) for all b, c
            # If mat is V (row=b), affected triples are (a, b, c) for all a, c
            # If mat is W (row=c), affected triples are (a, b, c) for all a, b
            
            affected = []
            if mat_idx == 0:
                a = row
                for b in range(mp):
                    for c in range(pn):
                        affected.append((a, b, c))
            elif mat_idx == 1:
                b = row
                for a in range(nm):
                    for c in range(pn):
                        affected.append((a, b, c))
            else:
                c = row
                for a in range(nm):
                    for b in range(mp):
                        affected.append((a, b, c))
            
            # Calculate current errors for affected triples
            curr_affected_errors = {}
            for tri in affected:
                curr_affected_errors[tri] = curr_errors.get(tri, 0) # 0 if not in errors dict means it was correct (error 0)
                # Wait, if it's not in curr_errors, error is 0.
                # So we need to know the current sum to update it.
                # Let's store the sum, not just the error.
            
            # This approach is getting complicated to update incrementally without storing sums.
            # Let's just re-evaluate the affected triples for a candidate move.
            # Since the number of affected triples is ~144, and we try ~5 deltas,
            # that's ~720 evaluations per iteration.
            # Each evaluation is O(r) ~ O(20-30).
            # 720 * 30 = 21600 ops per iteration.
            # 20000 iters * 21600 = 432M ops. Too slow in Python.
            
            # Optimization: Only consider moves that fix the *worst* error?
            # Or just accept if error count drops.
            
            # Let's try a simpler heuristic:
            # Pick a random violated triple.
            # Pick a random entry in one of its factors.
            # Try to adjust it to fix the triple.
            
            if not curr_errors:
                break
                
            # Pick a random violated triple
            violated_triples = list(curr_errors.keys())
            if not violated_triples:
                break
            a, b, c = random.choice(violated_triples)
            curr_sum = curr_errors[(a,b,c)] + (1 if (a,b,c) in active_set else 0)
            
            # Try to adjust U[a][t], V[b][t], or W[c][t]
            # We need to know the current sum.
            # Let's pick a random column t.
            t = random.randint(0, r - 1)
            
            # Try adjusting U[a][t]
            # New sum = old_sum - U[a][t]V[b][t]W[c][t] + (U[a][t]+d)V[b][t]W[c][t]
            #           = old_sum + d * V[b][t] * W[c][t]
            # We want New Sum == Target (1 or 0).
            # So d * V[b][t] * W[c][t] == Target - old_sum = -curr_error
            
            # This is a linear equation for d.
            # If V[b][t]*W[c][t] is non-zero, we can solve for d.
            # d = -curr_error / (V[b][t]*W[c][t])
            # If d is an integer, we can try it.
            
            # Try this for U, V, W
            found_move = False
            
            # Try U
            prod_vw = V[b][t] * W[c][t]
            if prod_vw != 0:
                # d = -curr_error / prod_vw
                if curr_error_divisible := (-curr_errors[(a,b,c)]) % prod_vw == 0:
                    d = -curr_errors[(a,b,c)] // prod_vw
                    # Check if this d is "small" (e.g., -2 to 2) to avoid explosion
                    if abs(d) <= 2:
                        # Apply move
                        old_val = U[a][t]
                        U[a][t] = old_val + d
                        
                        # Check if total error count decreased
                        # We only need to check affected triples.
                        # But calculating the change in error count is tricky.
                        # Let's just compute the new error count for affected triples.
                        
                        # Actually, let's just do a full re-evaluation of the error count
                        # if the move looks promising? No, too slow.
                        
                        # Let's estimate:
                        # The triple (a,b,c) is fixed.
                        # Other triples might be broken.
                        
                        # Let's just try the move and check if the *specific* triple is fixed.
                        # And hope for the best? No.
                        
                        # Let's revert for now and use a simpler, slower but more robust method:
                        # Random Hill Climbing on Error Count with full re-evaluation of affected set.
                        # But optimize the evaluation.
                        
                        pass # Skip this complex logic for now, use simpler random search below
            
            # Simpler approach: Random Perturbation
            # Pick random entry, random delta.
            # Compute new error count for affected triples.
            # If better, accept.
            
            # To make this fast, we need to store the current sums.
            # sums[a,b,c]
            
            # Let's restart with a pre-computed sums array.
            # This requires O(nm*mp*pn) memory, which is 36*12*12 = 5184 floats. Small.
            
            break # Break out of this complex inner loop and implement the sums-based approach outside
        
        return best_U, best_V, best_W, best_err_count
    
    # Re-implement local search with sums array
    def local_search_with_sums(U, V, W, r, max_iters=50000, time_limit=5.0):
        start_ls = time.time()
        nm, mp, pn = n*m, m*p, p*n
        active_set_local = active_set
        
        # Initialize sums
        sums = [[[0 for _ in range(pn)] for _ in range(mp)] for _ in range(nm)]
        # This is a 3D list. sums[a][b][c]
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0
                    for t in range(r):
                        s += U[a][t] * V[b][t] * W[c][t]
                    sums[a][b][c] = s
        
        # Count errors
        def get_error_count():
            cnt = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        expected = 1 if (a, b, c) in active_set_local else 0
                        if sums[a][b][c] != expected:
                            cnt += 1
            return cnt
        
        curr_err = get_error_count()
        best_err = curr_err
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        if curr_err == 0:
            return U, V, W, 0
        
        # Precompute affected indices
        # For each a, list of (b, c)
        # For each b, list of (a, c)
        # For each c, list of (a, b)
        
        for it in range(max_iters):
            if time.time() > time_limit or time.time() > deadline:
                break
            
            # Random move
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, nm - 1)
                col = random.randint(0, r - 1)
                old_val = U[row][col]
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if abs(new_val) > 10: continue
                
                # Update sums
                # Affected: (row, b, c) for all b, c
                # Change in sum: delta * V[b][col] * W[c][col]
                improvement = 0
                for b in range(mp):
                    for c in range(pn):
                        expected = 1 if (row, b, c) in active_set_local else 0
                        old_sum = sums[row][b][c]
                        if old_sum == expected:
                            old_err = 0
                        else:
                            old_err = 1
                            
                        delta_sum = delta * V[b][col] * W[c][col]
                        new_sum = old_sum + delta_sum
                        if new_sum == expected:
                            new_err = 0
                        else:
                            new_err = 1
                            
                        improvement += old_err - new_err
                        sums[row][b][c] = new_sum
                        
                if improvement > 0:
                    U[row][col] = new_val
                    curr_err -= improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                else:
                    # Revert sums
                    for b in range(mp):
                        for c in range(pn):
                            delta_sum = delta * V[b][col] * W[c][col]
                            sums[row][b][c] -= delta_sum
                            
            elif mat_idx == 1:
                row = random.randint(0, mp - 1)
                col = random.randint(0, r - 1)
                old_val = V[row][col]
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if abs(new_val) > 10: continue
                
                improvement = 0
                for a in range(nm):
                    for c in range(pn):
                        expected = 1 if (a, row, c) in active_set_local else 0
                        old_sum = sums[a][row][c]
                        if old_sum == expected:
                            old_err = 0
                        else:
                            old_err = 1
                            
                        delta_sum = U[a][col] * delta * W[c][col]
                        new_sum = old_sum + delta_sum
                        if new_sum == expected:
                            new_err = 0
                        else:
                            new_err = 1
                            
                        improvement += old_err - new_err
                        sums[a][row][c] = new_sum
                        
                if improvement > 0:
                    V[row][col] = new_val
                    curr_err -= improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                else:
                    for a in range(nm):
                        for c in range(pn):
                            delta_sum = U[a][col] * delta * W[c][col]
                            sums[a][row][c] -= delta_sum
                            
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, r - 1)
                old_val = W[row][col]
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if abs(new_val) > 10: continue
                
                improvement = 0
                for a in range(nm):
                    for b in range(mp):
                        expected = 1 if (a, b, row) in active_set_local else 0
                        old_sum = sums[a][b][row]
                        if old_sum == expected:
                            old_err = 0
                        else:
                            old_err = 1
                            
                        delta_sum = U[a][col] * V[b][col] * delta
                        new_sum = old_sum + delta_sum
                        if new_sum == expected:
                            new_err = 0
                        else:
                            new_err = 1
                            
                        improvement += old_err - new_err
                        sums[a][b][row] = new_sum
                        
                if improvement > 0:
                    W[row][col] = new_val
                    curr_err -= improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                else:
                    for a in range(nm):
                        for b in range(mp):
                            delta_sum = U[a][col] * V[b][col] * delta
                            sums[a][b][row] -= delta_sum
                            
        return best_U, best_V, best_W, best_err
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank(r, max_attempts=50):
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialization
            U = [[0] * r for _ in range(nm)]
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]
            
            # Sparse random init
            for a in range(nm):
                k = random.randint(1, 3)
                cols = random.sample(range(r), k)
                for c in cols:
                    U[a][c] = random.choice([-1, 1])
            for b in range(mp):
                k = random.randint(1, 3)
                cols = random.sample(range(r), k)
                for c in cols:
                    V[b][c] = random.choice([-1, 1])
            for c in range(pn):
                k = random.randint(1, 3)
                cols = random.sample(range(r), k)
                for c2 in cols:
                    W[c][c2] = random.choice([-1, 1])
            
            # Local search
            U, V, W, err = local_search_with_sums(U, V, W, r, max_iters=10000, time_limit=2.0)
            
            if err == 0:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    for r in range(18, best_rank):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r)
        if result:
            best_result = result
            best_rank = r
            break
            
    return best_result
# EVOLVE-BLOCK-END
