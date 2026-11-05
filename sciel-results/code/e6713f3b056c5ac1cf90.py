import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    # 150s wall-clock limit. 
    # scipy.milp can be slow. We need a safety margin.
    # Let's aim to finish by 140s to be safe.
    deadline = start_time + 140.0
    
    total_words = q ** n
    
    # Precompute digits for all words as a 2D array
    # This is needed for the final output and for potential symmetry breaking logic
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors (balls) as bitmasks
    # For q=3, n=8, total_words=6561. 
    # A bitmask of 6561 bits fits in ~102 64-bit integers.
    # Python integers are arbitrary precision, so this is efficient.
    
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
        
    # Generate balls
    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius (slower, but n=8 is small)
            # For the specific task q=3, n=8, r=1, the above branch is used.
            # We include this for robustness but it might be too slow for large n/r.
            # Given the 150s limit, if this takes too long, we might fail.
            # But for the target task, it's fast.
            import itertools
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * q_powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    # Symmetry Breaking:
    # The problem is invariant under translation in the group (Z_q)^n.
    # We can fix one codeword to be the zero word (index 0) without loss of generality.
    # This reduces the number of variables from N to N-1.
    # Let x_i be 1 if word i is in the code.
    # x_0 = 1.
    # Minimize sum(x_i) for i in 1..N-1 + 1.
    
    N_vars = total_words - 1
    var_indices = list(range(1, total_words))
    
    # Objective: Minimize sum of x_i
    c = np.ones(N_vars)
    
    # Constraints:
    # For every word w in {0..q-1}^n, the sum of x_c for all c in Ball(w) must be >= 1.
    # Note: Ball(0) contains 0. Since x_0=1, the constraint for w=0 is:
    # 1 + sum(x_c for c in Ball(0)\{0}) >= 1. This is always satisfied.
    # So we only need to generate constraints for w != 0.
    
    # However, we must be careful. If we fix x_0=1, we just remove variable 0.
    # The constraint for any w is:
    # (1 if 0 in Ball(w) else 0) + sum_{c in Ball(w), c != 0} x_c >= 1
    
    A_list = []
    b_list = []
    
    # We will build the constraint matrix in a sparse-like way, but scipy.milp takes dense or sparse.
    # Given N_vars ~ 6500 and constraints ~ 6500, a dense matrix is 6500x6500 ~ 42M floats ~ 336MB.
    # This might be too much memory or slow to construct.
    # Let's use scipy.sparse.
    
    from scipy.sparse import csr_matrix
    
    rows = []
    cols = []
    data = []
    
    # Map variable index (1..N-1) to column index (0..N_vars-1)
    # col_idx = word_idx - 1
    
    for w in range(1, total_words):
        ball_w = balls[w]
        # Check if 0 is in ball_w
        # 0 is in ball_w if dist(w, 0) <= radius.
        # If yes, the constant term is 1.
        # Constraint: sum_{c in ball_w, c != 0} x_c >= 1 - (1 if 0 in ball_w else 0)
        
        # Calculate if 0 is in ball_w
        # dist(w, 0) is the number of non-zero digits in w.
        # For radius 1, 0 is in ball_w iff w has at most 1 non-zero digit.
        
        # Let's just check the bitmask for bit 0.
        if (ball_w >> 0) & 1:
            rhs = 0.0
        else:
            rhs = 1.0
            
        # Add constraint row
        # We need to iterate through set bits in ball_w, excluding 0.
        # ball_w is a Python int.
        # We can iterate through the bits.
        
        # Optimization: 
        # If rhs is 0, the constraint is sum(x_c) >= 0, which is always true since x_c >= 0.
        # So we can skip adding this constraint entirely!
        # This is a huge speedup.
        # For radius 1, how many words w have dist(w,0) <= 1?
        # 1 (zero word) + n*(q-1) = 1 + 8*2 = 17 words.
        # So we skip 16 constraints (w=1..16 roughly, depending on order).
        # Actually, we skip all w where 0 is in ball_w.
        
        if rhs == 0.0:
            continue
            
        # Now, add the row for this w.
        # Iterate through set bits in ball_w
        m = ball_w
        while m:
            # Get lowest set bit
            lb = m & (-m)
            idx = lb.bit_length() - 1
            m ^= lb
            if idx == 0:
                continue # Skip x_0
            # idx is the word index. Variable index is idx - 1.
            rows.append(w) # Row index in the constraint matrix (just use w as ID)
            cols.append(idx - 1)
            data.append(1.0)
            
    # Construct sparse matrix
    # Note: rows are indices 1..total_words-1. We can use w-1 as row index for 0-based.
    # Let's shift rows to 0..total_words-2.
    rows = [r - 1 for r in rows]
    
    A = csr_matrix((data, (rows, cols)), shape=(total_words - 1, N_vars))
    
    # Constraints: A @ x >= b
    # b is 1 for all rows included.
    b = np.ones(A.shape[0])
    
    # LinearConstraint: A @ x >= b  =>  -A @ x <= -b
    # Or use LinearConstraint(A, b, np.inf) ? 
    # scipy LinearConstraint(A, lb, ub) enforces lb <= A @ x <= ub.
    # So LinearConstraint(A, b, np.inf) means b <= A @ x.
    
    cons = LinearConstraint(A, b, np.inf)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: all variables are binary (integer)
    integrality = np.ones(N_vars)
    
    # Solve
    # time_limit is in seconds.
    # We have about 140s - elapsed.
    elapsed = time.time() - start_time
    time_left = deadline - elapsed - 5.0 # 5s buffer for post-processing
    
    if time_left <= 0:
        # Fallback: Greedy
        return _greedy_fallback(q, n, radius, balls, digits_of, total_words)
        
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": time_left, "presolve": True})
        
        if res.status == 0 or res.status == 1: # 0=optimal, 1=feasible
            x = res.x
            # x contains values for variables 1..N-1.
            # x_0 is fixed to 1.
            code_indices = [0]
            for i in range(N_vars):
                if x[i] > 0.5:
                    code_indices.append(i + 1)
            
            # Verify validity
            cover_mask = 0
            for idx in code_indices:
                cover_mask |= balls[idx]
                
            all_mask = (1 << total_words) - 1
            if cover_mask != all_mask:
                # If ILP returned an invalid solution (should not happen, but safety)
                # Or if it timed out and returned a non-covering feasible solution? 
                # milp requires feasibility. If it times out, it might return a heuristic solution.
                # If that solution is not a cover, we must fix it.
                # The simplest fix is to add the missing words.
                # This might increase the size, but ensures validity.
                
                # Identify uncovered words
                uncovered = all_mask & ~cover_mask
                while uncovered:
                    lb = uncovered & (-uncovered)
                    idx = lb.bit_length() - 1
                    uncovered ^= lb
                    code_indices.append(idx)
                    cover_mask |= balls[idx]
                    
            # Remove duplicates (just in case)
            code_indices = list(set(code_indices))
            
            # Convert to list of lists
            return [list(digits_of[i]) for i in code_indices]
            
        else:
            # No feasible solution found? Or error.
            # Fallback to greedy
            return _greedy_fallback(q, n, radius, balls, digits_of, total_words)
            
    except Exception as e:
        # Fallback to greedy
        return _greedy_fallback(q, n, radius, balls, digits_of, total_words)

def _greedy_fallback(q, n, radius, balls, digits_of, total_words):
    """
    Fast greedy set cover fallback.
    """
    all_mask = (1 << total_words) - 1
    uncovered = all_mask
    code = []
    
    # Simple greedy: pick word covering most uncovered.
    # To speed up, we can just pick the first word that covers a lot?
    # No, standard greedy is better.
    
    # Optimization: Precompute "gain" is hard to update fast without bit operations.
    # But for 6561 words, it's okay.
    
    while uncovered:
        best_idx = -1
        max_gain = -1
        
        for i in range(total_words):
            if i in code: # O(N) check. Slow.
                continue
            gain = (balls[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        code.append(best_idx)
        uncovered &= ~balls[best_idx]
        
    return [list(digits_of[i]) for i in code]
