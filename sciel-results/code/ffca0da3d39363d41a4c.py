import time
import numpy as np
import itertools
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx

    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    # Precompute balls as indices
    # ball[i] is a list of indices in the ball of word i
    balls = np.empty((N, 1 + n * (q - 1)), dtype=np.int32)
    for i, w in enumerate(all_words):
        ball = [i]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w_list)))
            w_list[pos] = old_val
        balls[i] = ball

    # 1. Fast Greedy
    covered = np.zeros(N, dtype=bool)
    code_indices = []
    
    while not covered.all():
        if time.time() > deadline - 10: # Stop greedy if close to deadline
            break
        # Vectorized gain calculation
        # ~covered is boolean array of N
        # balls is (N, ball_size)
        # ~covered[balls] is (N, ball_size) boolean
        # Sum over axis 1 gives gain for each candidate
        gains = np.sum(~covered[balls], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        # Random tie-breaking to allow different local optima if we restart
        idx = candidates[random.randint(0, len(candidates) - 1)]
        code_indices.append(idx)
        covered[balls[idx]] = True
        
    # 2. Local Search: Redundancy Removal
    # We maintain a count of how many codewords cover each word
    # If a word is covered by >= 2 codewords, it's "safe"
    # If all words in a codeword's ball are safe, the codeword is redundant
    
    def remove_redundant(code_indices):
        if not code_indices:
            return code_indices
        # Count coverage
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_indices:
            counts[balls[idx]] += 1
            
        new_code = []
        for idx in code_indices:
            # Check if all neighbors are covered by at least 2 codewords
            # If so, we can remove this one
            if np.all(counts[balls[idx]] >= 2):
                # Remove it
                counts[balls[idx]] -= 1
            else:
                new_code.append(idx)
        return new_code

    # Iterate removal a few times
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        old_len = len(code_indices)
        code_indices = remove_redundant(code_indices)
        if len(code_indices) == old_len:
            break

    # Check if we still cover everything (should be true if logic is correct)
    # But if we broke out of greedy early, we might not.
    # Let's verify coverage
    covered = np.zeros(N, dtype=bool)
    for idx in code_indices:
        covered[balls[idx]] = True
        
    if not covered.all():
        # If we uncovered something (e.g. due to early break or bug), finish greedy
        while not covered.all():
            gains = np.sum(~covered[balls], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[random.randint(0, len(candidates) - 1)]
            code_indices.append(idx)
            covered[balls[idx]] = True
        # And remove redundant again
        code_indices = remove_redundant(code_indices)

    current_best = len(code_indices)
    best_code = code_indices.copy()

    # 3. ILP Polish
    # Only run ILP if we have enough time
    if time.time() < deadline - 10:
        # Build sparse matrix A
        rows = []
        cols = []
        vals = []
        for i in range(N):
            for j in balls[i]:
                rows.append(i)
                cols.append(j)
                vals.append(1)
        
        A = lil_matrix((N, N), dtype=np.int8)
        for r, c, v in zip(rows, cols, vals):
            A[r, c] = v
        A_csr = A.tocsr()
        
        c_obj = np.ones(N, dtype=np.float64)
        constraints = LinearConstraint(A_csr, lb=np.ones(N, dtype=np.float64), ub=np.inf)
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        integrality = np.ones(N, dtype=np.int8)
        
        options = {
            'time_limit': max(1.0, deadline - time.time() - 5.0),
            'mip_rel_gap': 0.0
        }
        
        try:
            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
            if res.status == 0 or res.status == 2:
                if res.x is not None:
                    x = np.round(res.x).astype(np.int8)
                    x = (x > 0.5).astype(np.int8)
                    ilp_code = np.where(x == 1)[0].tolist()
                    if len(ilp_code) < current_best:
                        current_best = len(ilp_code)
                        best_code = ilp_code
        except Exception:
            pass

    # Convert best_code to list of lists
    result = []
    for idx in best_code:
        result.append(list(idx_to_tuple(idx)))
        
    return result
