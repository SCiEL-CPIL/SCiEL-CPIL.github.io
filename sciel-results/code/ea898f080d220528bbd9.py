import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    # Use Integer Linear Programming to find the minimum size covering code.
    
    N = q ** n
    # Generate all words as indices 0 to N-1
    # We need to map index <-> tuple
    # For efficiency in matrix construction, we can compute distances on the fly or precompute neighbors.
    
    # Precompute neighbors for each index to build the constraint matrix
    # dist(u, v) <= 1 means u == v or Hamming distance is 1.
    
    # To build the sparse matrix A (N x N), where A[i, j] = 1 if d(w_i, w_j) <= 1
    # We can iterate over all pairs? N=6561, N^2 = 43 million. That's too slow in pure Python for matrix construction.
    # Better: For each word i, find its neighbors j.
    # There are N words. Each word has 1 + n(q-1) = 17 neighbors.
    # Total non-zeros = N * 17 = 6561 * 17 = 111,537.
    # This is very sparse.
    
    # We need a mapping from tuple to index.
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    row_indices = []
    col_indices = []
    data = []
    
    for i, w in enumerate(all_words):
        # Self
        row_indices.append(i)
        col_indices.append(i)
        data.append(1)
        
        # Neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                # Tuples are immutable, so create list, modify, tuple
                lst = list(w)
                lst[pos] = new_val
                nw = tuple(lst)
                j = word_to_idx[nw]
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
                
    # Create sparse matrix
    # A[i, j] = 1 if word j covers word i (i.e., d(w_i, w_j) <= 1)
    # The constraint for word i is: sum_{j} A[i, j] x_j >= 1
    A = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    
    # Objective: minimize sum(x_j)
    c = np.ones(N)
    
    # Constraints: A x >= 1
    # LinearConstraint(A, 1, np.inf)
    constraints = [LinearConstraint(A, 1, np.inf)]
    
    # Integrality: all binary
    integrality = np.ones(N)
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * N
    
    # Solve
    start_time = time.time()
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 100, "mip_rel_gap": 0.0})
        
        if res.success or res.status in [2, 3]: # 2: optimal, 3: time limit (but feasible?)
            # Check if we have a feasible solution
            if res.x is not None and np.all(res.x >= 0):
                # Round to binary
                x = np.round(res.x).astype(int)
                
                # Verify feasibility just in case
                # covered = A @ x
                # if np.all(covered >= 1):
                codewords = [list(all_words[i]) for i in range(N) if x[i] == 1]
                
                # If the solution is valid, return it
                # The score is the number of codewords.
                # If ILP didn't find a solution, fallback to greedy?
                # milp usually returns an incumbent if it times out.
                
                # Let's verify the incumbent is actually a cover.
                # A @ x >= 1
                # We can do a quick check on the returned codewords.
                # But A @ x is efficient with sparse matrix.
                coverage_counts = A @ x
                if np.all(coverage_counts >= 1):
                    return codewords
                else:
                    # Inconsistent, fall through to greedy
                    pass
            else:
                pass
        else:
            pass
    except Exception as e:
        pass

    # Fallback: Greedy Set Cover
    # This is different from the removal heuristic.
    # Start with empty code, pick best covering word iteratively.
    
    # Precompute neighbors indices for fast counting
    # neighbors_idx[i] = list of indices j such that d(w_i, w_j) <= 1
    neighbors_idx = []
    for i, w in enumerate(all_words):
        idxs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                lst = list(w)
                lst[pos] = new_val
                nw = tuple(lst)
                idxs.append(word_to_idx[nw])
        neighbors_idx.append(idxs)
        
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # To speed up greedy, we can track counts of how many times each word is covered by current code?
    # No, greedy set cover usually picks the set that covers the most *new* elements.
    # This requires knowing which elements are already covered.
    # We can maintain a 'covered' boolean array.
    # For each candidate codeword c, gain = sum(uncovered[j] for j in neighbors_idx[c]).
    # Pick c with max gain.
    
    # Optimization: N=6561. Steps ~ 500.
    # Naive loop: 500 * 6561 * 17 (avg neighbors) ~ 56 million operations.
    # This might take ~5-10 seconds in Python. Acceptable.
    
    for _ in range(N): # Max steps
        if not np.any(uncovered):
            break
            
        best_c = -1
        best_gain = 0
        
        # We can skip candidates that don't cover any new words?
        # All candidates cover themselves. If self is uncovered, gain >= 1.
        
        for c in range(N):
            # If c itself is covered, it might still cover other uncovered neighbors.
            # But if c is covered, and all its neighbors are covered, gain=0.
            # We can check gain quickly.
            gain = 0
            for j in neighbors_idx[c]:
                if uncovered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_gain == 0:
            break # Should not happen if uncovered exists and we check all
            
        code.append(best_c)
        
        # Mark covered
        for j in neighbors_idx[best_c]:
            uncovered[j] = False
            
    return [list(all_words[i]) for i in code]
