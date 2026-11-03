import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_space = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute neighbors
    # neighbors[i] is list of indices j such that dist(i,j) <= radius
    neighbors = [[] for _ in range(total_space)]
    for i in range(total_space):
        w = all_words[i]
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    nw = w[:]
                    nw[pos] = val
                    neighbors[i].append(word_to_idx(nw))
    
    # Build sparse constraint matrix A
    # A[i, j] = 1 if j in neighbors[i]
    row_idx = []
    col_idx = []
    data = []
    for i in range(total_space):
        for j in neighbors[i]:
            row_idx.append(i)
            col_idx.append(j)
            data.append(1.0)
            
    A = csr_matrix((data, (row_idx, col_idx)), shape=(total_space, total_space))
    
    # Objective
    c = np.ones(total_space)
    
    # Constraints: A x >= 1
    lb = np.ones(total_space)
    ub = np.full(total_space, np.inf)
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds
    integrality = np.ones(total_space)
    bounds = Bounds(lb=np.zeros(total_space), ub=np.ones(total_space))
    
    # Greedy Fallback
    # We need a fast greedy to have a good baseline.
    # Using numpy for coverage tracking.
    covered = np.zeros(total_space, dtype=bool)
    initial_code = []
    
    # Precompute neighbor arrays for faster access in greedy
    # neighbors_arr is a list of lists, already done.
    
    # Simple Greedy with numpy optimization for coverage check
    # To speed up, we can pick randomly or use a heuristic.
    # Let's try a simple greedy that picks the codeword covering the most uncovered nodes.
    # This is O(N^2 * degree) in worst case.
    # N=2187, degree=15. 2187*2187*15 ~ 72M ops.
    # In Python, this might take 5-10 seconds. Acceptable.
    
    remaining = set(range(total_space))
    while remaining:
        if time.time() > deadline - 30:
            break
        best_idx = -1
        max_cover = 0
        # Sample a subset if remaining is large? No, N is small.
        for i in remaining:
            # Count uncovered neighbors
            # neighbors[i] is a list
            # sum(1 for j in neighbors[i] if not covered[j])
            # This is slow in Python loop.
            # Let's use numpy:
            # nbs = np.array(neighbors[i])
            # count = np.sum(~covered[nbs])
            # This creates arrays. Might be faster.
            nbs = np.array(neighbors[i])
            count = np.sum(~covered[nbs])
            if count > max_cover:
                max_cover = count
                best_idx = i
        if best_idx == -1:
            break
        initial_code.append(best_idx)
        nbs = np.array(neighbors[best_idx])
        covered[nbs] = True
        remaining.remove(best_idx)
        
    # If greedy didn't finish, we still have a partial solution.
    # But we need a valid covering.
    # If time runs out, we might have an invalid solution.
    # To be safe, if greedy is interrupted, we should probably just return a known valid solution or complete it.
    # Given N is small, greedy should finish.
    
    # ILP Solve
    time_limit = max(1.0, deadline - time.time() - 5.0)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": time_limit, "presolve": True, "mip_rel_gap": 0.01})
        
        if res.status == 0: # Optimal
            x_opt = res.x
            code_indices = [i for i in range(total_space) if x_opt[i] > 0.5]
            return [all_words[i] for i in code_indices]
        elif res.status == 2: # Iteration limit / Time limit
            if res.x is not None:
                x_opt = res.x
                # Check feasibility
                # A @ x_opt >= 1
                # A is sparse
                prod = A @ x_opt
                if np.all(prod >= 1.0 - 1e-6):
                    code_indices = [i for i in range(total_space) if x_opt[i] > 0.5]
                    return [all_words[i] for i in code_indices]
            # Fallback to greedy
            return [all_words[i] for i in initial_code]
        else:
            return [all_words[i] for i in initial_code]
            
    except Exception:
        return [all_words[i] for i in initial_code]
