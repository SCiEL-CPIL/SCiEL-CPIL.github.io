import time
import random
import numpy as np
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin
    
    total_space = q ** n
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of word indices within Hamming distance 'radius' of i
    neighbors = [[] for _ in range(total_space)]
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Optimization: Precompute the set of neighbors for each position and digit change
    # But for n=7, q=3, total_space=2187, radius=1, it's small enough to just loop.
    # Each word has 1 + n*(q-1) neighbors.
    # 1 + 7*2 = 15 neighbors per word.
    # 2187 * 15 ~ 33k edges. Very small graph.
    
    for i in range(total_space):
        neighbors[i].append(i)
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                neighbors[i].append(i + diff * powers[pos])
                
    # Helper: Convert int to tuple
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Helper: Polish a solution (remove redundant, swap)
    def polish(selected):
        if not selected:
            return selected
            
        # Build counts
        counts = [0] * total_space
        for w in selected:
            for nb in neighbors[w]:
                counts[nb] += 1
                
        # Phase 1: Remove redundant
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                removable = True
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        removable = False
                        break
                if removable:
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True
            if not changed:
                break

        # Phase 2: Swaps
        max_swaps = 50
        swaps_made = 0
        while swaps_made < max_swaps and time.time() < deadline - 5:
            if not selected:
                break
            idx = random.randint(0, len(selected) - 1)
            w = selected[idx]
            uniques = [nb for nb in neighbors[w] if counts[nb] == 1]
            if not uniques:
                for nb in neighbors[w]:
                    counts[nb] -= 1
                selected.pop(idx)
                continue
                
            found = False
            best_candidate = None
            is_existing = False
            
            # Check existing
            for w2 in selected:
                if w2 == w:
                    continue
                ok = True
                for u in uniques:
                    if u not in neighbors[w2]:
                        ok = False
                        break
                if ok:
                    best_candidate = w2
                    is_existing = True
                    found = True
                    break
            
            if not found:
                # Scan all space
                for w2 in range(total_space):
                    if w2 == w:
                        continue
                    ok = True
                    for u in uniques:
                        if u not in neighbors[w2]:
                            ok = False
                            break
                    if ok:
                        best_candidate = w2
                        is_existing = False
                        found = True
                        break
            
            if found:
                for nb in neighbors[w]:
                    counts[nb] -= 1
                selected.pop(idx)
                if not is_existing:
                    selected.append(best_candidate)
                    for nb in neighbors[best_candidate]:
                        counts[nb] += 1
                swaps_made += 1
            else:
                pass
                
        return selected

    # Strategy:
    # 1. Generate a good initial solution using a fast heuristic (Greedy + Polish).
    # 2. Formulate the problem as an Integer Linear Program (ILP).
    #    Minimize sum(x_i)
    #    Subject to:
    #      sum(x_j for j in neighbors[i]) >= 1 for all i
    #      x_i in {0, 1}
    # 3. Use scipy.optimize.milp to solve it.
    # 4. If ILP finds a better solution, use it. Otherwise, use the heuristic.
    # 5. Polish the best solution found.

    # Step 1: Heuristic Initial Solution
    # Greedy Set Cover
    covered = [False] * total_space
    selected_greedy = []
    remaining = set(range(total_space))
    
    # To speed up greedy, we can precompute masks, but for 2187, simple list check is ok.
    # However, standard greedy iterates all 2187 words to find best gain.
    # 186 iterations * 2187 * 15 checks ~ 6M ops. Fast.
    
    while remaining and time.time() < deadline - 20:
        best_gain = -1
        best_word = -1
        # Optimization: Only consider words that cover at least one remaining node
        # And maybe restrict to a subset if it gets slow? No, 2187 is small.
        for w in remaining:
            gain = 0
            for nb in neighbors[w]:
                if nb in remaining:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
        if best_word == -1:
            break
        selected_greedy.append(best_word)
        for nb in neighbors[best_word]:
            if nb in remaining:
                remaining.remove(nb)
                
    # Polish the greedy solution
    selected_heuristic = polish(selected_greedy[:])
    
    best_solution = selected_heuristic[:]
    best_size = len(best_solution)
    
    # Step 2: ILP Formulation
    # If we have time, try to solve the ILP.
    # The ILP is exact. If HiGHS can solve it within the time limit, it's great.
    # 2187 binary variables, 2187 constraints.
    # This is a small ILP for modern solvers, but might take time.
    # We will try to solve it with a time limit.
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables: x_0 ... x_{N-1}
        # Objective: minimize sum(x)
        c = np.ones(total_space)
        
        # Constraints:
        # For each word i, sum(x_j for j in neighbors[i]) >= 1
        # This is a sparse matrix A of size 2187 x 2187.
        # A[i, j] = 1 if j is in neighbors[i]
        
        # Build A efficiently
        # Using scipy.sparse
        from scipy.sparse import csr_matrix
        
        rows = []
        cols = []
        data = []
        
        for i in range(total_space):
            for j in neighbors[i]:
                rows.append(i)
                cols.append(j)
                data.append(1)
                
        A = csr_matrix((data, (rows, cols)), shape=(total_space, total_space))
        
        # Constraint: A @ x >= 1
        # LinearConstraint(A, lb=np.ones(total_space), ub=np.inf)
        lb = np.ones(total_space)
        ub = np.full(total_space, np.inf)
        
        constraints = LinearConstraint(A, lb, ub)
        
        # Bounds: 0 <= x_i <= 1
        integrality = np.ones(total_space)
        bounds = Bounds(lb=0, ub=1)
        
        # Solve
        # We need to set a time limit. HiGHS in scipy doesn't directly expose a time limit in the simple API?
        # Actually, `milp` accepts `options`.
        # Let's check if we can pass a time limit.
        # If not, we just run it and hope it's fast, or catch timeout.
        # Given 2187 vars, it should be fast if the structure is good.
        # However, it might take longer than expected if the LP relaxation is weak.
        
        # Let's try. If it takes too long, we might not get a result.
        # To be safe, we only try ILP if we have > 30s left.
        if time.time() < deadline - 30:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={"time_limit": 30})
            
            if res.status == 0: # Optimal
                x = res.x
                # Round to integers
                x_int = (x > 0.5).astype(int)
                sol_ilp = [i for i in range(total_space) if x_int[i] == 1]
                
                # Verify validity
                if sol_ilp:
                    covered_check = [False] * total_space
                    for w in sol_ilp:
                        for nb in neighbors[w]:
                            covered_check[nb] = True
                    if all(covered_check):
                        if len(sol_ilp) < best_size:
                            best_solution = sol_ilp[:]
                            best_size = len(best_solution)
                            # Polish it
                            polished = polish(best_solution[:])
                            if len(polished) < best_size:
                                best_solution = polished[:]
                                best_size = len(polished)
                                
    except Exception as e:
        pass # Ignore ILP failures
        
    # Final Polish if time permits
    if time.time() < deadline - 5:
        final_polished = polish(best_solution[:])
        if len(final_polished) < best_size:
            best_solution = final_polished[:]
            best_size = len(final_polished)

    # Convert to list of tuples
    best_words = [int_to_tuple(v) for v in best_solution]
    return best_words
