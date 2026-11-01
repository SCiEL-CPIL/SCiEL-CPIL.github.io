import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for M (max entry)
    # We need total_diffs distinct differences in [1, M].
    # So M >= total_diffs.
    M_lower = total_diffs
    
    # Get an initial solution using greedy
    initial_rows = _greedy_construct(n_rows, k)
    best_rows = initial_rows
    best_M = _max_entry(initial_rows)
    
    # If greedy is already good enough or we have no time, return
    if time.time() > deadline - 10:
        return best_rows
        
    # Try to improve using Simulated Annealing on a "difference assignment" representation
    # Representation: A list of length total_diffs, where each element is the difference value.
    # However, differences must form valid rows. This is hard to represent as a simple permutation.
    
    # Alternative Representation:
    # Represent each row by its k non-zero elements.
    # State: List of n_rows lists, each of length k.
    # Objective: Minimize max(entry) subject to all differences distinct.
    
    # Let's use a different approach: 
    # We know the set of all differences used must be a subset of {1, ..., M}.
    # Let's fix M to be slightly above the lower bound and try to find a packing.
    # But SA works better if we can move between states.
    
    # Let's try SA on the rows themselves.
    # State: n_rows rows.
    # Move: Pick a random row, pick a random element, change it to a random value, 
    #        then re-sort and repair.
    # Problem: Repairing is hard.
    
    # Better SA approach:
    # Represent the solution as a set of differences.
    # Actually, let's stick to the row representation but use a smarter local search.
    
    # Let's try to improve the greedy solution by "shifting" rows or swapping elements.
    
    # Since the current program scores 1.16, it likely finds M around 100-110.
    # The optimal is 112. Wait, 112 is the BEST KNOWN. 
    # If the current program scores 1.16, and score is seed/value, and seed is 1.0...
    # If seed finds 112, value is 112. Score 1.0 means 112.
    # Score 1.16 means value = 112 / 1.16 ≈ 96.
    # This implies the current program finds a solution with max entry ~96?
    # But the task says "The best known set has largest entry 112". 
    # And "simple counting bound is 105".
    # If the current program finds 96, that would be a new record!
    # Let's re-read the score definition. "Score: the scope, the largest entry, minimized; it is an upper bound on m(n_rows, k)."
    # "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # If seed is the initial program. What does the initial program find?
    # The "initial program" usually finds a decent baseline.
    # If seed finds 120, and current finds 103, score is 120/103 = 1.16.
    # This makes sense. The counting bound is 105, so 103 is impossible?
    # "simple counting bound is 105". This usually means M >= 105.
    # So the best possible M is at least 105.
    # If current score is 1.16, and seed found, say, 120.
    # 120 / 1.16 = 103.4. This contradicts the bound 105.
    # Maybe the seed found 112? 112 / 1.16 = 96. Still < 105.
    # Maybe the "counting bound" is for a different parameter or I am misinterpreting.
    # Let's assume the lower bound is indeed around 105.
    # So the current program likely finds M around 105-110.
    
    # Let's try to find M closer to the lower bound.
    
    # Strategy: 
    # 1. Determine a tight lower bound LB.
    # 2. Try to solve for M = LB, LB+1, ... using a randomized backtracking / local search hybrid.
    
    # Let's implement a more robust randomized backtracking that can handle M close to LB.
    
    # The current _solve_packing uses random sampling of rows. This is very inefficient for tight M.
    # We need a more directed search.
    
    # Let's use a "Difference-Based" construction.
    # We need to assign n_rows * diffs_per_row distinct differences from {1, ..., M}.
    # Let's pick a random set of differences of size total_diffs from {1, ..., M}.
    # Then, try to partition them into n_rows sets of size diffs_per_row such that each set forms a valid row.
    # This is a difficult constraint satisfaction problem.
    
    # Alternative: 
    # Use the fact that rows are increasing sequences.
    # Let's try to construct rows one by one, but using a smarter heuristic.
    # Heuristic: When constructing a row, prefer differences that are "available" and "small" to keep the row compact?
    # Or prefer differences that are "large" to leave small differences for later?
    
    # Let's try a different structural change: 
    # **Iterative Deepening with Constraint Propagation**
    
    # For each M from LB to best_M - 1:
    #   Run a CP-SAT style search (backtracking with forward checking).
    
    # The current backtracking is weak. Let's strengthen it.
    
    def _solve_cp(n_rows, k, M, deadline):
        used_diffs = set()
        rows = []
        
        # Precompute all possible rows? No, too many.
        # But we can generate candidates on the fly, sorted by "compactness" or "randomness".
        
        # Let's use a recursive search that picks a row.
        # To break symmetry, we can fix the first row? No, rows are interchangeable.
        
        # Let's try to generate a pool of "good" candidates for each row.
        # A good candidate is one that uses small differences?
        
        # Actually, let's just improve the random restart strategy with a better row generator.
        # Instead of random.sample, let's use a weighted random selection that favors
        # configurations that are more likely to fit.
        
        # Or, let's use a **Genetic Algorithm** on the rows.
        
        # Let's stick to a robust randomized backtracking with a better candidate generation.
        
        def backtrack(idx):
            if time.time() > deadline - 2:
                return False
            if idx == n_rows:
                return True
                
            # Generate candidates for this row
            # We want to find a row disjoint from used_diffs.
            
            # Strategy: Try to construct a row that uses "rare" differences.
            # Or just try many random ones.
            
            # To speed up, let's try a limited number of random candidates.
            # If they fail, try a more systematic "greedy" construction for this row.
            
            # Greedy construction for a single row given used_diffs:
            # Pick smallest available x1, then smallest available x2, etc.
            
            # Let's try a mix.
            
            # 1. Try 50 random rows
            for _ in range(50):
                if time.time() > deadline - 2:
                    return False
                vals = random.sample(range(1, M + 1), k)
                vals.sort()
                row = [0] + vals
                d_set = set()
                valid = True
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in d_set or d in used_diffs:
                            valid = False
                            break
                        d_set.add(d)
                    if not valid:
                        break
                if valid:
                    rows.append(row)
                    used_diffs.update(d_set)
                    if backtrack(idx + 1):
                        return True
                    rows.pop()
                    used_diffs.difference_update(d_set)
                    
            # 2. If random fails, try a greedy row construction
            # Try to build a row using the smallest available differences
            # This is complex. Let's just do more random restarts.
            
            return False
            
        # Multiple restarts
        for restart in range(100):
            if time.time() > deadline - 2:
                break
            random.seed(restart * 1000 + M)
            used_diffs.clear()
            rows.clear()
            if backtrack(0):
                return rows
        return None

    # Let's calculate a better lower bound.
    # The counting bound is M(M+1)/2 >= total_diffs? No, differences are distinct integers.
    # We need total_diffs distinct integers in {1, ..., M}.
    # So M >= total_diffs.
    # For (7,5), total_diffs = 7 * 15 = 105.
    # So M >= 105.
    
    M_start = total_diffs
    
    # If M_start is much smaller than best_M, we have room to search.
    # But if best_M is close to M_start, we should focus on improving the current best.
    
    # Let's try M from M_start to best_M - 1.
    # To save time, we can skip some M if it's too slow.
    # But for small ranges, it's fine.
    
    # We can also parallelize? No, single process.
    
    # Let's try the CP solver for M in increasing order.
    # If we find a solution, update best and break.
    
    # To make it faster, we can pre-compute a list of "promising" rows for each M?
    # No, M changes.
    
    # Let's just run the CP solver.
    
    # Optimization: If M is close to total_diffs, the number of available differences is small.
    # The search space is constrained.
    
    # Let's try a few M values.
    # We can also try to improve the current best_rows using local search.
    
    # Let's implement a simple local search to reduce M of the current best_rows.
    # But M is determined by the max entry.
    # If we can find a different packing with smaller max entry, that's better.
    
    # The CP solver approach is the most direct way to find smaller M.
    
    # Let's ensure we don't spend too much time on a single M.
    # Time per M: (deadline - time.time()) / (best_M - M_start)
    
    # Let's just loop.
    
    # Note: The current program already does this but with a weaker solver.
    # My CP solver is also random-based. 
    # Is there a fundamentally different approach?
    
    # **Integer Programming**
    # We can model this as an ILP.
    # Variables: x_{r,i,j} binary, indicating if difference d is used in row r.
    # Constraints:
    # 1. Each row r uses exactly diffs_per_row differences.
    # 2. Each difference d is used at most once globally.
    # 3. The set of differences for row r must correspond to a valid increasing sequence of length k+1 starting at 0.
    #    This is the hard part. The "validity" of a set of differences is non-linear/complex.
    
    # However, for small k, we can pre-compute all valid difference sets for a given M?
    # No, M is up to 112. Number of valid rows is large.
    
    # Let's stick to the CP solver but make it smarter.
    # One idea: **Look-ahead**.
    # When choosing a row, check if the remaining differences can form the remaining rows.
    # This is expensive.
    
    # Let's try a **Greedy with Backtracking** that prioritizes rows that use "central" differences?
    
    # Actually, let's try to improve the random row generation.
    # Instead of uniform random, generate rows that are "spread out" or "compact".
    
    # Let's try a hybrid: 
    # 1. Generate a few random rows.
    # 2. Generate a few "greedy" rows (smallest available diffs).
    # 3. Generate a few "greedy" rows (largest available diffs).
    
    def _generate_candidate_row(used_diffs, M, k, strategy):
        # strategy: 'random', 'small', 'large'
        if strategy == 'random':
            vals = random.sample(range(1, M + 1), k)
            vals.sort()
            row = [0] + vals
        elif strategy == 'small':
            # Try to build a row using the smallest available differences
            # This is tricky because differences are pairwise.
            # Let's just pick small random values.
            vals = random.sample(range(1, min(M, 50) + 1), k)
            vals.sort()
            row = [0] + vals
        elif strategy == 'large':
            vals = random.sample(range(max(1, M - 50), M + 1), k)
            vals.sort()
            row = [0] + vals
        else:
            return None
            
        d_set = set()
        valid = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set or d in used_diffs:
                    valid = False
                    break
                d_set.add(d)
            if not valid:
                break
        if valid:
            return row, d_set
        return None, None

    def _solve_packing_v2(n_rows, k, M, deadline):
        used_diffs = set()
        rows = []
        
        def backtrack(idx):
            if time.time() > deadline - 2:
                return False
            if idx == n_rows:
                return True
                
            # Try different strategies
            strategies = ['random', 'small', 'large', 'random', 'random']
            random.shuffle(strategies)
            
            for strat in strategies:
                if time.time() > deadline - 2:
                    return False
                # Try a few candidates with this strategy
                for _ in range(10):
                    row, d_set = _generate_candidate_row(used_diffs, M, k, strat)
                    if row:
                        rows.append(row)
                        used_diffs.update(d_set)
                        if backtrack(idx + 1):
                            return True
                        rows.pop()
                        used_diffs.difference_update(d_set)
                        
            return False
            
        for restart in range(200):
            if time.time() > deadline - 2:
                break
            random.seed(restart * 777 + M)
            used_diffs.clear()
            rows.clear()
            if backtrack(0):
                return rows
        return None

    # Try M values
    # We can limit the number of M values we try if time is tight.
    # Let's try all from M_start to best_M - 1.
    
    for M in range(M_start, best_M):
        if time.time() > deadline - 5:
            break
        rows = _solve_packing_v2(n_rows, k, M, deadline)
        if rows is not None:
            best_rows = rows
            best_M = M
            break
            
    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True

