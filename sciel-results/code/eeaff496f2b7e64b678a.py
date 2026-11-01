import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    # 150s limit. 5s margin.
    deadline = start_time + 145.0

    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on M: need total_diffs_needed distinct differences in {1..M}
    M_lower = total_diffs_needed

    # Initial greedy solution to have a valid fallback
    greedy_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(greedy_rows)
    best_rows = greedy_rows

    # Search for a better M starting from the theoretical lower bound
    # We search upwards from M_lower. If we find a solution for M, that's likely optimal or near-optimal.
    # We stop if we exceed the greedy solution's M (since greedy is already a valid upper bound).
    
    # Optimization: We can also try to improve the greedy solution directly if M_lower search fails
    # or takes too long, but the set packing approach is generally more powerful for finding tight bounds.
    
    # We will try M values from M_lower up to best_M - 1.
    # If time runs out, we return best_rows.
    
    # To make the search efficient, we need to generate candidate rows for a given M.
    # A candidate row is a subset of {0, ..., M} of size k+1 containing 0,
    # such that its internal differences are distinct.
    
    # We will iterate M from M_lower to best_M - 1.
    # For each M, we generate a pool of valid rows.
    # Then we solve the set packing problem.
    
    # To avoid generating too many rows, we can limit the pool size or use a heuristic selection.
    # However, for small k (like 5) and moderate M (like 105-112), the number of valid rows is manageable.
    # Number of ways to choose k elements from 1..M is C(M, k).
    # For M=112, k=5, C(112, 5) ~ 50 million. This is too many to generate and store explicitly if we do it naively.
    
    # Alternative: Generate rows on the fly during backtracking?
    # Or, generate a random sample of valid rows?
    # Or, use a smarter generation:
    # We know that differences must be distinct.
    # Let's stick to a randomized restart backtracking that generates random valid rows.
    # This avoids the huge pre-computation.
    
    # Strategy:
    # For each M in range(M_lower, best_M):
    #   Try to solve using randomized backtracking with constraint propagation.
    #   If solved, update best.
    
    # The backtracking:
    # State: used_diffs (set), current_rows (list)
    # Goal: fill n_rows
    # Move: pick a random valid row from available diffs, add to current_rows.
    # Heuristic: 
    #   1. Order rows by "difficulty"? Hard to define.
    #   2. Forward checking: if adding a row leaves no possible rows for the next, backtrack.
    #      Checking "no possible rows" is expensive.
    #      Approximation: if the number of available diffs drops below diffs_per_row, fail.
    
    # Let's implement a randomized backtracking with restarts.
    # We will try multiple M values.
    
    # To improve success rate, we can use a "pool" of pre-generated rows for a specific M,
    # but limit the pool size to avoid memory/time explosion.
    # Actually, generating C(M, k) rows is the bottleneck.
    # Let's try generating rows lazily in the backtracking.
    
    def solve_for_M(M, time_budget):
        """
        Try to find n_rows disjoint valid rows with max entry <= M.
        Returns list of rows or None.
        """
        if M < k:
            return None
        
        # Random restarts
        max_restarts = 1000
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                return None
            if time.time() > start_time + time_budget:
                return None
            
            random.seed(restart * 7919 + M)
            
            used_diffs = set()
            rows = []
            
            # Backtracking search
            # We use a simple recursive backtracking with randomization.
            # To make it faster, we can pre-generate a few candidate rows at each step?
            # No, generating a valid row is O(k^2) check.
            
            # Let's use a helper to find a valid row given used_diffs and M.
            # This helper tries random samples until it finds one.
            
            def find_row(used_diffs_local, M_local):
                # Try up to N random samples
                for _ in range(500):
                    # Pick k distinct values from 1..M_local
                    vals = random.sample(range(1, M_local + 1), k)
                    vals.sort()
                    row = [0] + vals
                    
                    d_set = set()
                    valid = True
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in d_set:
                                valid = False
                                break
                            if d in used_diffs_local:
                                valid = False
                                break
                            d_set.add(d)
                    
                    if valid:
                        return row, d_set
                return None, None
            
            def backtrack(idx):
                if time.time() > deadline - 2:
                    return False
                if idx == n_rows:
                    return True
                
                # To improve efficiency, we can shuffle the order in which we try to find rows?
                # No, rows are indistinguishable in terms of constraints, only the set of used diffs matters.
                
                # Find a row
                row, d_set = find_row(used_diffs, M)
                if row is None:
                    return False
                
                # Forward checking: ensure we haven't used too many diffs
                # Total diffs available is M. We need diffs_per_row more.
                # if len(used_diffs) + d_set + (n_rows - idx - 1) * diffs_per_row > M:
                #     return False # This is a necessary condition, but not sufficient.
                # Actually, the condition is simply that there must be enough *unused* diffs.
                # But they must form valid rows. This check is weak.
                
                rows.append(row)
                used_diffs.update(d_set)
                
                if backtrack(idx + 1):
                    return True
                
                rows.pop()
                used_diffs.difference_update(d_set)
                
                # Optional: Try a few more random rows before backtracking further?
                # The current find_row already tries 500 samples.
                # If it fails, it returns None, and we backtrack.
                # This might be too aggressive. If find_row fails, it means we couldn't find *any* valid row in 500 tries.
                # This is likely a dead end.
                
                return False
            
            if backtrack(0):
                return rows
        
        return None

    # Allocate time for each M
    # We want to try M_lower first, then M_lower+1, etc.
    # We give more time to smaller M as they are more constrained and more valuable.
    # But if M_lower is impossible, we waste time.
    # Let's give a fixed budget per M, say 10-20 seconds, and iterate.
    
    # Since we don't know how many M values to try, let's just loop until time runs out.
    # We will try M in increasing order.
    
    # To make the search smarter, we can use a "pool" of pre-computed rows for the current M.
    # But generating the pool is expensive.
    # Let's stick to the on-the-fly generation but increase the sample count in find_row if we have time?
    # No, keep it simple.
    
    # One optimization: if we fail for M, the used_diffs state might be informative, but with random restarts it's discarded.
    
    # Let's try M from M_lower to best_M - 1.
    # We will break if we find a solution.
    
    # To avoid spending too much time on an impossible M, we limit the restarts per M.
    # Let's say 50 restarts per M, or 10 seconds per M, whichever comes first.
    
    for M_try in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
        
        # Budget for this M: min(10s, remaining_time)
        # Actually, let's just run solve_for_M with a time budget.
        # If we find a solution, we return it immediately as it's better than current best.
        
        # To make it more robust, let's try a few different random seeds or strategies.
        # The current solve_for_M uses random restarts.
        
        # Let's add a heuristic: if M is close to M_lower, the packing is tight.
        # We can try to construct rows that "spread out" their differences?
        # Or rows that are "dense"?
        # Random sampling usually works okay.
        
        res = solve_for_M(M_try, time_budget=10.0)
        if res is not None:
            best_M = M_try
            best_rows = res
            break # Found a solution for this M, which is the best possible for this M.
            # Since we are iterating M upwards, this is the minimal M found so far.
            # We could stop here if we want the absolute best, but maybe we can find an even smaller one?
            # No, we started from M_lower. So if we found a solution for M_try, it is the minimal M.
            # So we can return immediately.
            # Wait, is it possible that M_lower is not the true lower bound?
            # M_lower = total_diffs_needed. This is a counting bound. It is a valid lower bound.
            # So if we find a solution for M_lower, it is optimal.
            # If we find a solution for M_lower+1, it is the next best.
            # So yes, break.

    return best_rows

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
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
    """
    Check if rows are valid.
    """
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
