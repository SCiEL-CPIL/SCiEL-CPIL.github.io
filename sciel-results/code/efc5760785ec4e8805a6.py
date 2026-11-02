import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    # 150s limit, use 115s for safety to allow for final processing
    deadline = start_time + 115.0
    
    # Theoretical lower bound: n_rows * k * (k+1) / 2
    # For n_rows=4, k=7: 4 * 7 * 8 / 2 = 112
    lower_bound = n_rows * k * (k + 1) // 2
    
    # We will search for a solution with scope M, starting from lower_bound
    # and increasing M until a solution is found or time runs out.
    
    # To ensure we return *something* valid, we first run a fast greedy
    # to get a baseline solution. This guarantees we don't return [] or an
    # invalid solution if the exact search takes too long.
    
    baseline_solution, baseline_scope = greedy_solve(n_rows, k)
    
    if not baseline_solution:
        # Fallback to a very simple random construction if greedy fails (unlikely)
        # But greedy almost always succeeds for these parameters.
        return []

    best_solution = baseline_solution
    best_scope = baseline_scope
    
    # If the baseline is already at the lower bound, we can't do better.
    if best_scope <= lower_bound:
        return best_solution

    # Now, try to improve the scope by searching for solutions with
    # M < best_scope.
    # We iterate M from lower_bound to best_scope - 1.
    
    # Random seed for reproducibility within the run, but we want diversity
    # so we don't fix it globally, but we can use it for tie-breaking.
    
    for M in range(lower_bound, best_scope):
        if time.time() > deadline:
            break
            
        # Try to find a solution with max entry <= M
        # We use a randomized backtracking search with bitset optimization.
        
        # To avoid getting stuck in the same local minima for different M,
        # we can randomize the order of candidates slightly or use a 
        # deterministic order with pruning.
        
        sol = solve_dts(n_rows, k, M, deadline)
        
        if sol is not None:
            best_solution = sol
            best_scope = M
            # Since we are searching in increasing M, the first solution found
            # for a given M is valid. If we find one for M, we can stop 
            # searching for that M. We continue to M+1? No, we want the 
            # *smallest* M. So if we find a solution for M, that's the 
            # best possible for this iteration. We can return immediately 
            # because we are iterating from the lower bound up.
            # Wait, the loop is `for M in range(lower_bound, best_scope)`.
            # If we find a solution for M, it is the minimum possible 
            # greater than or equal to lower_bound. 
            # So we can return immediately.
            return sol

    return best_solution

def greedy_solve(n_rows, k):
    """
    Fast greedy construction to get a valid baseline solution.
    This is not optimized for minimal scope, just for validity.
    """
    used_diffs = 0  # Bitset of used differences
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        # We need k more numbers
        # Greedy: pick the smallest possible x that doesn't conflict
        for _ in range(k):
            x = row[-1] + 1
            while True:
                # Calculate conflicts with current row
                conflict_mask = 0
                for a in row:
                    diff = x - a
                    if diff <= 0:
                        continue
                    conflict_mask |= (1 << diff)
                
                if (used_diffs & conflict_mask) == 0:
                    # No conflict
                    row.append(x)
                    used_diffs |= conflict_mask
                    break
                x += 1
        rows.append(row)
        
    return rows, max(max(row) for row in rows)

def solve_dts(n_rows, k, max_scope, deadline):
    """
    Solves DTS with max entry <= max_scope.
    Returns a list of rows or None.
    """
    used_diffs = 0  # Bitset of used differences globally
    rows = []
    
    # We build rows one by one.
    # For each row, we build it element by element.
    
    # Pre-compute powers of 2 for speed? Python handles big ints well.
    
    def backtrack_row(row_idx, pos, current_last, row_so_far, row_mask):
        """
        Recursively builds the current row.
        row_idx: index of the row being built
        pos: how many elements are in row_so_far (including 0)
        current_last: the last element in row_so_far
        row_so_far: list of elements
        row_mask: bitset of differences used within this row (not yet added to global)
                   Actually, we should check against global + local.
                   It's better to maintain a single 'available' mask or 
                   check against (used_diffs | row_mask).
        """
        if time.time() > deadline:
            return None
        
        if pos == k + 1:
            return row_so_far
        
        # Determine the range for the next number x
        # x > current_last
        # x <= max_scope
        # Also, we must be able to fit the remaining (k+1 - pos) numbers.
        # The smallest possible next numbers are current_last + 1, current_last + 2, ...
        # So the smallest possible last number would be current_last + (k+1 - pos).
        # This must be <= max_scope.
        min_possible_last = current_last + (k + 1 - pos)
        if min_possible_last > max_scope:
            return None
        
        # Iterate candidates x from current_last + 1 to max_scope
        # We can prune if x is too large such that we can't fill the rest.
        # But the min_possible_last check handles the lower bound of x.
        # The upper bound for x is max_scope - (k - pos)
        # because we need k-pos more numbers after x.
        
        max_x = max_scope - (k - pos)
        
        # To add some randomness to avoid deterministic failure loops,
        # we can shuffle the candidate range slightly, but for DTS,
        # smaller x is generally better for packing.
        # However, sometimes a larger x early on allows better packing later.
        
        # Let's try a randomized approach: pick a random subset of candidates?
        # Or just iterate. Iteration is deterministic.
        # To escape local minima, we can randomize the start point or order.
        
        # Let's stick to increasing order for the first few, but allow 
        # "jumps" if we get stuck? 
        # Actually, for small M, the search tree is small. 
        # We will iterate x from current_last + 1 to max_x.
        
        for x in range(current_last + 1, max_x + 1):
            # Check conflicts with global used_diffs and current row_mask
            # Differences are x - a for all a in row_so_far
            conflict_mask = 0
            for a in row_so_far:
                d = x - a
                conflict_mask |= (1 << d)
            
            # Check if any of these differences are already used (global or local)
            if (used_diffs | row_mask) & conflict_mask:
                continue
            
            # If no conflict, recurse
            new_row_mask = row_mask | conflict_mask
            new_row = row_so_far + [x]
            
            res = backtrack_row(row_idx, pos + 1, x, new_row, new_row_mask)
            if res is not None:
                return res
        
        return None

    for i in range(n_rows):
        # Start row with [0]
        row = backtrack_row(i, 1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        # Add this row's differences to the global used_diffs
        for j in range(len(row)):
            for l in range(j+1, len(row)):
                d = row[l] - row[j]
                used_diffs |= (1 << d)
        
        # Check time
        if time.time() > deadline:
            return None

    return rows

# To improve the search, we can add a randomized restart mechanism.
# The above solve_dts is deterministic. If it fails for a specific M, it will always fail.
# We should modify solve_dts to accept a seed or randomize the candidate order.

def solve_dts_random(n_rows, k, max_scope, deadline, seed=0):
    """
    Solves DTS with max entry <= max_scope using randomized backtracking.
    """
    rng = random.Random(seed)
    used_diffs = 0
    rows = []
    
    def backtrack_row(row_idx, pos, current_last, row_so_far, row_mask):
        if time.time() > deadline:
            return None
        
        if pos == k + 1:
            return row_so_far
        
        min_possible_last = current_last + (k + 1 - pos)
        if min_possible_last > max_scope:
            return None
        
        max_x = max_scope - (k - pos)
        
        # Generate valid candidates
        candidates = []
        for x in range(current_last + 1, max_x + 1):
            conflict_mask = 0
            for a in row_so_far:
                d = x - a
                conflict_mask |= (1 << d)
            
            if ((used_diffs | row_mask) & conflict_mask) == 0:
                candidates.append(x)
        
        if not candidates:
            return None
        
        # Randomize the order of candidates to explore different branches
        rng.shuffle(candidates)
        
        # Try each candidate
        for x in candidates:
            conflict_mask = 0
            for a in row_so_far:
                d = x - a
                conflict_mask |= (1 << d)
            
            new_row_mask = row_mask | conflict_mask
            new_row = row_so_far + [x]
            
            res = backtrack_row(row_idx, pos + 1, x, new_row, new_row_mask)
            if res is not None:
                return res
        
        return None

    for i in range(n_rows):
        row = backtrack_row(i, 1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        for j in range(len(row)):
            for l in range(j+1, len(row)):
                d = row[l] - row[j]
                used_diffs |= (1 << d)
        
        if time.time() > deadline:
            return None

    return rows

# Override the main construct to use the randomized solver with multiple attempts
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    lower_bound = n_rows * k * (k + 1) // 2
    
    baseline_solution, baseline_scope = greedy_solve(n_rows, k)
    if not baseline_solution:
        return []
        
    best_solution = baseline_solution
    best_scope = baseline_scope
    
    if best_scope <= lower_bound:
        return best_solution

    # Try to find a better solution
    # We iterate M from lower_bound to best_scope - 1
    for M in range(lower_bound, best_scope):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random seeds to increase the chance of finding a solution
        # for this M if the deterministic one failed (though we use random solver)
        # The random solver might fail due to bad luck. We try a few times.
        attempts = 0
        max_attempts = 5
        
        while not found and attempts < max_attempts and time.time() < deadline:
            sol = solve_dts_random(n_rows, k, M, deadline, seed=random.randint(0, 1000000))
            if sol is not None:
                best_solution = sol
                best_scope = M
                found = True
                # Since we are looking for the minimum M, we can return immediately
                return sol
            attempts += 1
        
        # If we didn't find a solution for M, continue to M+1

    return best_solution
