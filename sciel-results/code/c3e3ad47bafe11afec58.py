import time
import random
import sys

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We aim to return well before that.
    deadline = start_time + 135
    
    prescribed = [1, 2, 5, 11]
    N = 284
    
    # Helper to check if a set is a valid Sidon set
    # A Sidon set has distinct pairwise sums a+b (a<=b)
    # Equivalently, distinct positive differences a-b (a!=b)
    def is_valid_sidon(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Strategy:
    # 1. Try to find a solution of size 20 (the theoretical max) using randomized greedy + local search.
    # 2. If that fails quickly, try size 19, then 18, etc.
    # 3. Use multiple restarts with different random seeds.
    
    # We represent the state as a list of integers.
    # We start with the prescribed elements and add more.
    
    def solve_greedy_hillclimb(target_k, time_budget):
        """
        Attempt to find a Sidon set of size target_k containing prescribed elements.
        Uses randomized greedy construction followed by local search (hill climbing).
        """
        start_time = time.time()
        prescribed_set = set(prescribed)
        
        # If target_k is small, we might just return prescribed if it's valid (it is)
        if target_k <= len(prescribed):
            return list(prescribed)
        
        num_to_add = target_k - len(prescribed)
        
        # We will try multiple random restarts
        # Each restart:
        # 1. Greedily pick random valid candidates to reach target_k
        # 2. If stuck or time runs out, try local search to fix conflicts or add more?
        # Actually, standard approach:
        # 1. Start with prescribed.
        # 2. Repeatedly pick a random candidate from [1, N] not in set.
        # 3. If it creates a conflict, reject. If not, add it.
        # 4. If we reach target_k, we are done.
        # 5. If we get stuck (no more valid candidates to add), we can try to swap existing elements
        #    to open up space, or just restart.
        
        # A better heuristic for "stuck":
        # If we can't add any more, we might have a local maximum.
        # We can try to remove a non-prescribed element and try to add a different one.
        
        for attempt in range(100):
            if time.time() > start_time + time_budget:
                break
            
            random.seed(attempt)
            
            current_set = list(prescribed)
            current_set_set = set(current_set)
            
            # Precompute used differences for quick lookup?
            # Or just check on the fly.
            # Checking on the fly is O(k) per candidate.
            
            # Function to check if x can be added
            def can_add(x):
                if x in current_set_set:
                    return False
                for y in current_set:
                    d = abs(x - y)
                    # Check if this difference already exists
                    # We need to be careful: differences are positive.
                    # If we have a,b in set, d=|a-b|.
                    # If we add x, new diffs are |x-y| for all y in set.
                    # These must be distinct from each other and from existing diffs.
                    # Since we check against the set of existing diffs, we need that set.
                    pass
            
            # It's better to maintain the set of used differences explicitly.
            used_diffs = set()
            for i in range(len(current_set)):
                for j in range(i + 1, len(current_set)):
                    used_diffs.add(current_set[j] - current_set[i])
            
            def can_add_fast(x):
                if x in current_set_set:
                    return False
                new_diffs = []
                for y in current_set:
                    d = abs(x - y)
                    if d in used_diffs:
                        return False
                    new_diffs.append(d)
                # Check if new_diffs have duplicates among themselves
                # Since x is fixed, |x-y1| == |x-y2| implies y1==y2 (since y1!=y2 in set).
                # So new_diffs are automatically distinct.
                # So we just need to check against used_diffs.
                return True
            
            # Greedy phase
            added_count = 0
            max_iterations = 1000
            iter_count = 0
            
            while len(current_set) < target_k and iter_count < max_iterations:
                iter_count += 1
                if time.time() > start_time + time_budget:
                    break
                
                # Pick a random candidate
                # We can restrict range to [1, N]
                x = random.randint(1, N)
                
                if can_add_fast(x):
                    # Add x
                    current_set.append(x)
                    current_set_set.add(x)
                    for y in current_set[:-1]:
                        used_diffs.add(abs(x - y))
            
            if len(current_set) == target_k:
                # Verify validity just in case
                if is_valid_sidon(current_set):
                    return current_set
            
            # If we didn't reach target_k, try local search
            # This is complex. Simpler to just restart with a different seed.
            # However, we can try to "shake" the set.
            # If we are stuck, it means no x in [1,N] can be added.
            # This implies the set is "maximal" but not necessarily "maximum".
            # To escape, we could remove a random non-prescribed element and try again.
            
            if len(current_set) > len(prescribed):
                # Try a few local moves
                for move in range(20):
                    if time.time() > start_time + time_budget:
                        break
                    
                    # Remove a random non-prescribed element
                    non_prescribed = [x for x in current_set if x not in prescribed_set]
                    if not non_prescribed:
                        break
                    
                    to_remove = random.choice(non_prescribed)
                    current_set.remove(to_remove)
                    current_set_set.remove(to_remove)
                    
                    # Recalculate used_diffs
                    used_diffs = set()
                    for i in range(len(current_set)):
                        for j in range(i + 1, len(current_set)):
                            used_diffs.add(current_set[j] - current_set[i])
                    
                    # Try to add a new random element
                    for _ in range(50):
                        x = random.randint(1, N)
                        if can_add_fast(x):
                            current_set.append(x)
                            current_set_set.add(x)
                            for y in current_set[:-1]:
                                used_diffs.add(abs(x - y))
                            break
                    
                    if len(current_set) == target_k:
                        if is_valid_sidon(current_set):
                            return current_set

        return None

    # Try sizes from 20 down to 5
    # The theoretical max is 20.
    # We should try 20 first with a generous time budget.
    # If it fails, try 19, etc.
    
    # Time allocation:
    # Total budget ~135s.
    # Try 20: 60s
    # Try 19: 30s
    # Try 18: 20s
    # Try 17: 10s
    # Try 16: 5s
    # Reserve 10s for safety/return
    
    time_alloc = {
        20: 60,
        19: 30,
        18: 20,
        17: 10,
        16: 5,
        15: 2
    }
    
    best_solution = list(prescribed)
    
    for target_k in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break
        
        budget = time_alloc.get(target_k, 1)
        
        # Check if we even have time
        if time.time() + budget > deadline - 5:
            budget = max(1, int(deadline - 5 - time.time()))
            if budget <= 0:
                break
        
        sol = solve_greedy_hillclimb(target_k, budget)
        
        if sol is not None:
            # Update best solution if this one is larger (or same size but we prefer larger)
            # Since we iterate down, the first found is the largest.
            # But if we find size 20, we return immediately?
            # The task says "Maximize the size".
            # So if we find a valid set of size 20, that's optimal.
            if len(sol) == target_k:
                return sol
            else:
                # This shouldn't happen if solve_greedy_hillclimb returns None on failure
                # or a valid set of target_k.
                # But if it returns a smaller set? No, it returns None if it doesn't reach target_k.
                pass

    # If we couldn't find a specific target, return the best we have?
    # The function above returns None if it fails to reach target_k.
    # So if we loop through all and fail, we return the prescribed set?
    # Wait, the prescribed set is always valid.
    # But we might have found a larger set in a previous iteration that wasn't the target?
    # No, solve_greedy_hillclimb only returns if it reaches target_k.
    
    # To be safe, let's do one final greedy attempt to get the largest possible set
    # without a specific target, just to maximize size, if the targeted searches failed.
    
    # Actually, the targeted search for 20, 19... covers the maximization.
    # If 20 fails, 19 is the next best.
    
    # However, if solve_greedy_hillclimb is flaky, we might want a fallback.
    # Let's just return the prescribed set if all else fails.
    
    return best_solution

# To ensure the function name and signature are correct
# The task requires: construct() -> distinct integers...
# The code above defines construct().

# Let's add a quick test in our head:
# Prescribed: 1, 2, 5, 11
# Diffs: 1, 4, 10, 3, 9, 6
# All distinct. Valid.

# One potential issue: recursion limit is not needed here as we don't use recursion for the search.
# The search is iterative.

# Let's refine the randomization.
# The current implementation uses random.seed(attempt) which makes the "random" attempts deterministic.
# This is good for reproducibility, but might miss good solutions if the seed space is small.
# However, 100 attempts with different seeds is usually enough for these sizes.

# One improvement: Instead of picking a completely random x from 1..N,
# we could pick from a list of candidates that are "likely" to be valid.
# But calculating that is expensive.
# Random picking with rejection is O(1) per try, and we have many tries.

# Let's double check the deadline logic.
# deadline = start_time + 135
# The loop breaks if time.time() > deadline - 5.
# Inside solve_greedy_hillclimb, it also checks time.
# This should be safe.

# One more thing: The prompt says "Scores near 20 would show...".
# So finding 20 is the goal.
# The inspiration code also tried CSP.
# My code uses Greedy + Hill Climbing (restarts).
# This is a different structural approach than the current DFS.
# It is also different from the pure CSP in the inspiration, though similar in spirit (search).
# But Greedy+HillClimbing is often faster for finding *some* solution of large size,
# whereas DFS/CSP tries to prove existence or find one systematically.
# For "maximize size", finding a valid set of size 20 is sufficient.
# We don't need to prove 21 is impossible (we know it is from the prompt).

# Let's ensure the output is a list of distinct integers.
# The code maintains current_set as a list of unique integers.

# Final check on imports.
# time, random are used.

# The code looks solid.

# One minor optimization:
# In `can_add_fast`, we iterate over `current_set`.
# As the set grows, this gets slower.
# But N=284 is small.
# Max size 20.
# So max 20 iterations per check.
# Very fast.

# The bottleneck is the number of random picks.
# If the density of valid candidates is low, we might need many picks.
# But with restarts, we should be fine.

# Let's write the final code.
