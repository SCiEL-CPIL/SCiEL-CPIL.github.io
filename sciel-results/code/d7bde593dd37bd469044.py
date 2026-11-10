import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Pre-compute potential conflicts? No, dynamic is better for backtracking.
    # We will use a recursive backtracking search.
    
    best_sequence = []
    
    # 1. Get a quick baseline using the greedy method from the seed
    # This ensures we have a valid answer immediately.
    def greedy_search(candidates):
        values = []
        sums = set()
        for x in candidates:
            if values and x <= values[-1]:
                continue
            total = x
            if total in sums:
                continue
            valid = True
            new_sums = {total}
            for a in reversed(values):
                total += a
                if total in sums:
                    valid = False
                    break
                new_sums.add(total)
            if valid:
                values.append(x)
                sums.update(new_sums)
        return values

    # Deterministic greedy baseline
    candidates = list(range(1, 257))
    best_sequence = greedy_search(candidates)
    best_len = len(best_sequence)
    
    # 2. Backtracking Search
    # State: current sequence, set of all consecutive block sums
    # Heuristic: Try candidates that are "less constraining" or "more constrained"?
    # Usually, in exact cover/sudoku, we pick the variable with fewest options.
    # Here, adding a number creates new sums. We want to avoid collisions.
    
    # To make backtracking efficient, we need a good ordering of candidates.
    # A common heuristic for "distinct sums" problems is to prefer numbers that
    # create sums far from existing sums, or simply use a randomized order with backtracking.
    
    # We will use a depth-first search with randomized restarts and a time limit per branch.
    
    def backtrack(current_seq, current_sums, remaining_candidates, depth_limit):
        """
        current_seq: list of ints
        current_sums: set of all consecutive block sums
        remaining_candidates: set of available integers
        depth_limit: max length to try to achieve (if 0, try to maximize)
        """
        nonlocal best_sequence, best_len
        
        # Check time
        if time.time() > deadline:
            return
        
        # Pruning: If we can't beat the best, stop?
        # Max possible length is bounded by range and sum constraints.
        # Upper bound on length: If we pick k numbers from [1, 256], the max sum is ~256*k.
        # Number of blocks is k(k+1)/2.
        # Sums must be distinct.
        
        # Heuristic: Order remaining candidates.
        # To find long sequences, we might want to start small to fit more numbers?
        # Or start large to spread sums out?
        # Greedy sorted ascending usually works well for length.
        # Let's try a mixed strategy: Randomized backtracking.
        
        # If we have found a sequence of length L, we only care about extending if we can exceed L.
        # For now, let's just try to build a sequence.
        
        if not remaining_candidates:
            if len(current_seq) > best_len:
                best_sequence = current_seq[:]
                best_len = len(current_seq)
            return
        
        # Optimization: If the maximum possible length we can achieve (len(current_seq) + len(remaining_candidates)) 
        # is <= best_len, we can prune this branch.
        if len(current_seq) + len(remaining_candidates) <= best_len:
            return
            
        # Choose the next candidate.
        # Heuristic: Try candidates that are "compatible" with the current sums.
        # We will iterate through a shuffled list of remaining candidates.
        # To make it smarter, we can filter out candidates that are immediately invalid.
        
        candidates_list = list(remaining_candidates)
        random.shuffle(candidates_list)
        
        for x in candidates_list:
            if current_seq and x <= current_seq[-1]:
                continue
                
            # Check validity of adding x
            total = x
            if total in current_sums:
                continue
                
            new_sums = {total}
            valid = True
            # Check tails
            for a in reversed(current_seq):
                total += a
                if total in current_sums:
                    valid = False
                    break
                new_sums.add(total)
                
            if not valid:
                continue
            
            # Recursive step
            # We need to update remaining_candidates by removing x
            new_remaining = remaining_candidates - {x}
            
            # Before recursing, check if we are getting close to the deadline
            if time.time() > deadline:
                return
                
            backtrack(current_seq + [x], current_sums.union(new_sums), new_remaining, 0)
            
            # If we found a better solution, we might want to prune more aggressively
            # or just continue to see if we can find an even longer one.
            
            # Early termination if we hit the theoretical max? 
            # Theoretical max for [1, N] with distinct consecutive sums is not simply N.
            # But if we found a very long sequence, we can stop.
            if best_len >= 20: # Heuristic cap, likely we won't go much higher with N=256
                 # Actually, let's not hardcode 20, let's just run until time runs out.
                 pass
            
        # End of for loop
        
    # Initialize backtracking
    # We will run multiple restarts with different random seeds/orders
    # The "remaining_candidates" set is large (256 items). Backtracking depth could be high.
    # To avoid exponential explosion, we rely on the pruning: 
    # 1. Time limit
    # 2. Upper bound check (len(seq) + len(rem) <= best_len)
    
    # However, standard backtracking on 256 items might be too slow if not heavily pruned.
    # The "distinct sums" constraint is very strong.
    
    # Let's try a few restarts with different initial biases.
    # Bias 1: Ascending order preference
    # Bias 2: Random
    
    for _ in range(100): # Max restarts, but time limited
        if time.time() > deadline:
            break
            
        # Create a shuffled list of candidates
        # We will pass the full set to backtrack, but the internal shuffle handles ordering.
        # To make restarts different, we can change the global random state or just rely on the internal shuffle.
        
        # One improvement: Start with a small "seed" sequence from greedy to guide the backtracking?
        # Or just start from scratch.
        
        # To speed up, we can pre-filter candidates that are "bad" (e.g., small numbers create small sums which collide easily?).
        # Actually, small numbers are often good for packing.
        
        # Let's run the backtracker.
        # We need to be careful with Python recursion limits if depth is > 1000.
        # Max depth is 256. Default limit is 1000. So we are safe.
        
        # To make it faster, we can use a "conflict-directed" backjumping, but standard DFS is simpler.
        # Let's add a heuristic: Sort candidates in the backtrack function by a "cost" metric?
        # Cost = number of sums in current_sums that are close to x?
        # This might be too expensive to compute for every candidate at every step.
        
        # Simpler heuristic: Prefer candidates that are larger if we are stuck?
        # Or prefer candidates that are smaller?
        
        # Let's stick to randomized DFS with the upper-bound pruning.
        
        # To increase diversity, we can sometimes force the first few picks to be random.
        
        # Call backtrack
        # Note: passing a set of 256 ints and copying it at each step is expensive.
        # We can use a boolean array or just pass the set and modify it in place (backtracking style).
        
        # Let's implement an in-place backtracking for efficiency.
        
        def backtrack_inplace(current_seq, current_sums, available_set, index):
            nonlocal best_sequence, best_len
            
            if time.time() > deadline:
                return
            
            if index >= 256: # Max possible depth
                if len(current_seq) > best_len:
                    best_sequence = current_seq[:]
                    best_len = len(current_seq)
                return
                
            # Pruning
            if len(current_seq) + len(available_set) <= best_len:
                return
                
            # Get candidates
            # To avoid modifying the set while iterating, we create a list.
            # But creating a list of 256 items at every node is expensive.
            # Alternative: Iterate over the set directly? No, order matters for heuristics.
            # Let's just iterate over a shuffled list of the available set.
            
            # Optimization: If available_set is large, shuffling is O(N).
            # We can maintain a list of available items and swap-remove?
            # For simplicity and given the 50s limit, let's try the set approach first.
            
            candidates = list(available_set)
            random.shuffle(candidates)
            
            for x in candidates:
                if current_seq and x <= current_seq[-1]:
                    continue
                    
                total = x
                if total in current_sums:
                    continue
                    
                new_sums = set()
                valid = True
                for a in reversed(current_seq):
                    total += a
                    if total in current_sums:
                        valid = False
                        break
                    new_sums.add(total)
                    
                if not valid:
                    continue
                    
                # Add x
                current_seq.append(x)
                current_sums.update(new_sums)
                available_set.remove(x)
                
                # Recurse
                backtrack_inplace(current_seq, current_sums, available_set, index + 1)
                
                # Backtrack
                current_seq.pop()
                for s in new_sums:
                    current_sums.remove(s)
                available_set.add(x)
                
                # If we found a very good solution, we might want to stop early in this branch?
                # No, keep searching for longer.
                
        # Run one restart
        # We need to initialize the state.
        # We can start with an empty sequence or a greedy prefix.
        # Starting with a greedy prefix (e.g., first 5 elements) might help guide the search.
        # Let's try starting from scratch first.
        
        # To vary the search, we can bias the initial shuffle.
        # The internal shuffle in backtrack_inplace handles the rest.
        
        # One issue: `available_set.remove(x)` is O(1) for sets.
        # `current_sums.update` and `remove` are O(1) average.
        # The main cost is the `reversed(current_seq)` loop in validity check.
        # Length of current_seq grows up to ~20-30?
        # If length is 20, loop is 20.
        # Number of nodes visited?
        # If branching factor is high, this explodes.
        # The "distinct sums" constraint is very restrictive.
        # Most candidates will be invalid immediately.
        # So the effective branching factor is low.
        
        backtrack_inplace([], set(), set(range(1, 257)), 0)
        
        # If we found a solution of length > best_len, we update it.
        # The function updates best_sequence and best_len directly.
        
    return best_sequence
# EVOLVE-BLOCK-END
