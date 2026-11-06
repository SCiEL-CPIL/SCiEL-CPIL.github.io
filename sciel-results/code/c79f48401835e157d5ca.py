"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def construct():
    # Deadline guard: 150s wall-clock limit, aim to finish in < 120s
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_set = []
    best_size = 0
    best_counts = {}
    
    # We will run multiple randomized greedy attempts.
    # Each attempt uses a different random seed and a slightly different heuristic weight.
    
    num_attempts = 50  # Max attempts, but limited by time
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
            
        random.seed(attempt * 12345 + 7)
        
        # Heuristic parameters
        # We want to avoid sums that are already at count 2 (would become 3 -> invalid)
        # We also want to avoid sums that are at count 1 (would become 2 -> valid but less flexible)
        # Ideally, we want to add x such that x+y creates new sums or sums with count 0.
        
        chosen = []
        counts = {}
        
        # Candidate pool: all numbers 1..1000
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        # To make it more interesting, we can bias the shuffle or selection.
        # A simple "greedy with lookahead" style:
        # Instead of processing in fixed shuffled order, we can re-evaluate the "best" next move.
        # However, full re-evaluation is O(N^2) per step -> O(N^3) total. 1000^3 is too slow.
        
        # Alternative: Process candidates in a shuffled order, but allow "skipping" and "retrying"
        # or use a priority queue based on "potential".
        
        # Let's stick to a fast randomized greedy:
        # 1. Shuffle candidates.
        # 2. Iterate through them.
        # 3. If valid, add it.
        # This is the same as the seed but randomized order.
        # The seed used ASCENDING order. Randomizing order might help escape local minima of the ascending greedy.
        
        # To improve further, let's use a "soft" constraint or a scoring function.
        # Score of adding x:
        # Penalty for each sum s = x + y (y in chosen, y <= x) or 2x:
        # If counts[s] == 2: Invalid (Penalty = Infinity)
        # If counts[s] == 1: Penalty = 1 (uses up flexibility)
        # If counts[s] == 0: Penalty = 0 (new sum, good)
        
        # We want to minimize total penalty.
        # Since we must have penalty 0 (validity), we minimize the number of "tightened" sums.
        
        current_candidates = candidates[:]
        
        # We will iterate through the shuffled list.
        # To make it smarter, we can try to pick the "least restrictive" valid candidate at each step.
        # But that requires checking all remaining candidates at each step.
        # O(N^2) checks * N steps = O(N^3) = 10^9 ops. Too slow in Python.
        
        # Compromise: 
        # 1. Shuffle candidates.
        # 2. Iterate.
        # 3. If valid, add.
        # This is O(N^2) per attempt. 1000^2 = 10^6. 
        # 50 attempts * 10^6 = 50M ops. Should be fast in Python if optimized.
        
        for x in current_candidates:
            if time.time() > deadline:
                break
            
            # Check validity
            is_valid = True
            new_counts = {} # We don't need to copy the whole dict if we just check max
            
            # Sums involving x:
            # 2*x
            s = 2 * x
            if counts.get(s, 0) >= 2:
                is_valid = False
            else:
                # Count how many sums would go from 1->2
                # Actually, we just need to know if ANY sum goes > 2.
                # Since we check counts[s] >= 2, if it is 2, invalid.
                # If it is 1, it becomes 2. This is allowed.
                # So the check `counts.get(s, 0) >= 2` is sufficient for validity.
                
                # Wait, the task says "at most two representations".
                # So if count is 2, adding another makes it 3 -> Invalid.
                # If count is 1, adding another makes it 2 -> Valid.
                # If count is 0, adding another makes it 1 -> Valid.
                
                pass # Logic above holds.
            
            if not is_valid:
                continue
                
            # Additional check for all y in chosen
            # Optimization: If x is large, x+y is large.
            # We can break early if x + min(chosen) > max_sum_with_limit? No, sums can be anything.
            
            valid = True
            # We need to check all y in chosen
            for y in chosen:
                if y > x: 
                    # We only care about a <= b. 
                    # If y > x, then the pair is (x, y) which is same as (y, x) already counted when y was added?
                    # No, when y was added, we checked y + z for z in chosen_at_that_time.
                    # If x is added now, we check x + z for z in chosen.
                    # If z < x, pair is (z, x).
                    # If z > x, pair is (x, z).
                    # The set of sums is the same regardless of order.
                    # So we just check all y in chosen.
                    pass
                
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # It is valid. Add it.
            # Update counts
            counts[2 * x] = counts.get(2 * x, 0) + 1
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            
            chosen.append(x)
            
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = chosen[:]
                best_counts = counts.copy()
        
        # After one pass, we might have left valid numbers behind because they were invalid *at that time*
        # but might be valid now? No, adding numbers only increases counts, so if x was invalid before,
        # it remains invalid after adding more numbers.
        # So a single pass through the shuffled list is optimal for that specific order.
        
        # However, a different order yields a different set.
        # Randomizing the order is the key improvement over deterministic ascending.
        
    return best_set

# EVOLVE-BLOCK-END
