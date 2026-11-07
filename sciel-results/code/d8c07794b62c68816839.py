import time
import random
import sys

def construct():
        start_time = time.time()
        deadline = start_time + 50.0
        
        MAX_VAL = 256
        MAX_SUM = MAX_VAL * 30 # Upper bound for sums
        
        def validate(elements):
            """
            Returns True if the set of elements is valid.
            1. Distinct integers in 1..256.
            2. Unique non-empty subset sums.
            3. No two distinct subset sums divide each other.
            """
            if not elements:
                return True
            
            # Check distinct and range
            if len(set(elements)) != len(elements):
                return False
            if any(x < 1 or x > MAX_VAL for x in elements):
                return False
            
            k = len(elements)
            # Generate subset sums
            # Using bit operations or iterative doubling
            sums = set()
            # Optimization: if k is large, this is expensive.
            # But we only call this in local search where k is around 10-20.
            # 2^20 is 1M, might be slow in Python if done many times.
            # We can prune: if we just want to check validity, we can stop early?
            # No, we need all sums for divisibility.
            
            # Efficient sum generation
            current_sums = {0}
            for x in elements:
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + x)
                current_sums.update(new_sums)
            
            # Remove 0
            current_sums.discard(0)
            
            # Check uniqueness
            if len(current_sums) != (1 << k) - 1:
                return False
            
            # Check divisibility
            # Max sum in current_sums
            max_s = max(current_sums)
            
            # Boolean array for fast lookup
            # Size needs to be at least max_s
            # But we check multiples up to max_s
            is_sum = [False] * (max_s + 1)
            for s in current_sums:
                is_sum[s] = True
            
            # Check if any s divides another
            # For each s in current_sums:
            #   Check if 2s, 3s, ... are in is_sum
            for s in current_sums:
                if s == 0: continue
                # Check multiples
                m = 2
                while s * m <= max_s:
                    if is_sum[s * m]:
                        return False
                    m += 1
            
            return True

        def greedy_build(randomize=False):
            current_set = []
            # Sort candidates?
            # To maximize size, we should pick small numbers?
            # But small numbers create small sums which divide large sums easily.
            # So maybe pick numbers that are "incompatible" with each other.
            
            # Strategy: Try to add elements one by one.
            # Candidates are 1..256.
            # We enforce superincreasing?
            # If we enforce superincreasing, we don't need to check uniqueness, only divisibility.
            # And sum generation is still needed for divisibility.
            
            # Let's try standard greedy without superincreasing constraint, but with validation.
            # This is slow.
            
            # Let's stick to Superincreasing for the initial build to ensure speed,
            # then use local search to break superincreasing if it helps?
            # Actually, superincreasing is a strong constraint.
            # Let's try to build a superincreasing set first.
            
            sum_so_far = 0
            for _ in range(30):
                added = False
                # Range for next element: sum_so_far + 1 to 256
                start = sum_so_far + 1
                if start > 256:
                    break
                
                # Get candidates
                candidates = list(range(start, 257))
                if randomize:
                    random.shuffle(candidates)
                
                for x in candidates:
                    new_set = current_set + [x]
                    if validate(new_set):
                        current_set = new_set
                        sum_so_far += x
                        added = True
                        break
            
            return current_set

        def local_search(initial_set, max_steps=1000):
            current_set = list(initial_set)
            best_set = list(current_set)
            
            # Pre-compute sums for current set?
            # Just call validate.
            
            # Moves:
            # 1. Add a random element
            # 2. Remove a random element
            # 3. Swap a random element with a random other number
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Try to add
                if len(current_set) < 30:
                    # Pick random candidate from 1..256 not in set
                    candidates = [x for x in range(1, 257) if x not in current_set]
                    if candidates:
                        x = random.choice(candidates)
                        new_set = current_set + [x]
                        if validate(new_set):
                            current_set = new_set
                            if len(current_set) > len(best_set):
                                best_set = list(current_set)
                            continue # Restart loop to try adding more? Or just continue
        
                # Try to swap
                if len(current_set) > 0:
                    # Pick index to remove
                    idx = random.randint(0, len(current_set) - 1)
                    old_x = current_set[idx]
                    
                    # Pick new x
                    # Avoid picking old_x
                    candidates = [x for x in range(1, 257) if x != old_x and x not in current_set]
                    if candidates:
                        new_x = random.choice(candidates)
                        new_set = current_set.copy()
                        new_set[idx] = new_x
                        if validate(new_set):
                            current_set = new_set
                            # Size didn't change, but maybe this allows future additions
                            # We don't update best_set here as size is same,
                            # unless we consider "potential" which is hard.
                            # But if we can add later, we will.
                            continue

                # Try to remove (to escape local minima?)
                if len(current_set) > 1:
                    idx = random.randint(0, len(current_set) - 1)
                    new_set = current_set.copy()
                    new_set.pop(idx)
                    if validate(new_set):
                        current_set = new_set
                        continue

            return best_set

        best_found = []
        
        # Run multiple random greedy starts
        for i in range(20):
            if time.time() > deadline:
                break
            
            # Greedy build
            # Use different random seeds or strategies
            # Strategy 1: Greedy smallest
            # Strategy 2: Greedy largest
            # Strategy 3: Randomized Greedy
            
            # Let's just do randomized greedy
            s = greedy_build(randomize=True)
            
            if len(s) > len(best_found):
                best_found = s
                
            # Local search on this result
            if time.time() > deadline:
                break
            
            improved = local_search(s, max_steps=500)
            if len(improved) > len(best_found):
                best_found = improved
            
            # Also try to grow best_found directly?
            # local_search already tries to add.
            
        return list(best_found)

    # Test locally? No, just output.
    # Ensure types are correct.
    # return list of ints.
