import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def check_move_add(S, x):
        """Check if x can be added to S without violating sum-free property."""
        # Violation if:
        # 1. x = a + b for some a, b in S
        # 2. x + a = b for some a, b in S
        for a in S:
            if x - a in S:
                return False
            if x + a in S:
                return False
        return True

    def solve_instance(vals):
        n = len(vals)
        if n == 0:
            return []
        
        # Initial Solution: Odds
        S = set(v for v in vals if v % 2 != 0)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any element
            for v in vals:
                if v not in S:
                    if check_move_add(S, v):
                        S.add(v)
                        improved = True
                        break # Restart loop to check again with new S
            
            if not improved:
                # 2. Try 2-swaps: Remove 1, Add 2
                # This allows escaping local optima where no single element can be added.
                found_swap = False
                # Limit attempts to prevent timeout
                attempts = 0
                max_attempts = 100
                
                while not found_swap and attempts < max_attempts:
                    if time.time() > deadline - 5:
                        break
                    attempts += 1
                    
                    if not S:
                        break
                    
                    # Pick a random element to remove
                    rem = random.choice(list(S))
                    temp_S = S - {rem}
                    
                    # Find candidates to add
                    # We want to add 2 elements that were previously blocked
                    # Just pick 2 random candidates from vals not in temp_S
                    candidates = [v for v in vals if v not in temp_S]
                    if len(candidates) < 2:
                        continue
                    
                    add1, add2 = random.sample(candidates, 2)
                    new_S = temp_S | {add1, add2}
                    
                    # Check if new_S is sum-free
                    # Efficient check: Only check violations involving add1 or add2
                    # But since we removed 'rem', some old violations might be resolved, 
                    # but we only added add1, add2. 
                    # We must ensure the entire new_S is sum-free.
                    # Given small N, full check is safe.
                    
                    valid = True
                    # Check all pairs in new_S
                    # Optimization: Only check if (a+b) is in new_S
                    for a in new_S:
                        for b in new_S:
                            if a + b in new_S:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        S = new_S
                        improved = True
                        found_swap = True
                        break # Restart main loop

        return list(S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 10:
            # Fallback to odds if running out of time
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        
        # Run solver
        best_S = solve_instance(vals)
        
        # Optional: Try a second heuristic if time permits
        # E.g., start with numbers in top half
        if time.time() < deadline - 20:
            # Create a different initial set
            # Sort by value, take top half
            sorted_vals = sorted(vals)
            mid = len(sorted_vals) // 2
            init_S = set(sorted_vals[mid:])
            
            # Run a simplified local search from this start
            # We can reuse solve_instance but it starts with odds.
            # Let's just trust the odds+swap for now to keep code simple and fast.
            pass
            
        results.append(best_S)
        
    return results
