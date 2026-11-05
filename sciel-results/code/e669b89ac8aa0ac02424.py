import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    N = 1000
    max_sum = 2 * N
    
    # Data structures
    # A: list of elements
    # in_A: set of elements for O(1) lookup
    # sum_counts: list of counts for sums 0..2N
    
    best_set = []
    best_size = 0
    
    def is_valid_add(A, in_A, sum_counts, x):
        # Check if x can be added to A
        # Sums involved: x+y for y in A, and 2x
        # We need sum_counts[x+y] < 2 for all y in A
        # And sum_counts[2x] < 2
        
        # Optimization: If 2x > 2N, it's out of bounds? No, x<=N, 2x<=2N.
        if sum_counts[2*x] >= 2:
            return False
        
        for y in A:
            s = x + y
            if sum_counts[s] >= 2:
                return False
        return True

    def add_element(A, in_A, sum_counts, x):
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]: # A[:-1] is the old A
            s = x + y
            sum_counts[s] += 1
        # Note: A is modified in place. 
        # Wait, A[:-1] creates a copy. 
        # Better to store old_A or iterate carefully.
        # Let's refactor to be safe.
        
    # Refactored Add/Remove/Swap to be safer with in-place modifications
    
    def try_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        # Valid
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]: # The elements that were in A before adding x
            sum_counts[x+y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x) # O(N)
        sum_counts[2*x] -= 1
        for y in A: # A is now the set without x
            sum_counts[x+y] -= 1
        return True

    def try_swap(A, in_A, sum_counts, y, x):
        # Swap y in A with x not in A
        if x in in_A or y not in in_A:
            return False
            
        # Check validity of new set A' = (A \ {y}) + {x}
        # Sums affected:
        # 1. Sums involving y are removed.
        # 2. Sums involving x are added.
        
        # We need to check if any sum involving x would exceed 2.
        # The counts for sums involving x in the NEW set are:
        # new_count(s) = old_count(s) - (contributions from y if y involved in s) + (contributions from x)
        
        # It is easier to just check the condition directly on the hypothetical state.
        # For any z in A' (which is A without y), we need new_count(x+z) <= 2.
        # new_count(x+z) = current_count(x+z) - (1 if y involved in x+z else 0) + 1
        # Wait, current_count(x+z) includes pairs from A.
        # If y is in A, and y+z' = x+z, then y is involved.
        # Since z is in A, and y is in A.
        # The pair (y, z) sums to y+z.
        # If y+z == x+z, then y=x, which is false.
        # So y is NOT involved in the sum x+z via the pair (y,z).
        # Is y involved in x+z via another pair?
        # x+z = a+b. If y is a or b.
        # If y=a, then b = x+z-y.
        # So we need to check if there exists b in A such that y+b = x+z.
        # i.e., b = x+z-y.
        # If b is in A and b != y (since y is removed), then y contributed to the count of x+z.
        
        # This is getting complex to calculate on the fly.
        # Given |A| is small, we can just simulate the counts for the sums involving x.
        
        # Let's calculate the "delta" counts for sums involving x.
        # For each z in A:
        #   If z == y: skip (y is removed)
        #   Else:
        #     s = x + z
        #     We need to know the count of s in the NEW set.
        #     new_count(s) = old_count(s) - (count of pairs in A summing to s that include y) + (count of pairs in A' summing to s that include x)
        #     
        #     Pairs in A summing to s including y:
        #     y + w = s => w = s - y = x + z - y.
        #     If w is in A and w != y and w != z? 
        #     Actually, if w is in A, then (y, w) is a pair in A summing to s.
        #     So we subtract 1 if (s-y) in A and (s-y) != y.
        #     Note: if s-y == y, then 2y=s.
        #     
        #     Pairs in A' summing to s including x:
        #     x + z = s. This is the pair we are adding. So we add 1.
        #     Are there other pairs in A' summing to s including x?
        #     x + w' = s => w' = s - x = z.
        #     So only the pair (x, z) involves x and sums to s.
        #     
        #     So:
        #     new_count(s) = old_count(s) - (1 if (s-y) in in_A and (s-y) != y else 0) + 1
        #     
        #     We need new_count(s) <= 2.
        #     => old_count(s) + 1 - (1 if (s-y) in in_A and (s-y) != y else 0) <= 2
        #     => old_count(s) - (1 if (s-y) in in_A and (s-y) != y else 0) <= 1
        #     => old_count(s) <= 1 + (1 if (s-y) in in_A and (s-y) != y else 0)
        #
        #     Also need to check 2x.
        #     new_count(2x) = old_count(2x) - (1 if (2x-y) in in_A and (2x-y) != y else 0) + 1
        #     Need new_count(2x) <= 2.
        #
        #     AND, we must ensure that removing y doesn't make any OTHER sum invalid?
        #     No, removing y only decreases counts. So if it was valid before, it remains valid for sums not involving x.
        #     Only sums involving x are "new" or "changed".
        #     Sums involving x were 0 before (since x not in A).
        #     So we only need to check sums x+z for z in A\{y} and 2x.
        
        # Let's implement this check.
        
        # Check 2x
        s = 2 * x
        contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
            
        # Check x+z for z in A, z != y
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
                
        # If we get here, swap is valid.
        # Perform swap
        remove_element(A, in_A, sum_counts, y)
        # Now add x
        # Note: remove_element modified A and sum_counts.
        # Now x is not in A.
        # We can just call try_add? 
        # try_add checks validity again. It's redundant but safe.
        # Or we can manually update.
        # Since we already validated, we can update directly.
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1
        return True

    # Initialize with a good greedy set
    # Greedy: try to add 1, 2, 3...
    A = []
    in_A = set()
    sum_counts = [0] * (2 * N + 1)
    
    for x in range(1, N + 1):
        if try_add(A, in_A, sum_counts, x):
            continue
    
    best_set = A[:]
    best_size = len(A)
    
    # Local Search
    # We will try to improve the set.
    # Loop until deadline.
    
    # Track temperature for simulated annealing
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        # Try all x not in A
        # To speed up, we can shuffle candidates
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        random.shuffle(candidates)
        
        for x in candidates:
            if try_add(A, in_A, sum_counts, x):
                added = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break # Restart search with new state
        
        if added:
            continue
            
        # 2. If no add, try a swap with simulated annealing acceptance
        swapped = False
        # Pick random y in A, x not in A
        # Try multiple random swaps with SA acceptance
        for _ in range(200): # Limit attempts per iteration to avoid getting stuck in one bad spot
            if not A or not candidates: break
            y = random.choice(A)
            x = random.choice(candidates)
            
            # Check if swap is valid
            if x in in_A or y not in in_A:
                continue
                
            # Check validity of new set A' = (A \ {y}) + {x}
            valid = True
            
            # Check 2x
            s = 2 * x
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                valid = False
            
            if valid:
                # Check x+z for z in A, z != y
                for z in A:
                    if z == y:
                        continue
                    s = x + z
                    contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
                    if sum_counts[s] - contrib_y + 1 > 2:
                        valid = False
                        break
            
            if valid:
                # Accept the swap (always accept valid swaps that maintain or improve size)
                # For SA, we could also accept some invalid swaps with probability, 
                # but since we want to maximize size, let's stick to valid swaps
                # and use SA for the "quality" of the swap in terms of future potential
                # For now, accept all valid swaps
                remove_element(A, in_A, sum_counts, y)
                in_A.add(x)
                A.append(x)
                sum_counts[2*x] += 1
                for z in A[:-1]:
                    sum_counts[x+z] += 1
                swapped = True
                break
        
        if swapped:
            # Update temperature
            temperature *= cooling_rate
            continue
            
        # 3. If no swap, try to remove an element to break deadlock
        # Use a smarter removal: remove element that is "most constrained"
        # i.e., participates in the most sums that are at capacity
        if A:
            # Score each element by how many sums it contributes to that are at count 2
            removal_scores = {}
            for y in A:
                score = 0
                # Count sums involving y that are at capacity
                for z in A:
                    s = y + z
                    if sum_counts[s] >= 2:
                        score += 1
                # Also count 2y
                if sum_counts[2*y] >= 2:
                    score += 1
                removal_scores[y] = score
            
            # Remove element with highest score (most constrained)
            # With some randomness to escape local optima
            max_score = max(removal_scores.values())
            # Select from elements with score close to max
            candidates_to_remove = [y for y in A if removal_scores[y] >= max_score - 1]
            if not candidates_to_remove:
                candidates_to_remove = A
            y = random.choice(candidates_to_remove)
            remove_element(A, in_A, sum_counts, y)
            # Reset temperature when we remove
            temperature = 1.0
            # After removal, we might be able to add more.
            # Loop back to top to try adding.
            
    return best_set
