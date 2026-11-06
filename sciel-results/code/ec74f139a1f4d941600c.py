import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Seed set
    A = set(range(64, 129))
    
    # counts[s] = number of unordered pairs {a,b} in A such that a+b=s
    counts = [0] * 257
    
    # Helper to compute counts from scratch
    def compute_counts(A_set):
        c = [0] * 257
        for a in A_set:
            for b in A_set:
                if b > a:
                    s = a + b
                    if 128 <= s <= 256:
                        c[s] += 1
            s = 2 * a
            if 128 <= s <= 256:
                c[s] += 1
        return c

    counts = compute_counts(A)
    
    def get_max_mult(c):
        return max(c[128:257])
        
    def get_sum_squares(c):
        return sum(x*x for x in c[128:257])
        
    def is_valid(c):
        return min(c[128:257]) > 0

    curr_max = get_max_mult(counts)
    curr_ss = get_sum_squares(counts)
    
    best_max = curr_max
    best_ss = curr_ss
    best_A = set(A)
    
    # Local Search
    # We will alternate between trying to reduce Max, then reducing Sum of Squares.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Removals
        # Sort elements by their contribution to the current peak
        # Contribution of a: number of sums s in [128,256] where counts[s] == curr_max and s-a in A (or s/2==a)
        
        # For simplicity, iterate all
        for a in list(A):
            if time.time() > deadline:
                break
            
            # Check if removing a is valid
            # A sum s becomes 0 if counts[s] == 1 and a is part of the unique pair for s
            critical = False
            for b in A:
                if b < a:
                    s = a + b
                    if 128 <= s <= 256 and counts[s] == 1:
                        critical = True
                        break
            if not critical:
                s = 2 * a
                if 128 <= s <= 256 and counts[s] == 1:
                    critical = True
            
            if critical:
                continue
            
            # Simulate removal
            # Update counts
            for b in A:
                if b < a:
                    s = a + b
                    if 128 <= s <= 256:
                        counts[s] -= 1
            s = 2 * a
            if 128 <= s <= 256:
                counts[s] -= 1
            
            A.remove(a)
            
            new_max = get_max_mult(counts)
            new_ss = get_sum_squares(counts)
            
            if new_max < curr_max:
                curr_max = new_max
                curr_ss = new_ss
                improved = True
                if curr_max < best_max:
                    best_max = curr_max
                    best_ss = curr_ss
                    best_A = set(A)
            elif new_max == curr_max and new_ss < curr_ss:
                curr_ss = new_ss
                improved = True
                # Update best if max is same but ss is better?
                # The primary goal is min max.
                # If max is same, better ss is good for future steps.
                if curr_max < best_max or (curr_max == best_max and curr_ss < best_ss):
                    best_max = curr_max
                    best_ss = curr_ss
                    best_A = set(A)
            else:
                # Rollback
                A.add(a)
                for b in A:
                    if b < a:
                        s = a + b
                        if 128 <= s <= 256:
                            counts[s] += 1
                s = 2 * a
                if 128 <= s <= 256:
                    counts[s] += 1
        
        if improved:
            continue
        
        # 2. Try Swaps if no removal worked
        # Pick a random a to remove
        if A:
            a = random.choice(list(A))
            
            # Check if a can be removed (ignoring the add for a moment, just check if it creates a hole)
            # If it creates a hole, the swap must fix it by adding b that covers that hole.
            # This is complex. 
            # Simpler: Just try random swaps and check validity of the resulting set.
            
            candidates = [x for x in range(129) if x not in A]
            if candidates:
                b = random.choice(candidates)
                
                # Simulate Swap: Remove a, Add b
                # 1. Remove a
                for c in A:
                    if c < a:
                        s = a + c
                        if 128 <= s <= 256:
                            counts[s] -= 1
                s = 2 * a
                if 128 <= s <= 256:
                    counts[s] -= 1
                A.remove(a)
                
                # 2. Add b
                for c in A:
                    if c < b:
                        s = b + c
                        if 128 <= s <= 256:
                            counts[s] += 1
                s = 2 * b
                if 128 <= s <= 256:
                    counts[s] += 1
                A.add(b)
                
                # Check Validity
                if is_valid(counts):
                    new_max = get_max_mult(counts)
                    new_ss = get_sum_squares(counts)
                    
                    if new_max < curr_max:
                        curr_max = new_max
                        curr_ss = new_ss
                        improved = True
                        if curr_max < best_max:
                            best_max = curr_max
                            best_ss = curr_ss
                            best_A = set(A)
                    elif new_max == curr_max and new_ss < curr_ss:
                        curr_ss = new_ss
                        improved = True
                        if curr_max < best_max or (curr_max == best_max and curr_ss < best_ss):
                            best_max = curr_max
                            best_ss = curr_ss
                            best_A = set(A)
                    else:
                        # Rollback
                        # Remove b
                        for c in A:
                            if c < b:
                                s = b + c
                                if 128 <= s <= 256:
                                    counts[s] -= 1
                        s = 2 * b
                        if 128 <= s <= 256:
                            counts[s] -= 1
                        A.remove(b)
                        
                        # Add a
                        for c in A:
                            if c < a:
                                s = a + c
                                if 128 <= s <= 256:
                                    counts[s] += 1
                        s = 2 * a
                        if 128 <= s <= 256:
                            counts[s] += 1
                        A.add(a)
        
        # If no improvement for a while, we might be stuck.
        # We can add a "cooling" phase or random restart, but given the small space,
        # simple hill climbing with swaps should suffice.
        
    return list(best_A)
