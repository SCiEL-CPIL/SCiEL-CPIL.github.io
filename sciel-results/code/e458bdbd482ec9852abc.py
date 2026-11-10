import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    N = 256
    
    def count_3aps(A):
        """Count distinct 3-term arithmetic progressions in A."""
        A_set = set(A)
        count = 0
        # A 3-AP is (a, a+d, a+2d) with d > 0
        # We can iterate over middle term b, and check b-d, b+d
        for b in A:
            # d can range such that b-d >= 1 and b+d <= N
            # max d is min(b-1, N-b)
            max_d = min(b - 1, N - b)
            for d in range(1, max_d + 1):
                if (b - d) in A_set and (b + d) in A_set:
                    count += 1
        return count

    def is_4ap_free_with_x(A_set, x):
        """Check if adding x to A_set creates a 4-AP."""
        # A 4-AP is (y, y+d, y+2d, y+3d)
        # x can be at position 0, 1, 2, or 3.
        
        # Case 1: x is the 1st term (index 0).
        # Need x+d, x+2d, x+3d in A_set.
        # x+3d <= N => d <= (N-x)//3
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in A_set and (x + 2*d) in A_set and (x + 3*d) in A_set:
                return False
        
        # Case 2: x is the 2nd term (index 1).
        # Need x-d, x+d, x+2d in A_set.
        # x-d >= 1 => d <= x-1
        # x+2d <= N => d <= (N-x)//2
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in A_set and (x + d) in A_set and (x + 2*d) in A_set:
                return False
        
        # Case 3: x is the 3rd term (index 2).
        # Need x-2d, x-d, x+d in A_set.
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= N => d <= N-x
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2*d) in A_set and (x - d) in A_set and (x + d) in A_set:
                return False
        
        # Case 4: x is the 4th term (index 3).
        # Need x-3d, x-2d, x-d in A_set.
        # x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d) in A_set and (x - 2*d) in A_set and (x - d) in A_set:
                return False
        
        return True

    def generate_initial_set():
        """Generate a 4-AP-free set using randomized greedy."""
        A = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_4ap_free_with_x(A, x):
                A.add(x)
        return A

    # Initial construction
    best_A = generate_initial_set()
    best_score = count_3aps(best_A)
    
    # Local Search
    # We will try to improve the set by adding, removing, or swapping elements.
    # To speed up, we can maintain a list of "blocked" elements or just re-check.
    # Given N=256, re-checking is fast enough.
    
    current_A = set(best_A)
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 1000 # Stop if stuck for a while
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding elements
        # Iterate through all x not in current_A
        for x in range(1, N + 1):
            if x not in current_A:
                if is_4ap_free_with_x(current_A, x):
                    # Check if adding x increases 3-AP count
                    # We can calculate delta in 3-APs
                    # New 3-APs involving x:
                    # x as first: (x, x+d, x+2d)
                    # x as second: (x-d, x, x+d)
                    # x as third: (x-2d, x-d, x)
                    
                    delta = 0
                    # x as 1st
                    max_d = (N - x) // 2
                    for d in range(1, max_d + 1):
                        if (x + d) in current_A and (x + 2*d) in current_A:
                            delta += 1
                    # x as 2nd
                    max_d = min(x - 1, N - x)
                    for d in range(1, max_d + 1):
                        if (x - d) in current_A and (x + d) in current_A:
                            delta += 1
                    # x as 3rd
                    max_d = (x - 1) // 2
                    for d in range(1, max_d + 1):
                        if (x - 2*d) in current_A and (x - d) in current_A:
                            delta += 1
                    
                    if delta > 0:
                        current_A.add(x)
                        current_score += delta
                        improved = True
                        # Continue trying to add more? Or restart?
                        # Restarting the loop allows adding multiple elements if possible.
                        break 
                    elif delta == 0:
                        # Adding doesn't help score, but might enable future moves?
                        # For now, skip to keep set minimal and dense.
                        pass
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_A = set(current_A)
            no_improve_count = 0
            continue
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

        # 2. Try swapping elements
        # Remove a, add b
        # This is O(N^2) per iteration, which is 256^2 = 65536 checks.
        # Each check is O(N) for 4-AP. Total ~16M ops. Might be too slow in Python if done naively in a tight loop.
        # Optimization: Only try swaps if we are stuck.
        
        # Heuristic: Pick a random element to remove, and find the best element to add.
        if current_A:
            a = random.choice(list(current_A))
            # Try removing a
            temp_A = set(current_A)
            temp_A.remove(a)
            
            # Find best b to add to temp_A
            best_b = None
            best_delta = -1000
            best_new_score = -1
            
            for b in range(1, N + 1):
                if b not in temp_A:
                    if is_4ap_free_with_x(temp_A, b):
                        # Calculate score of temp_A U {b}
                        # Score = score(temp_A) + delta(b, temp_A)
                        # We need score(temp_A).
                        # Let's just calculate full score if candidate is good?
                        # Or estimate delta.
                        
                        # Delta of adding b to temp_A
                        delta = 0
                        # b as 1st
                        max_d = (N - b) // 2
                        for d in range(1, max_d + 1):
                            if (b + d) in temp_A and (b + 2*d) in temp_A:
                                delta += 1
                        # b as 2nd
                        max_d = min(b - 1, N - b)
                        for d in range(1, max_d + 1):
                            if (b - d) in temp_A and (b + d) in temp_A:
                                delta += 1
                        # b as 3rd
                        max_d = (b - 1) // 2
                        for d in range(1, max_d + 1):
                            if (b - 2*d) in temp_A and (b - d) in temp_A:
                                delta += 1
                        
                        # We also lose 3-APs involving a that are no longer formed?
                        # No, we are comparing (current_A) vs (temp_A U {b}).
                        # current_score is known.
                        # new_score = count_3aps(temp_A) + delta
                        # count_3aps(temp_A) = current_score - (3-APs involving a)
                        
                        # Let's calculate 3-APs involving a in current_A
                        delta_a = 0
                        # a as 1st
                        max_d = (N - a) // 2
                        for d in range(1, max_d + 1):
                            if (a + d) in current_A and (a + 2*d) in current_A:
                                delta_a += 1
                        # a as 2nd
                        max_d = min(a - 1, N - a)
                        for d in range(1, max_d + 1):
                            if (a - d) in current_A and (a + d) in current_A:
                                delta_a += 1
                        # a as 3rd
                        max_d = (a - 1) // 2
                        for d in range(1, max_d + 1):
                            if (a - 2*d) in current_A and (a - d) in current_A:
                                delta_a += 1
                        
                        # Note: delta_a counts 3-APs in current_A that contain a.
                        # Are there 3-APs in temp_A U {b} that contain both a and b? No, a is removed.
                        # Are there 3-APs in current_A that contain a and b?
                        # If so, they are counted in delta_a (if b is in current_A? No, b is not in current_A).
                        # Wait, b is not in current_A. So no 3-AP in current_A contains b.
                        # So, 3-APs in current_A containing a are exactly the ones lost.
                        # 3-APs in temp_A U {b} containing b are exactly delta.
                        # 3-APs in temp_A U {b} not containing b are exactly 3-APs in temp_A not containing a.
                        # 3-APs in temp_A = 3-APs in current_A - 3-APs in current_A containing a.
                        # So, new_score = (current_score - delta_a) + delta.
                        
                        new_score = current_score - delta_a + delta
                        
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_delta = new_score - current_score
                            best_b = b
            
            if best_b is not None and best_new_score > current_score:
                current_A = temp_A
                current_A.add(best_b)
                current_score = best_new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(current_A)
                no_improve_count = 0

    return sorted(best_A)

# Keep the function name and signature
# The task requires construct() -> a nonempty subset of [1,n=256]
# The code above implements this.
