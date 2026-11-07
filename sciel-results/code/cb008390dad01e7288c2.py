import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = [31, 32, 34]
    MAX_SUM = 8000

    def greedy_search(candidates):
        current_set = []
        current_sums = set()
        forbidden = set()
        
        # Pre-compute multiples for efficiency? No, dynamic.
        
        for x in candidates:
            # Generate new sums
            new_sums = set()
            new_sums.add(x)
            for s in current_sums:
                new_sums.add(s + x)
            
            # Check collisions with old sums
            # If any s in new_sums is in current_sums, invalid
            # Actually, if s_new == s_old, then s_new - x == s_old - x? No.
            # s_new = s_old + x. If s_new == s_old2 (old), then s_old + x = s_old2.
            # This is a collision.
            if new_sums & current_sums:
                continue

            # Check if any new sum is in forbidden
            if any(s in forbidden for s in new_sums):
                continue

            # Check divisibility among new sums
            # Sort new sums to break early?
            sorted_new = sorted(new_sums)
            valid_pair = True
            n = len(sorted_new)
            for i in range(n):
                si = sorted_new[i]
                # si cannot divide sj for j > i
                for j in range(i + 1, n):
                    sj = sorted_new[j]
                    if sj % si == 0:
                        valid_pair = False
                        break
                if not valid_pair:
                    break
            
            if not valid_pair:
                continue

            # Valid! Update state
            current_set.append(x)
            current_sums.update(new_sums)
            
            # Update forbidden
            for s in new_sums:
                # Add multiples of s
                for k in range(2, MAX_SUM // s + 1):
                    forbidden.add(s * k)
                    
        return current_set

    # Try largest first
    candidates = list(range(256, 0, -1))
    s1 = greedy_search(candidates)
    if len(s1) > len(best_set):
        best_set = s1

    # Randomized search
    while time.time() < deadline:
        random.shuffle(candidates)
        s = greedy_search(candidates)
        if len(s) > len(best_set):
            best_set = s

    return best_set
