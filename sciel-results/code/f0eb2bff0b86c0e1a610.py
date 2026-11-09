import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85  # Safety margin before 90s kill

    # Fallback known valid set
    best_set = [31, 32, 34]
    best_size = 3

    def is_valid(S):
        if not S:
            return True
        sums = [0]
        for x in S:
            new_sums = [s + x for s in sums]
            sums.extend(new_sums)
        non_empty_sums = [s for s in sums if s != 0]
        if len(set(non_empty_sums)) != len(non_empty_sums):
            return False
        n = len(non_empty_sums)
        for i in range(n):
            u = non_empty_sums[i]
            for j in range(n):
                if i == j:
                    continue
                v = non_empty_sums[j]
                if v % u == 0 or u % v == 0:
                    return False
        return True

    def is_valid_incremental(S, x):
        if not S:
            return True
        sums_S = {0}
        for val in S:
            new_sums = set()
            for s in sums_S:
                new_sums.add(s + val)
            sums_S = sums_S | new_sums
        new_sums = {x + s for s in sums_S}
        if sums_S & new_sums:
            return False
        non_empty_new = [s for s in new_sums if s != 0]
        non_empty_old = [s for s in sums_S if s != 0]
        n = len(non_empty_new)
        for i in range(n):
            u = non_empty_new[i]
            for j in range(n):
                if i == j:
                    continue
                v = non_empty_new[j]
                if v % u == 0 or u % v == 0:
                    return False
        for u in non_empty_new:
            for v in non_empty_old:
                if v % u == 0 or u % v == 0:
                    return False
        return True

    # Precompute candidate pool: numbers 1..256
    # We will use a smart ordering: prioritize numbers that are "independent"
    # A good heuristic for avoiding divisibility is to pick numbers that are
    # close to each other but not dividing, or numbers with distinct prime factors.
    # However, the "binary" idea suggests picking numbers like 2^k + j.
    # Let's try a mix of strategies.

    # Strategy 1: Greedy with randomized ordering
    # Strategy 2: Local search (swap elements to allow adding new ones)

    # To speed up, we limit the size of S we try to extend incrementally.
    # If S is large, the check is expensive.
    
    max_restarts = 20000

    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break

        # Generate a random candidate order
        # We can bias towards larger numbers to avoid small divisors
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        S = []
        # Greedy build
        for x in candidates:
            if len(S) >= best_size + 2:
                # If we are significantly ahead, we can be more selective
                # But let's just keep trying
                pass

            if time.time() - start_time > time_limit - 1:
                break

            # Only try incremental check if S is small enough to be fast
            # If S is large, we might want to skip some candidates to save time
            # or just accept the cost.
            
            # Heuristic: if len(S) > 8, the check is O(2^k * k^2) which is heavy.
            # 2^8 = 256 sums. 256^2 = 65k ops. Fast.
            # 2^10 = 1024 sums. 1024^2 = 1M ops. ~0.05s.
            # 2^12 = 4096 sums. 4096^2 = 16M ops. ~0.5s.
            # 2^14 = 16384 sums. 16384^2 = 268M ops. ~5s. Too slow for many restarts.
            
            # So we should be careful with large S.
            # If len(S) > 10, we might want to break early in the restart
            # or only try a few candidates.
            
            if len(S) > 10:
                # We are in a promising region.
                # Try a few more, but if we fail, break to save time for other restarts?
                # Or continue but with a time check per step?
                # Let's just break if we've already beaten best_size significantly
                # or if we are taking too long.
                if len(S) > best_size:
                    # Update best
                    best_size = len(S)
                    best_set = S[:]
                # Continue trying a bit more to see if we can get even larger
                # but be strict on time
                if time.time() - start_time > time_limit - 5:
                    break
            
            if is_valid_incremental(S, x):
                S.append(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                
                # If we get very large, stop this restart to save time
                if len(S) > 12:
                    # Update best
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S[:]
                    break

        # After greedy build, try local search?
        # Local search: try to swap an element in S with an element not in S
        # to see if we can add another element afterwards.
        # This is complex. Let's stick to greedy for now but with many restarts.
        
        # Optimization: if we found a set of size > 3, we can try to extend it
        # by trying to add elements in a specific order (e.g., smallest first, largest first).
        # But for now, random restarts should cover this.

    # Final verification
    if best_size <= 12:
        if not is_valid(best_set):
            best_set = [31, 32, 34]
            best_size = 3
    else:
        # Trust incremental check for larger sets, but do a quick check if time permits
        if time.time() - start_time < time_limit - 2:
            if not is_valid(best_set):
                # Fallback to a smaller known valid set if the large one is invalid
                # This shouldn't happen if is_valid_incremental is correct
                best_set = [31, 32, 34]
                best_size = 3

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
