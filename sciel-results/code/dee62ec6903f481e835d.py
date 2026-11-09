import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def is_valid_set(elements):
        """Check if all nonempty subset sums are pairwise non-dividing and unique."""
        n = len(elements)
        if n == 0:
            return True
        # Generate all subset sums using bitmask
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False  # duplicate sum
            sums.add(s)
        # Check no two distinct sums divide each other
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            si = sorted_sums[i]
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % si == 0:
                    return False
        return True

    def is_valid_fast(elements):
        """Faster validity check using incremental subset sums."""
        n = len(elements)
        if n == 0:
            return True
        if n > 12:
            return is_valid_set(elements)
        # Generate all subset sums
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            si = sorted_sums[i]
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % si == 0:
                    return False
        return True

    def try_add(current_set, candidate):
        """Check if adding candidate to current_set keeps validity."""
        return is_valid_fast(current_set + [candidate])

    # Strategy 1: Superincreasing sequence search
    # A superincreasing sequence (each element > sum of all previous) guarantees distinct subset sums.
    # Then we only need to check divisibility.
    def build_superincreasing():
        """Build a superincreasing sequence from candidates, checking divisibility."""
        best = []
        # Try different starting points and greedy extensions
        for start_val in range(1, 257):
            if time.time() > deadline:
                break
            current = [start_val]
            running_sum = start_val
            for cand in range(start_val + 1, 257):
                if time.time() > deadline:
                    break
                if cand > running_sum:
                    # Superincreasing property holds, check divisibility
                    if is_valid_fast(current + [cand]):
                        current.append(cand)
                        running_sum += cand
            if len(current) > len(best):
                best = list(current)
        return best

    # Strategy 2: General greedy with backtracking
    def greedy_search():
        """Greedy search with multiple restarts."""
        best = []
        candidates = list(range(256, 0, -1))  # descending: larger numbers first
        
        # Multiple greedy passes with different orderings
        import random
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Shuffle candidates with bias toward larger values
            if attempt == 0:
                order = candidates[:]
            elif attempt == 1:
                order = sorted(candidates, key=lambda x: -x)
            else:
                order = candidates[:]
                random.shuffle(order)
                # Bias: put larger elements earlier
                order.sort(key=lambda x: -x * (0.5 + 0.5 * random.random()))
            
            current = []
            for cand in order:
                if time.time() > deadline:
                    break
                if try_add(current, cand):
                    current.append(cand)
            
            if len(current) > len(best):
                best = list(current)
        
        return best

    # Strategy 3: Local improvement - try to extend or swap
    def local_improve(initial_set):
        """Try to extend the set by adding elements, or swap to improve."""
        best = list(initial_set)
        candidates = list(range(256, 0, -1))
        
        # Try adding elements
        for cand in candidates:
            if time.time() > deadline:
                break
            if cand not in best:
                if try_add(best, cand):
                    best.append(cand)
        
        # Try single swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    if cand not in best:
                        new_set = best[:i] + best[i+1:] + [cand]
                        new_set.sort()
                        if try_add(new_set[:i], new_set[i]) and is_valid_fast(new_set):
                            if len(new_set) > len(best) or (len(new_set) == len(best) and sum(new_set) > sum(best)):
                                best = new_set
                                improved = True
                                break
                if improved:
                    break
        
        return best

    # Run strategies
    best_overall = []
    
    # Strategy 1: Superincreasing
    si_set = build_superincreasing()
    if len(si_set) > len(best_overall):
        best_overall = si_set
    
    # Strategy 2: General greedy
    g_set = greedy_search()
    if len(g_set) > len(best_overall):
        best_overall = g_set
    
    # Strategy 3: Local improvement on best so far
    if best_overall:
        li_set = local_improve(best_overall)
        if len(li_set) > len(best_overall):
            best_overall = li_set
    
    # Final: try to extend best_overall one more time
    if best_overall:
        final_set = list(best_overall)
        for cand in range(256, 0, -1):
            if time.time() > deadline:
                break
            if cand not in final_set:
                if try_add(final_set, cand):
                    final_set.append(cand)
        if len(final_set) > len(best_overall):
            best_overall = final_set
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_overall]
