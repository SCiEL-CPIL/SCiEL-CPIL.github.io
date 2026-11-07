import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    best_set = [31, 32, 34]  # fallback
    
    def get_subset_sums(elements):
        """Compute all nonempty subset sums as a set."""
        sums = set()
        for e in elements:
            new_sums = set()
            for s in sums:
                ns = s + e
                if ns not in sums and ns not in new_sums:
                    new_sums.add(ns)
            if e not in sums and e not in new_sums:
                new_sums.add(e)
            sums = sums | new_sums
        return sums
    
    def check_divisibility(sums):
        """Check if any two distinct sums divide each other."""
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                if b % a == 0:
                    return False
        return True
    
    def try_add(current_elements, current_sums, candidate):
        """Try adding candidate to current_elements. Returns new_sums if valid, else None."""
        new_sums = set(current_sums)
        # Generate new sums by adding candidate to each existing sum
        for s in list(current_sums):
            ns = s + candidate
            if ns in new_sums:
                return None  # collision
            new_sums.add(ns)
        if candidate in new_sums:
            return None  # collision
        new_sums.add(candidate)
        
        # Check divisibility: for each pair, one divides the other
        s_list = sorted(new_sums)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                if b % a == 0:
                    return None
        return new_sums
    
    # Backtracking search
    def backtrack(elements, sums, candidates, depth):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(elements) > len(best_set):
            best_set = list(elements)
        
        if not candidates:
            return
        
        for i, cand in enumerate(candidates):
            new_sums = try_add(elements, sums, cand)
            if new_sums is not None:
                backtrack(elements + [cand], new_sums, candidates[i+1:], depth + 1)
                if time.time() > deadline:
                    return
    
    # Try different starting points and orderings
    # Use numbers from a range that's likely to work
    # Strategy: try to build a superincreasing-like sequence
    
    # Candidate pool: try various ranges
    for start_val in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 30, 32, 36, 40, 48, 56, 64]:
        if time.time() > deadline:
            break
        # Generate candidates: multiples or specific patterns
        candidates = list(range(start_val, 257))
        # Shuffle to try different orderings
        random.seed(start_val)
        random.shuffle(candidates)
        # But also try sorted
        for ordering in [candidates, sorted(candidates)]:
            if time.time() > deadline:
                break
            backtrack([], set(), ordering, 0)
            if time.time() > deadline:
                break
    
    # Also try a more targeted approach: superincreasing sequences
    for base in range(2, 20):
        if time.time() > deadline:
            break
        elements = []
        sums = set()
        total = 0
        for val in range(base, 257):
            if time.time() > deadline:
                break
            new_sums = try_add(elements, sums, val)
            if new_sums is not None:
                elements.append(val)
                sums = new_sums
                total += val
                if len(elements) > len(best_set):
                    best_set = list(elements)
    
    # Ensure best_set is valid and return as list of ints
    result = list(best_set)
    # Verify all are int
    result = [int(x) for x in result]
    return result
