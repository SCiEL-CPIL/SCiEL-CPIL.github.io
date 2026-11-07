import time
import random

def is_valid(s):
    """
    Check if the set s satisfies the cardinality-separated subset sum constraint.
    """
    k = len(s)
    # sums_by_card[c] will store the set of all possible sums of subsets of size c
    sums_by_card = [set() for _ in range(k + 1)]
    sums_by_card[0].add(0)
    
    # Efficiently compute subset sums by cardinality
    # We can use DP or just iterate combinations for small k
    # k is at most 10-12, so combinations are feasible for checking
    # But for checking validity, we just need to see if any sum appears in two different cardinality sets.
    
    # Generate all subset sums
    # For k=10, 2^10 = 1024 subsets. Very fast.
    # For k=20, 2^20 = 1M subsets. Still fast in Python if optimized.
    
    for i in range(k + 1):
        # Get combinations of size i
        # Using itertools.combinations
        from itertools import combinations
        for comb in combinations(s, i):
            sm = sum(comb)
            sums_by_card[i].add(sm)
            
    # Check for overlaps between different cardinalities
    seen_sums = {}
    for i in range(k + 1):
        for sm in sums_by_card[i]:
            if sm in seen_sums:
                return False
            seen_sums[sm] = i
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s buffer
    
    # Base valid set: Superincreasing powers of 2 up to 256
    # 1, 2, 4, 8, 16, 32, 64, 128, 256
    base_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_set = base_set[:]
    
    if not is_valid(best_set):
        # Fallback, though it should be valid
        return [1, 2, 4, 8, 16, 32, 64, 128]
        
    # Try to find a larger set (size 10)
    # Heuristic: Try to replace some large elements with slightly smaller ones to free up space for a 10th?
    # Or just try random sets of size 10?
    # Random search for size 10 in [1, 256] is a huge space.
    # However, valid sets are likely structured.
    
    # Let's try a local search starting from the base set.
    # Try to add a 10th element by modifying the set.
    
    # Since finding a size 10 set is hard, and size 9 is guaranteed,
    # we will attempt a limited search for size 10.
    
    # Strategy: 
    # 1. Try to find a valid set of size 10 using a greedy approach with backtracking?
    # 2. Or random restarts with local improvements?
    
    # Given the time limit, a simple random search for size 10 might not succeed.
    # But we can try to "compress" the superincreasing sequence.
    # e.g. 1, 2, 4, 8, 16, 32, 64, 120, 180, 240?
    
    # Let's try a few hardcoded candidates or a small random search.
    
    random.seed(42)
    
    for _ in range(5000): # Limit iterations to keep time low
        if time.time() > deadline:
            break
            
        # Generate a random set of size 10
        # Bias towards smaller numbers? Or spread out?
        # Let's try to perturb the base set.
        candidate = base_set[:]
        # Replace last few elements with random ones
        for i in range(7, 10): # indices 7, 8, 9 (if we expand)
            if i < len(candidate):
                candidate[i] = random.randint(1, 256)
            else:
                candidate.append(random.randint(1, 256))
        
        # Ensure distinct and sorted
        candidate = sorted(list(set(candidate)))
        
        if len(candidate) < 10:
            # Fill with random distinct numbers
            while len(candidate) < 10:
                x = random.randint(1, 256)
                if x not in candidate:
                    candidate.append(x)
            candidate = sorted(candidate)
            
        if is_valid(candidate):
            best_set = candidate
            break # Found a size 10 set
            
    # If we didn't find size 10, return best found (size 9)
    # Ensure return type is list of ints
    return list(best_set)
