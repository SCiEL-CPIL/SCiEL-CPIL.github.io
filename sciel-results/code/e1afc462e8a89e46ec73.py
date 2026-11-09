import time
import random
from itertools import combinations

def is_sidon_4(subset):
    # subset is a tuple of 4 distinct integers
    # Check if all unordered pair sums (including diagonals) are unique
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_4(a_set):
    for comb in combinations(a_set, 4):
        if is_sidon_4(comb):
            return True
    return False

def count_sidon_4(a_set):
    count = 0
    for comb in combinations(a_set, 4):
        if is_sidon_4(comb):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s limit
    
    best_set = list(range(12))
    best_score = 4  # Baseline max Sidon size
    
    # We want to find a set with max Sidon size 3.
    # This means NO 4-subset is Sidon.
    
    # Strategy 1: Random search in small ranges
    # Bias towards smaller ranges to increase collision probability
    for _ in range(100000):
        if time.time() > deadline:
            break
        
        # Try small ranges first
        if random.random() < 0.7:
            r = random.randint(12, 20)
        else:
            r = random.randint(20, 35)
            
        if r < 11:
            continue
            
        s = random.sample(range(r + 1), 12)
        
        # Check if any 4-subset is Sidon
        if not has_sidon_4(s):
            # Found a set with max Sidon size 3
            return s
            
    # Strategy 2: Local search - try to break Sidon-4 subsets
    # Start from a random set in a small range and iteratively fix collisions
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        r = random.randint(12, 25)
        s = random.sample(range(r + 1), 12)
        s.sort()
        
        # Try to improve this set by moving elements to break Sidon-4s
        for step in range(20):
            if time.time() > deadline:
                break
                
            # Find a Sidon-4 subset if one exists
            target_subset = None
            for comb in combinations(s, 4):
                if is_sidon_4(comb):
                    target_subset = comb
                    break
            
            if target_subset is None:
                return s  # Found a set with no Sidon-4
                
            # Try to move one element to break the Sidon-4
            # Pick an element from the target subset
            idx_to_move = random.choice([i for i, v in enumerate(s) if v in target_subset])
            old_val = s[idx_to_move]
            
            # Try several new values
            for _ in range(15):
                new_val = random.randint(0, r)
                if new_val not in s:
                    s[idx_to_move] = new_val
                    s.sort()
                    
                    if not has_sidon_4(s):
                        return s  # Success!
                        
                    # Revert if no immediate success
                    s[idx_to_move] = old_val
                    s.sort()
                    
    # Strategy 3: Systematic search in very small ranges
    # Try all combinations in range [0, r] for small r
    for r in range(12, 18):
        if time.time() > deadline:
            break
        
        # For small ranges, we can be more systematic
        # Try random samples more aggressively
        for _ in range(5000):
            if time.time() > deadline:
                break
            s = random.sample(range(r + 1), 12)
            if not has_sidon_4(s):
                return s
    
    # If we didn't find a 3, return the baseline (which is 4)
    return best_set
