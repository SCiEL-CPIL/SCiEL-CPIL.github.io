"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # 2. For each target, find the minimum number of terms in a valid antichain representation
    # We'll use a systematic search for each target, trying smaller counts first
    
    # Precompute which smooth numbers can divide which others
    # For antichain check: no term divides another
    # We need distinct terms, so we check all pairs
    
    def is_antichain(terms):
        """Check if no term divides another (antichain property)."""
        n = len(terms)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = terms[i], terms[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True
    
    # For each target, find minimal representation
    answers = [None] * 128
    
    # Sort smooth numbers for efficient searching
    smooth_sorted = sorted(smooth)
    
    # For efficiency, precompute for each target the valid representations by count
    # Start with count=1, then 2, then 3, etc.
    
    for target in range(1, 129):
        if time.time() > deadline:
            break
        
        found = False
        # Try counts from 1 up to 30
        for count in range(1, 31):
            if time.time() > deadline:
                break
            
            # Get smooth numbers <= target
            candidates = [x for x in smooth_sorted if x <= target]
            
            if len(candidates) < count:
                break
            
            # Try combinations of 'count' terms from candidates
            # For efficiency, use pruning
            for combo in itertools.combinations(candidates, count):
                if sum(combo) == target and is_antichain(combo):
                    answers[target - 1] = list(combo)
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            # Fallback: should not happen, but just in case
            answers[target - 1] = [target] if target in smooth else [1, target - 1] if target > 1 else [1]
    
    # Ensure all answers are valid
    for i in range(128):
        if answers[i] is None:
            # Last resort fallback
            n = i + 1
            if n in smooth:
                answers[i] = [n]
            else:
                # Find any valid representation
                for count in range(1, 31):
                    candidates = [x for x in smooth_sorted if x <= n]
                    if len(candidates) >= count:
                        for combo in itertools.combinations(candidates, count):
                            if sum(combo) == n and is_antichain(combo):
                                answers[i] = list(combo)
                                break
                    if answers[i] is not None:
                        break
                if answers[i] is None:
                    answers[i] = [n]
    
    return answers
# EVOLVE-BLOCK-END
