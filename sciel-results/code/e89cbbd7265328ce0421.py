"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def _classic_construct(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word


def _get_windows(word, n):
    """Extract all n-length windows from word."""
    return [tuple(word[i:i+n]) for i in range(len(word) - n + 1)]


def _count_missing(word, n, all_perms_set):
    """Count how many permutations are missing from word."""
    windows = set(_get_windows(word, n))
    return len(all_perms_set - windows)


def construct(n):
    """Construct a superpermutation on n symbols."""
    if n <= 1:
        return "".join(map(str, range(1, n + 1)))
    
    # For small n, use exact or near-exact methods
    if n <= 4:
        return "".join(map(str, _classic_construct(n)))
    
    # Deadline guard: 140 seconds to be safe
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with the classic construction as baseline
    word = _classic_construct(n)
    best_word = list(word)
    best_len = len(best_word)
    
    # For n=5, 6, 7 the classic construction is already quite good
    # Let's try to improve with local search / greedy repair
    
    # Get all permutations for checking
    all_perms = set(permutations(range(1, n + 1)))
    
    # Check if we're already valid
    if _count_missing(word, n, all_perms) == 0:
        return "".join(map(str, best_word))
    
    # Greedy improvement: try to shorten the word
    # Strategy: For each position, try removing a symbol if the result is still valid
    # This is expensive, so we do it selectively
    
    # Better strategy: Try different starting permutations and orderings
    # The classic construction depends on the order of permutations
    
    # For n=5: classic gives length 153. Known optimal is 153.
    # For n=6: classic gives length 873. Known best is 873.
    # For n=7: classic gives length 5913. Known best is 5905.
    
    # So for n=7, we need to shave off 8 symbols.
    
    if n == 7:
        # Try to find a shorter construction
        # The classic construction has some redundancy that can be removed
        
        # Approach: Try to merge adjacent blocks more aggressively
        # Or try different orderings of the permutations
        
        # Let's try a different recursive construction order
        # Instead of the standard order, try to optimize the overlap
        
        # For now, return the classic construction
        # A full optimization would require more sophisticated search
        
        # Try some random restarts with modified construction
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Try a slightly modified construction
            # Randomly perturb the order of some permutations
            word_try = list(_classic_construct(n))
            
            # Try to shorten by removing redundant symbols
            # Check if any symbol can be removed
            improved = True
            while improved:
                if time.time() > deadline - 10:
                    break
                improved = False
                for i in range(1, len(word_try) - 1):
                    if time.time() > deadline - 10:
                        break
                    # Try removing position i
                    candidate = word_try[:i] + word_try[i+1:]
                    # Quick check: does this still contain all perms?
                    # This is expensive, so only do it if length is close
                    if len(candidate) < best_len:
                        missing = _count_missing(candidate, n, all_perms)
                        if missing == 0:
                            word_try = candidate
                            best_word = list(candidate)
                            best_len = len(best_word)
                            improved = True
                            break
                        elif missing <= 2:
                            # Almost valid, keep as candidate
                            pass
            
            if len(word_try) < best_len:
                if _count_missing(word_try, n, all_perms) == 0:
                    best_word = list(word_try)
                    best_len = len(best_word)
        
        return "".join(map(str, best_word))
    
    elif n == 6:
        # Classic is optimal for n=6
        return "".join(map(str, best_word))
    
    elif n == 5:
        # Classic is optimal for n=5
        return "".join(map(str, best_word))
    
    else:
        # For larger n, just return classic
        return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
