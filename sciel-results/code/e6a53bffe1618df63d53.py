"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Classic recursive construction
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # Try to shorten the word by finding and removing redundant parts
    # For n >= 5, the classic construction is not optimal
    if n >= 5 and time.time() < deadline:
        word = try_shorten(word, n, deadline)
    
    return "".join(map(str, word))

def try_shorten(word, n, deadline):
    """Try to shorten a superpermutation by finding redundant patterns."""
    best = list(word)
    best_len = len(best)
    
    # Try local search: look for patterns that can be compressed
    # One approach: find repeated (n-1)-length windows and see if we can merge them
    # This is a heuristic and may not always work
    
    # For now, return the original word if no improvement is found quickly
    # A more sophisticated approach would use simulated annealing or genetic algorithms
    # but that may be too slow for the time budget
    
    return best
# EVOLVE-BLOCK-END
