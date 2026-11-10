import time
import math
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Generate classic superpermutation
    word = generate_classic(n)
    
    target_count = math.factorial(n)
    
    # Local search with optimized tracking
    word = local_search_optimize(word, n, target_count, start_time, deadline)
    
    return "".join(map(str, word))

def generate_classic(n):
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

def local_search_optimize(word, n, target_count, start_time, deadline):
    """
    Aggressive local search using random deletions and swaps with exact validity checks.
    """
    L = len(word)
    if L < target_count + n - 1:
        return word
    
    # Precompute lookup table for permutation ranks
    perms = list(itertools.permutations(range(1, n + 1)))
    rank_lookup = {p: i for i, p in enumerate(perms)}
    
    def is_valid_exact(candidate, n, target_count):
        L_c = len(candidate)
        if L_c < target_count + n - 1:
            return False
        seen = bytearray(target_count)
        for i in range(L_c - n + 1):
            r = rank_lookup[tuple(candidate[i:i+n])]
            seen[r] = 1
        # Check if all permutations are seen
        return sum(seen) == target_count

    best_word = word[:]
    no_improvement_count = 0
    max_no_improvement = 5000  # Threshold to try swaps or break
    
    while time.time() < deadline:
        L = len(best_word)
        
        # Phase 1: Random Deletions
        improved = False
        attempts = 100
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            i = random.randint(1, L - 2)
            # Delete i
            best_word[i] = best_word[-1]
            best_word.pop()
            
            if is_valid_exact(best_word, n, target_count):
                improved = True
                no_improvement_count = 0
                break
            else:
                # Revert
                best_word.append(best_word[i])
                best_word[i] = best_word[-1]
                best_word.pop()
        
        if not improved:
            no_improvement_count += attempts
            if no_improvement_count > max_no_improvement:
                # Phase 2: Random Swaps to escape local minima
                swap_improved = False
                swap_attempts = 100
                for _ in range(swap_attempts):
                    if time.time() > deadline:
                        break
                    
                    i = random.randint(1, L - 2)
                    # Swap i and i+1
                    best_word[i], best_word[i+1] = best_word[i+1], best_word[i]
                    
                    if is_valid_exact(best_word, n, target_count):
                        swap_improved = True
                        no_improvement_count = 0 # Reset counter on successful swap
                        break
                    else:
                        # Revert
                        best_word[i], best_word[i+1] = best_word[i+1], best_word[i]
                
                if not swap_improved:
                    # Stuck
                    break
        else:
            no_improvement_count = 0

    return best_word
