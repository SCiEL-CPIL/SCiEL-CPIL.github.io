import time
import math
import random

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate classic superpermutation
    word = generate_classic(n)
    
    target_count = math.factorial(n)
    
    # Local search with bitset tracking
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
    Local search using efficient window tracking.
    Tries single deletions and block removals to reduce length.
    """
    L = len(word)
    if L < target_count + n - 1:
        return word
    
    # Precompute factorial for ranking
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i
    
    def perm_rank(p):
        """Compute rank of permutation p (0-indexed) using Lehmer code"""
        rank = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, p[i]):
                if not used[j]:
                    count += 1
            rank += count * fact[n - 1 - i]
            used[p[i]] = True
        return rank
    
    # Build the set of all valid permutation ranks
    seen = bytearray(target_count)
    seen_count = 0
    
    # Initialize: compute rank for each window
    window_ranks = []
    for i in range(L - n + 1):
        r = perm_rank(word[i:i+n])
        window_ranks.append(r)
        if not seen[r]:
            seen[r] = 1
            seen_count += 1
    
    # Verify we have all permutations
    assert seen_count == target_count, f"Only found {seen_count}/{target_count} permutations"
    
    best_word = word[:]
    best_len = L
    
    # Greedy deletion search with restarts
    max_rounds = 100
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        L = len(best_word)
        improved = False
        
        # Rebuild tracking structures for current best_word
        seen = bytearray(target_count)
        seen_count = 0
        window_ranks = []
        for i in range(L - n + 1):
            r = perm_rank(best_word[i:i+n])
            window_ranks.append(r)
            if not seen[r]:
                seen[r] = 1
                seen_count += 1
        
        # Build rank -> count map for quick uniqueness check
        rank_count = [0] * target_count
        for r in window_ranks:
            rank_count[r] += 1
        
        # Try each single deletion
        for i in range(1, L - 1):  # Don't delete first or last char
            if time.time() > deadline:
                break
            
            # Quick check: if the window at position i has a unique permutation, skip
            # Window i covers positions i..i+n-1
            # If we delete position i, window i changes
            # The key insight: only windows near position i are affected
            
            j_min = max(0, i - n + 1)
            j_max = min(i, L - n)
            
            # Check if any permutation in affected windows becomes uncovered
            # A permutation r is at risk if all its occurrences are in [j_min, j_max]
            # AND it doesn't appear in the new affected windows
            
            # For speed, just check if rank_count[r] > 1 for all affected windows
            # This is a conservative heuristic that may miss some valid deletions
            # but is much faster
            
            valid = True
            for j in range(j_min, j_max + 1):
                r = window_ranks[j]
                if rank_count[r] == 1:
                    # This permutation only appears in this window
                    # We need to verify it appears in a new window
                    # For now, reject (conservative)
                    valid = False
                    break
            
            if valid:
                # Perform the deletion
                best_word = best_word[:i] + best_word[i+1:]
                improved = True
                best_len = len(best_word)
                break
        
        if not improved:
            # Try block removals (remove k consecutive elements)
            for k in range(2, min(n + 2, L - 1)):
                if time.time() > deadline:
                    break
                
                for s in range(1, L - k):  # Start position
                    if time.time() > deadline:
                        break
                    
                    # Check if removing positions s..s+k-1 is valid
                    # This affects windows that overlap with [s, s+k-1]
                    j_min = max(0, s - n + 1)
                    j_max = min(s + k - 1, L - n)
                    
                    # Conservative check: all affected windows must have count > 1
                    valid = True
                    for j in range(j_min, j_max + 1):
                        if rank_count[window_ranks[j]] == 1:
                            valid = False
                            break
                    
                    if valid:
                        best_word = best_word[:s] + best_word[s+k:]
                        improved = True
                        best_len = len(best_word)
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return best_word
