"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Standard recursive construction
    def standard_construction(n):
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Generate the standard word
    word = standard_construction(n)
    base_len = len(word)
    
    # Verify validity
    all_perms = set(permutations(range(1, n+1)))
    covered = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in all_perms:
            covered.add(w)
    
    if len(covered) != len(all_perms):
        return "".join(map(str, word))
    
    # Try to improve with greedy symbol-by-symbol reconstruction
    # with multiple restarts and randomization
    symbols = list(range(1, n+1))
    best_word = word[:]
    best_len = base_len
    
    def greedy_reconstruct(start_word, time_budget):
        """Rebuild word greedily, trying to minimize length."""
        uncovered = set(all_perms)
        new_word = list(start_word[:n])
        first_window = tuple(new_word)
        if first_window in uncovered:
            uncovered.discard(first_window)
        
        t0 = time.time()
        while uncovered and time.time() - t0 < time_budget:
            best_sym = None
            best_score = -1
            best_new_uncovered = None
            
            for sym in symbols:
                candidate = new_word + [sym]
                new_window = tuple(candidate[-n:])
                
                # Score: prefer covering new permutations
                if new_window not in all_perms:
                    score = 0
                elif new_window in uncovered:
                    score = 10
                else:
                    score = 1
                
                # Bonus for keeping the last n-1 as a valid (n-1)-perm
                last_n1 = tuple(candidate[-(n-1):])
                if len(set(last_n1)) == n - 1:
                    score += 3
                
                # Penalty for repeating symbols in the window
                if len(set(candidate[-n:])) < n:
                    score -= 5
                
                if score > best_score:
                    best_score = score
                    best_sym = sym
                    if new_window in all_perms and new_window in uncovered:
                        best_new_uncovered = new_window
            
            if best_sym is None:
                break
            
            new_word.append(best_sym)
            new_window = tuple(new_word[-n:])
            if new_window in uncovered:
                uncovered.discard(new_window)
        
        if not uncovered and len(new_word) < best_len:
            return new_word
        return None
    
    # Try multiple restarts with different initial segments
    num_restarts = 5
    time_per_restart = (deadline - start_time) / (num_restarts + 1)
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create a modified starting word
        start = list(word[:n])
        if r > 0:
            # Try different starting permutations
            random.shuffle(symbols)
            start = symbols[:n]
        
        result = greedy_reconstruct(start, time_per_restart * 0.8)
        if result:
            best_word = result
            best_len = len(result)
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
