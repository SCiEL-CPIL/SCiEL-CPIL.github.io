import random
import time
import math

def construct(n):
    # Step 1: Generate the classic greedy superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of m-1 symbols in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of the current 'new' word
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # If n < 7, the greedy construction is optimal
    if n < 7:
        return "".join(map(str, word))
        
    # For n >= 7, try to minimize length using local search (deletions)
    # The greedy length for n=7 is 5913. Optimal is 5905.
    # We attempt to remove redundant characters.
    
    start_time = time.time()
    # Allow 110 seconds for search, leaving 10s for safety/overhead
    deadline = start_time + 110.0 
    
    # Helper to calculate permutation counts
    def get_counts(w):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    # Initial counts
    current_word = list(word)
    current_counts = get_counts(current_word)
    
    best_word = current_word
    best_len = len(current_word)
    
    # Try multiple random deletion orders to escape local minima
    # The gap is small (8 chars), so this should be effective
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Start from the best found so far
        current_word = list(best_word)
        current_counts = get_counts(current_word)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Generate random order of indices to try deletion
            indices = list(range(len(current_word)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Identify windows affected by removing index i
                # A window of size n starting at 'start' includes index i if start <= i < start+n
                # So start can be from max(0, i-n+1) to min(i, len(current_word)-n)
                affected = []
                min_start = max(0, i - n + 1)
                max_start = min(i, len(current_word) - n)
                
                for start in range(min_start, max_start + 1):
                    p = tuple(current_word[start:start+n])
                    affected.append(p)
                
                # Check if removal is valid (no count drops to 0)
                valid = True
                for p in affected:
                    current_counts[p] -= 1
                    if current_counts[p] == 0:
                        valid = False
                        break
                
                if valid:
                    # Commit the removal
                    current_word.pop(i)
                    improved = True
                    # Restart the inner loop with new indices
                    break
                else:
                    # Restore counts
                    for p in affected:
                        current_counts[p] += 1
                    
        if len(current_word) < best_len:
            best_word = current_word
            best_len = len(current_word)
            # If we reached the known optimal for n=7, we can stop early
            if n == 7 and best_len == 5905:
                break

    return "".join(map(str, best_word))
