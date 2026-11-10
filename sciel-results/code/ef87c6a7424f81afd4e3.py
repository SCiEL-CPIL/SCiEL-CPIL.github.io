import collections
import random

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
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
            # Find maximal overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 6:
        return "".join(map(str, word))
        
    # For n >= 7, use Randomized Greedy Deletion
    current_word = word[:]
    L = len(current_word)
    
    # Initialize counts
    def compute_counts(w):
        c = collections.Counter()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                c[window] += 1
        return c

    counts = compute_counts(current_word)
    
    random.seed(42)
    
    # Try to remove as many characters as possible using random order
    # We do multiple passes with random orders
    for _ in range(10):
        indices = list(range(len(current_word)))
        random.shuffle(indices)
        
        removed_any = False
        for i in indices:
            # Check if i is still valid index
            if i >= len(current_word):
                continue
            
            # Identify windows containing i in current_word
            s_min = max(0, i - n + 1)
            s_max = min(i, len(current_word) - n)
            
            destroyed = []
            for s in range(s_min, s_max + 1):
                w = tuple(current_word[s:s+n])
                if len(set(w)) == n:
                    destroyed.append(w)
            
            # Check if any destroyed window has count 1
            safe = True
            for w in destroyed:
                if counts[w] <= 1:
                    safe = False
                    break
            
            if safe:
                # Perform deletion
                val = current_word[i]
                del current_word[i]
                
                # Update counts:
                # 1. Decrement destroyed
                for w in destroyed:
                    counts[w] -= 1
                    if counts[w] == 0:
                        del counts[w] # Optional cleanup
                
                # 2. Increment created
                # New windows containing the "gap"
                # The gap is at index i in the new word?
                # The character at i is now old i+1.
                # Windows containing index i in new word:
                new_s_min = max(0, i - n + 1)
                new_s_max = min(i, len(current_word) - n)
                
                for s in range(new_s_min, new_s_max + 1):
                    w = tuple(current_word[s:s+n])
                    if len(set(w)) == n:
                        counts[w] += 1
                
                removed_any = True
                # Note: indices in `indices` list are now stale for j > i
                # But we are iterating over a shuffled list of original indices.
                # If we removed i, then for any j > i in the list, the actual index is j-1.
                # This is tricky.
                # Simpler: Just restart the pass if we remove anything?
                # Or break and start a new pass.
                break # Break inner loop, restart outer pass with new length
                
        if not removed_any:
            break
            
    return "".join(map(str, current_word))
