"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction of superpermutation
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, the classic construction is optimal, return immediately
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, attempt greedy deletion to shorten the word
    # Precompute all permutations of 1..n
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Count how many times each permutation appears as a window
    from collections import Counter
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        perm_counts[window] += 1
    
    # A permutation is "covered" if its count is >= 1
    # We can delete position i if no permutation becomes uncovered
    
    def is_valid_fast(word_list):
        """Check if word_list contains all permutations as windows."""
        if len(word_list) < n:
            return False
        seen = set()
        for i in range(len(word_list) - n + 1):
            seen.add(tuple(word_list[i:i + n]))
        return all_perms.issubset(seen)
    
    # Greedy deletion: try to remove each position
    # To be efficient, track which permutations would be affected
    changed = True
    while changed and (time.time() - start_time) < deadline - 5:
        changed = False
        for i in range(len(word)):
            if time.time() - start_time > deadline - 5:
                break
            
            # Determine which windows are affected by removing position i
            # Windows that include position i: from max(0, i-n+1) to min(i, len(word)-n)
            # A window starting at s includes position i if s <= i <= s+n-1
            # So s ranges from max(0, i-n+1) to min(i, len(word)-n)
            
            # Find all windows that contain position i
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            
            # Check if removing position i would uncover any permutation
            # For each affected window, check if it's the only occurrence
            safe = True
            for s in range(s_min, s_max + 1):
                window = tuple(word[s:s + n])
                if window in all_perms and perm_counts[window] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove position i
                word.pop(i)
                changed = True
                # Update perm_counts for affected windows
                # After removal, windows that were at positions >= i shift left by 1
                # We need to recompute counts for windows near position i
                # For efficiency, recompute all counts if word is not too large
                if len(word) < 20000:
                    perm_counts = Counter()
                    for s in range(len(word) - n + 1):
                        window = tuple(word[s:s + n])
                        perm_counts[window] += 1
                break  # Restart scan as indices changed
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
