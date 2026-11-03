"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
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
    if n >= 3 and time.time() < deadline:
        word = try_shorten_fast(word, n, deadline)
    
    return "".join(map(str, word))

def get_permutations_covered(word, n):
    """Get set of all permutations covered by the word."""
    covered = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            covered.add(window)
    return covered

def try_shorten_fast(word, n, deadline):
    """Try to shorten a superpermutation by removing redundant symbols using fast count updates."""
    current = list(word)
    L = len(current)
    
    # Initialize permutation counts
    perm_counts = defaultdict(int)
    total_perms = 0
    for i in range(L - n + 1):
        window = tuple(current[i:i + n])
        if len(set(window)) == n:
            perm_counts[window] += 1
            total_perms += 1
            
    # We want to keep all n! permutations
    # Note: The initial word should cover all permutations.
    # If not, we can't shorten.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate from right to left to keep indices stable during removal
        for pos in range(len(current) - 1, -1, -1):
            if time.time() >= deadline:
                break
            
            if len(current) - 1 < n:
                continue
                
            # Identify affected windows in current word
            # Windows containing index 'pos' are those starting at k where pos-n+1 <= k <= pos
            start_k = max(0, pos - n + 1)
            end_k = min(pos, len(current) - n)
            
            # Windows to remove (decrement)
            windows_to_remove = []
            for k in range(start_k, end_k + 1):
                w = tuple(current[k:k + n])
                if len(set(w)) == n:
                    windows_to_remove.append(w)
            
            # Windows to add (increment) in the new word
            # New word length is L-1.
            # Windows starting at j in new word that are affected are those spanning the deletion.
            # Deletion is at 'pos'.
            # New windows starting at j where j < pos and j+n-1 >= pos?
            # Actually, in the new word, the characters after pos are shifted left.
            # So new window starting at j (where j < pos) will take chars from current[j:pos] and current[pos+1:pos+1+(j+n-pos)]
            # This is valid if j+n-1 >= pos in the new word.
            # In new word, index pos is removed.
            # So new window starting at j ends at j+n-1.
            # If j+n-1 >= pos, it crosses the gap.
            # The characters are current[j:pos] + current[pos+1:pos+1+(j+n-pos)]
            # This is equivalent to current[j:j+n] with current[pos] removed.
            
            windows_to_add = []
            # j ranges from 0 to len(new)-n
            # Affected j are those where the window content changes.
            # Content changes if the window in 'current' included 'pos' or shifted.
            # Windows in 'current' starting at k < pos are unchanged if k+n-1 < pos.
            # So affected j in new word correspond to k in [pos-n+1, pos-1] in old word?
            # Let's just calculate the new windows for j in [pos-n+1, pos-1] (clamped to valid range)
            # And also, windows starting at j >= pos in new word are exactly windows starting at j+1 in old word.
            # So they are already accounted for in perm_counts (they don't change content, just index).
            # Wait, if we remove 'pos', then new[j] = old[j+1] for j >= pos.
            # So new window starting at j is old window starting at j+1.
            # So the set of windows starting at j >= pos in new word is the same as set of windows starting at j+1 >= pos+1 in old word.
            # So we don't need to add/remove these, they are just shifted.
            # The only windows that change content are those starting at j < pos that now include chars from after pos.
            # These are j such that j < pos and j+n-1 >= pos.
            # i.e., pos-n+1 <= j <= pos-1.
            
            for j in range(max(0, pos - n + 1), pos):
                # Construct new window
                # It takes current[j:pos] and current[pos+1:pos+1+(j+n-pos)]
                # Length of first part: pos - j
                # Length of second part: (j+n) - pos
                # Total length n.
                # Check if within bounds of current
                # current length L.
                # Max index in second part: pos + (j+n-pos) - 1 = j+n-1.
                # Since j <= pos-1, j+n-1 <= pos+n-2.
                # We need j+n-1 < L.
                if j + n - 1 < L:
                    w_new = tuple(current[j:pos] + current[pos+1:j+n])
                    if len(set(w_new)) == n:
                        windows_to_add.append(w_new)
            
            # Apply changes
            valid = True
            for w in windows_to_remove:
                perm_counts[w] -= 1
                if perm_counts[w] == 0:
                    valid = False
                    # Revert
                    for w2 in windows_to_remove:
                        perm_counts[w2] += 1
                    break
            
            if valid:
                for w in windows_to_add:
                    perm_counts[w] += 1
                
                # If valid, perform removal
                current = current[:pos] + current[pos+1:]
                improved = True
                break # Restart loop with new length
    
    return current
# EVOLVE-BLOCK-END
