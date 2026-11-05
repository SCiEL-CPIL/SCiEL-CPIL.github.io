"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
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
    
    # Build standard construction
    def build_standard(n):
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
    
    # Verify a word contains all permutations
    def verify(word, n):
        all_perms = set(permutations(range(1, n+1)))
        covered = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            if w in all_perms:
                covered.add(w)
        return len(covered) == len(all_perms), covered
    
    if n == 4:
        # Known shortest is 33
        # Use the standard construction and try to shorten
        word = build_standard(n)
        
        # Try removal-based local search
        all_perms = set(permutations(range(1, n+1)))
        
        # Count occurrences of each permutation window
        def count_windows(word, n):
            counts = {}
            for i in range(len(word) - n + 1):
                w = tuple(word[i:i+n])
                counts[w] = counts.get(w, 0) + 1
            return counts
        
        counts = count_windows(word, n)
        
        # Try removing symbols from positions where the window is covered elsewhere
        improved = list(word)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            counts = count_windows(improved, n)
            # Find positions where removal is safe
            for pos in range(len(improved) - 1, 0, -1):
                if time.time() > deadline - 5:
                    break
                # Check if removing position pos is safe
                # We need to check all windows that include this position
                # A window at index i includes position pos if i <= pos <= i+n-1
                # So i ranges from max(0, pos-n+1) to min(pos, len(improved)-n)
                safe = True
                for i in range(max(0, pos - n + 1), min(pos, len(improved) - n) + 1):
                    w = tuple(improved[i:i+n])
                    if w in all_perms:
                        if counts.get(w, 0) <= 1:
                            safe = False
                            break
                if safe:
                    improved = improved[:pos] + improved[pos+1:]
                    changed = True
                    break
        
        if verify(improved, n)[0]:
            word = improved
        return "".join(map(str, word))
    
    # For n >= 5, use standard construction then try to shorten
    word = build_standard(n)
    base_len = len(word)
    
    # Verify standard word is valid
    valid, _ = verify(word, n)
    if not valid:
        return "".join(map(str, word))
    
    # Try removal-based local search to shorten
    all_perms = set(permutations(range(1, n+1)))
    
    def count_windows(word, n):
        counts = {}
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            counts[w] = counts.get(w, 0) + 1
        return counts
    
    improved = list(word)
    
    # Multiple passes of removal
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline - 10:
            break
        changed = False
        counts = count_windows(improved, n)
        
        # Try removing from the end first (often safer)
        for pos in range(len(improved) - 1, 0, -1):
            if time.time() > deadline - 10:
                break
            # Check if removing position pos is safe
            safe = True
            for i in range(max(0, pos - n + 1), min(pos, len(improved) - n) + 1):
                w = tuple(improved[i:i+n])
                if w in all_perms:
                    if counts.get(w, 0) <= 1:
                        safe = False
                        break
            if safe:
                improved = improved[:pos] + improved[pos+1:]
                changed = True
                break
        
        if not changed:
            break
    
    # Verify the improved word
    valid_improved, _ = verify(improved, n)
    if valid_improved:
        word = improved
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
