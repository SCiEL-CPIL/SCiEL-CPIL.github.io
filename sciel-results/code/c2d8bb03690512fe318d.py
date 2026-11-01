"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s limit
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
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
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Efficient polish using coverage counts
    def build_coverage(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts
    
    improved = list(word)
    counts = build_coverage(improved, n)
    L = len(improved)
    
    # Use a worklist approach: try all positions, remove if safe
    # Process in multiple passes; each pass scans left to right
    pass_count = 0
    while time.time() < deadline:
        pass_count += 1
        changed = False
        # Scan left to right; after removal, indices shift, so restart scan
        i = 0
        while i < len(improved):
            if time.time() > deadline:
                return "".join(map(str, improved))
            
            # Windows affected by removing position i
            j_min = max(0, i - n + 1)
            j_max = min(i, len(improved) - n)
            
            safe = True
            affected = []
            for j in range(j_min, j_max + 1):
                w = tuple(improved[j:j+n])
                affected.append(w)
                if counts[w] <= 1:
                    safe = False
                    break
            
            if safe:
                for w in affected:
                    counts[w] -= 1
                improved.pop(i)
                changed = True
                # Don't increment i; check same position again (new element shifted in)
            else:
                i += 1
        
        if not changed:
            # Try a second pass in reverse direction for different removals
            i = len(improved) - 1
            while i >= 0:
                if time.time() > deadline:
                    return "".join(map(str, improved))
                
                j_min = max(0, i - n + 1)
                j_max = min(i, len(improved) - n)
                
                safe = True
                affected = []
                for j in range(j_min, j_max + 1):
                    w = tuple(improved[j:j+n])
                    affected.append(w)
                    if counts[w] <= 1:
                        safe = False
                        break
                
                if safe:
                    for w in affected:
                        counts[w] -= 1
                    improved.pop(i)
                    changed = True
                else:
                    i -= 1
            
            if not changed:
                break
    
    return "".join(map(str, improved))
# EVOLVE-BLOCK-END
