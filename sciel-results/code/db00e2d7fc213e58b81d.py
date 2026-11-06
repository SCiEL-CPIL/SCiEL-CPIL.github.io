"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_sidon(subset):
    """Check if subset is Sidon: all unordered pair sums (i<=j) are unique."""
    n = len(subset)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(S):
    """Find the maximum size of a Sidon subset of S via exhaustive search."""
    n = len(S)
    # Check from largest possible size down
    for size in range(n, 0, -1):
        for combo in combinations(range(n), size):
            sub = [S[i] for i in combo]
            if _is_sidon(sub):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Start with a strong candidate: dense cluster
    # range(12) is dense, but maybe structured clusters are better
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    # Candidate generation
    candidates = set()
    candidates.add(tuple(sorted(best_set)))
    
    # 1. Arithmetic Progressions (often bad for Sidon if step is small? No, AP is isomorphic to range)
    # But AP with large step is sparse -> large Sidon subsets.
    # We want DENSE sets.
    
    # 2. Clusters
    # Try various cluster configurations
    # k clusters of size m
    for m in [2, 3, 4, 6]:
        k = 12 // m
        # Gap between clusters
        for gap in range(0, 30):
            # Base offset
            for base in range(0, 128 - (k-1)*(m+gap) + 1):
                s = []
                valid = True
                for c in range(k):
                    start = base + c * (m + gap)
                    if start + m - 1 > 128:
                        valid = False
                        break
                    s.extend(range(start, start + m))
                if valid and len(set(s)) == 12:
                    candidates.add(tuple(sorted(s)))
                    
    # 3. Random Dense Sets
    random.seed(42)
    for _ in range(500):
        # Sample from a smaller range to encourage density
        # Range 0 to 40
        s = random.sample(range(41), 12)
        candidates.add(tuple(sorted(s)))
        
    # 4. Structured "Collision" Sets
    # E.g., pairs of numbers that sum to same value
    # a, b, c, d such that a+b = c+d
    # This is hard to construct directly, random dense sets cover this.
    
    # Evaluate candidates
    for s in candidates:
        if time.time() > deadline:
            break
        score = _max_sidon_size(s)
        if score < best_score:
            best_score = score
            best_set = list(s)
            
    # Local Search
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_set_copy = list(best_set)
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                old_val = best_set_copy[i]
                # Try replacing with nearby values to maintain density
                # Or all values?
                # Since density is key, try values in [0, 50]
                for new_val in range(0, 51):
                    if new_val == old_val:
                        continue
                    # Check uniqueness
                    others = [x for j, x in enumerate(best_set_copy) if j != i]
                    if new_val in others:
                        continue
                    
                    best_set_copy[i] = new_val
                    score = _max_sidon_size(best_set_copy)
                    if score < best_score:
                        best_score = score
                        best_set = list(best_set_copy)
                        improved = True
                        break
                    best_set_copy[i] = old_val
                if improved:
                    break
                    
    return best_set

# EVOLVE-BLOCK-END
