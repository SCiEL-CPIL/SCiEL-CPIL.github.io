import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    if n == 0:
        return 0
    
    # We need to find the largest subset S of arr such that all sums s_i + s_j for i <= j are distinct.
    # This is equivalent to: for all (i,j) and (k,l) with i<=j, k<=l, if s_i+s_j == s_k+s_l then (i,j)==(k,l).
    
    # Brute force: try all subsets from largest to smallest
    # For n=12, C(12,12)+C(12,11)+... is manageable with pruning
    
    def is_sidon(subset):
        """Check if subset has all pair sums (with replacement) unique."""
        m = len(subset)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Try subsets in decreasing size order
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [arr[i] for i in combo]
            if is_sidon(subset):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Strategy: Search for a 12-element set in [0,128] that minimizes the largest Sidon subset.
    # Start with the tight AP {0,...,11} which is known to give max Sidon subset = 4.
    # Try to find if any set can achieve max Sidon subset = 3.
    # If not, confirm 4 is minimal and return the best set found.
    
    # First, verify the baseline
    baseline = list(range(12))
    baseline_score = _max_sidon_subset_size(baseline)
    
    if time.time() > deadline:
        return baseline
    
    # Try some structured alternatives that might reduce collisions further
    # or create more structured sets with lower max Sidon subsets
    
    # Try various APs with different starting points and differences
    candidates = [baseline]
    
    # AP with difference 1, different starts
    for start in range(0, 117):
        if time.time() > deadline:
            break
        cand = [start + i for i in range(12)]
        if max(cand) <= 128:
            candidates.append(cand)
    
    # APs with other small differences
    for diff in range(1, 11):
        for start in range(0, 128 - 11*diff + 1):
            if time.time() > deadline:
                break
            cand = [start + i*diff for i in range(12)]
            if max(cand) <= 128:
                candidates.append(cand)
    
    # Evaluate all candidates and track the best
    best_candidate = baseline
    best_score = baseline_score
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _max_sidon_subset_size(cand)
        if score < best_score:
            best_score = score
            best_candidate = cand
        elif score == best_score and cand != baseline:
            # Tie-breaking: prefer more "spread out" or different structure
            pass
    
    # If we haven't found score 3, the minimum is likely 4
    # Try a few more diverse sets to confirm
    if best_score > 3 and time.time() < deadline:
        # Try sets with more additive structure
        # E.g., union of two interleaved APs
        for d1 in range(1, 6):
            for d2 in range(1, 6):
                if time.time() > deadline:
                    break
                # Set: {0, d1, 2*d1, ..., 5*d1} union {offset, offset+d2, ..., offset+5*d2}
                for offset in range(0, 128 - 5*max(d1,d2)):
                    if time.time() > deadline:
                        break
                    set1 = set(i*d1 for i in range(6))
                    set2 = set(offset + i*d2 for i in range(6))
                    combined = set1 | set2
                    if len(combined) == 12 and max(combined) <= 128:
                        cand = sorted(combined)
                        score = _max_sidon_subset_size(cand)
                        if score < best_score:
                            best_score = score
                            best_candidate = cand
    
    return best_candidate
# EVOLVE-BLOCK-END
