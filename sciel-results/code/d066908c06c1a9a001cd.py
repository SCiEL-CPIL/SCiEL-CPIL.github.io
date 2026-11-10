import time
import itertools
import random

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    if n == 0:
        return 0
    
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

def _count_3ap(subset):
    """Count number of 3-term arithmetic progressions in subset."""
    s = set(subset)
    count = 0
    for a in subset:
        for b in subset:
            if a != b:
                c = 2*b - a
                if c in s and c > b: # Avoid double counting and self
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Baseline
    baseline = list(range(12))
    baseline_score = _max_sidon_subset_size(baseline)
    
    best_candidate = baseline
    best_score = baseline_score
    
    if best_score <= 3:
        return best_candidate

    random.seed(42)
    
    # Strategy 1: Systematic search in small windows with heuristic filtering
    # We search for sets with many additive collisions.
    # Sets with many 3-term APs are likely to have smaller max Sidon subsets.
    
    # Generate all subsets of [0, 20]
    # C(21, 12) = 293930
    # We can't evaluate all exactly. We filter by 3-AP count.
    
    base_range = range(21)
    candidates = []
    
    # Pre-compute 3-AP count for all subsets?
    # Or just generate and filter.
    
    # To save time, we can sample or use a smarter generation.
    # Let's try to generate all subsets of [0, 18] first.
    # C(19, 12) = 50388.
    
    for combo in itertools.combinations(range(19), 12):
        if time.time() > deadline - 10:
            break
        cand = list(combo)
        # Heuristic: count 3-APs
        # A faster 3-AP count:
        s = set(cand)
        ap_count = 0
        # Since size is small, simple loop is fine
        for i in range(12):
            for j in range(12):
                if i != j:
                    # Check if 2*cand[j] - cand[i] is in s
                    val = 2 * cand[j] - cand[i]
                    if val in s:
                        ap_count += 1
        candidates.append((ap_count, cand))
        
    # Sort by ap_count descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Evaluate top candidates
    eval_count = 0
    for ap_count, cand in candidates:
        if time.time() > deadline - 5:
            break
        score = _max_sidon_subset_size(cand)
        eval_count += 1
        if score < best_score:
            best_score = score
            best_candidate = cand
            if best_score == 3:
                return best_candidate
        # Stop if we've evaluated enough or if ap_count is very low
        if eval_count > 5000:
            break
            
    # Strategy 2: Random search in larger windows
    # If we didn't find a good one, try random sets in [0, 30]
    for span in range(20, 35):
        if time.time() > deadline:
            break
        max_start = 128 - span
        num_samples = 200
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            start = random.randint(0, max_start)
            cand = random.sample(range(start, start + span + 1), 12)
            score = _max_sidon_subset_size(cand)
            if score < best_score:
                best_score = score
                best_candidate = cand
                if best_score == 3:
                    return best_candidate

    return best_candidate
