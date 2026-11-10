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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Baseline: AP {0, ..., 11}
    baseline = list(range(12))
    baseline_score = _max_sidon_subset_size(baseline)
    
    best_candidate = baseline
    best_score = baseline_score
    
    if best_score <= 3:
        return best_candidate

    random.seed(42)
    
    # Exhaustive search for small spans
    # Translation invariance: we can assume min(element) = 0
    # So we search subsets of {0, ..., span} of size 12.
    
    # Spans 11 to 16 are small enough to be exhaustive quickly
    for span in range(11, 17):
        if time.time() > deadline:
            break
            
        base_set = list(range(span + 1))
        
        # Generate all 12-subsets of {0, ..., span}
        for combo in itertools.combinations(base_set, 12):
            if time.time() > deadline:
                break
            cand = list(combo)
            score = _max_sidon_subset_size(cand)
            if score < best_score:
                best_score = score
                best_candidate = cand
                if best_score == 3:
                    return best_candidate
                    
    # Random search for larger spans
    # Focus on spans 17 to 50
    # Also try some structured sets
    
    # Structured sets: Unions of small APs
    # Example: {0, 1, 2, 3} U {10, 11, 12, 13} U {20, 21, 22, 23}
    # This creates many collisions within clusters.
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Random span
        span = random.randint(17, 60)
        start = random.randint(0, 128 - span)
        
        # Random sample
        cand = random.sample(range(start, start + span + 1), 12)
        score = _max_sidon_subset_size(cand)
        if score < best_score:
            best_score = score
            best_candidate = cand
            if best_score == 3:
                return best_candidate

    # Try specific structured patterns
    # Pattern 1: Clusters
    for c1 in range(1, 6):
        for c2 in range(1, 6):
            for c3 in range(1, 6):
                if time.time() > deadline:
                    break
                # 4 clusters of size 3
                # {0, 1, 2}, {10, 11, 12}, {20, 21, 22}, {30, 31, 32}
                # Vary the gaps
                gap = random.randint(5, 15)
                cand = []
                base = 0
                for i in range(4):
                    for j in range(3):
                        cand.append(base + j)
                    base += gap
                if max(cand) <= 128:
                    cand = sorted(set(cand))
                    if len(cand) == 12:
                        score = _max_sidon_subset_size(cand)
                        if score < best_score:
                            best_score = score
                            best_candidate = cand
                            if best_score == 3:
                                return best_candidate

    return best_candidate
