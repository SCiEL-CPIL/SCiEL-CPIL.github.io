"""Covering code construction for ternary length-7, radius 1."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _all_words(q, n):
    """Generate all words as a numpy array of shape (q^n, n)."""
    return np.array(list(product(range(q), repeat=n)), dtype=np.uint8)

def _hamming_dist(w1, w2):
    """Hamming distance between two 1D arrays."""
    return np.sum(w1 != w2)

def _covered_by(w, codewords, radius):
    """Check if word w is within radius of any codeword."""
    for c in codewords:
        if _hamming_dist(w, c) <= radius:
            return True
    return False

def _ball_words(w, q, n, radius):
    """Generate all words within Hamming distance <= radius of w."""
    result = [list(w)]
    for r in range(1, radius + 1):
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    result.append(new_w)
    return result

def _ball_set(w, q, n, radius):
    """Generate set of words within Hamming distance <= radius of w as tuples."""
    s = {tuple(w)}
    for r in range(1, radius + 1):
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    s.add(tuple(new_w))
    return s

def _greedy_cover(q, n, radius, time_limit, rng_seed=42):
    """Greedy set covering algorithm with randomization."""
    rng = random.Random(rng_seed)
    start_time = time.time()
    
    all_words = _all_words(q, n)
    total = q ** n
    word_index = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls for each possible codeword position
    # For efficiency, we'll compute on the fly
    
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # Greedy: pick word whose ball covers most uncovered words
    while time.time() - start_time < time_limit * 0.4:
        best_word = None
        best_count = -1
        
        # Sample candidate words (not all 2187 for speed)
        candidates = list(range(total))
        rng.shuffle(candidates)
        # Check a subset first, then refine
        for idx in candidates[:200]:
            w = all_words[idx]
            ball = _ball_set(tuple(w), q, n, radius)
            count = sum(1 for b in ball if not covered[word_index[b]])
            if count > best_count:
                best_count = count
                best_word = idx
        
        if best_word is None or best_count <= 0:
            break
            
        w = all_words[best_word]
        ball = _ball_set(tuple(w), q, n, radius)
        for b in ball:
            covered[word_index[b]] = True
        codewords.append(list(w))
        
        if covered.all():
            break
    
    # Check if we've covered everything
    if not covered.all():
        # Add more codewords for uncovered words
        for idx in range(total):
            if not covered[idx] and time.time() - start_time < time_limit * 0.6:
                w = all_words[idx]
                ball = _ball_set(tuple(w), q, n, radius)
                for b in ball:
                    covered[word_index[b]] = True
                codewords.append(list(w))
    
    return codewords

def _local_optimize(codewords, q, n, radius, time_limit, rng_seed=42):
    """Try to reduce the number of codewords via local search."""
    rng = random.Random(rng_seed)
    start_time = time.time()
    
    all_words = _all_words(q, n)
    total = q ** n
    word_index = {tuple(w): i for i, w in enumerate(all_words)}
    
    current = [list(c) for c in codewords]
    
    # Try removing codewords one at a time
    for i in range(len(current)):
        if time.time() - start_time > time_limit:
            break
        # Check if removing current[i] still covers everything
        test_covered = np.zeros(total, dtype=bool)
        for j, c in enumerate(current):
            if j == i:
                continue
            ball = _ball_set(tuple(c), q, n, radius)
            for b in ball:
                test_covered[word_index[b]] = True
        if test_covered.all():
            current.pop(i)
            # Restart the removal check
            i = 0
    
    # Try replacing codewords with better ones
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        i = rng.randint(0, len(current) - 1)
        old_w = current[i]
        # Try random new codeword
        new_w = list(rng.choice(all_words))
        # Check if replacing old_w with new_w still covers everything
        test_covered = np.zeros(total, dtype=bool)
        for j, c in enumerate(current):
            if j == i:
                c = new_w
            ball = _ball_set(tuple(c), q, n, radius)
            for b in ball:
                test_covered[word_index[b]] = True
        if test_covered.all():
            # Check if new_w covers more than old_w (net gain)
            old_ball = _ball_set(tuple(old_w), q, n, radius)
            new_ball = _ball_set(tuple(new_w), q, n, radius)
            # Only accept if it doesn't increase size (it won't) and we can potentially remove another later
            current[i] = new_w
    
    return current

def _direct_sum_construction(q, n, radius):
    """Direct sum of ternary tetracode with all words on remaining coordinates."""
    if n < 4:
        # For small n, use exhaustive
        return [list(w) for w in product(range(q), repeat=n)]
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in product(range(q), repeat=n - 4)]

def construct(q, n, radius):
    """Construct a covering code of length n over alphabet {0..q-1} with covering radius."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with direct sum construction as baseline
    baseline = _direct_sum_construction(q, n, radius)
    best = baseline
    
    # Try greedy covering
    greedy_result = _greedy_cover(q, n, radius, time_limit=30, rng_seed=42)
    if len(greedy_result) < len(best):
        best = greedy_result
    
    # Try multiple random seeds for greedy
    for seed in [1, 7, 13, 99]:
        if time.time() > deadline - 40:
            break
        greedy_result = _greedy_cover(q, n, radius, time_limit=15, rng_seed=seed)
        if len(greedy_result) < len(best):
            best = greedy_result
    
    # Local optimization on best result
    if time.time() < deadline - 20:
        optimized = _local_optimize(best, q, n, radius, time_limit=deadline - time.time() - 5, rng_seed=42)
        if len(optimized) < len(best):
            best = optimized
    
    # Ensure we return distinct codewords
    seen = set()
    unique_best = []
    for c in best:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_best.append(c)
    
    return unique_best
# EVOLVE-BLOCK-END
