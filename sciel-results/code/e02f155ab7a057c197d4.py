"""Optimized ternary covering code constructor."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter


def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))


def _is_covered(word, codewords, q, n, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _deltas_within_radius(q, n, radius):
    """Generate all delta vectors with Hamming weight <= radius."""
    deltas = []
    deltas.append([0] * n)
    for weight in range(1, radius + 1):
        for positions in itertools.combinations(range(n), weight):
            for values in itertools.product(range(1, q), repeat=weight):
                delta = [0] * n
                for pos, val in zip(positions, values):
                    delta[pos] = val
                deltas.append(delta)
    return deltas


def _compute_coverage_counts(codewords, q, n, radius, deltas):
    """
    For each word in the space, count how many codewords cover it.
    Returns a dict mapping word (tuple) -> count.
    Only words with count >= 1 are included.
    """
    coverage = Counter()
    for cw in codewords:
        cw_tuple = tuple(cw)
        for delta in deltas:
            word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
            coverage[word] += 1
    return coverage


def _greedy_cover(q, n, radius, time_limit=110.0):
    """
    Greedy set cover approach with efficient removal using coverage counts.
    """
    start_time = time.time()
    
    # Precompute deltas
    deltas = _deltas_within_radius(q, n, radius)
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n <= 4:
        if n == 1:
            return [[i] for i in range(q)]
        elif n == 2:
            return [list(p) for p in itertools.product(range(q), repeat=2)]
        elif n == 3:
            return [[a, b, (a + b) % 3] for a in range(3) for b in range(3)]
        else:
            return tetracode[:9] if len(tetracode) >= 9 else tetracode
    
    # Build initial code using direct sum
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    random.shuffle(initial_code)
    codewords = list(initial_code)
    
    # Compute initial coverage counts
    coverage = _compute_coverage_counts(codewords, q, n, radius, deltas)
    
    # Phase 1: Repeatedly remove codewords that don't uniquely cover any word
    # A codeword can be removed if for every word it covers, coverage[word] > 1
    max_remove_iters = 200
    for iteration in range(max_remove_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find a removable codeword
        removable_idx = -1
        for idx in range(len(codewords)):
            cw = codewords[idx]
            cw_tuple = tuple(cw)
            removable = True
            for delta in deltas:
                word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
                if coverage[word] <= 1:
                    removable = False
                    break
            if removable:
                removable_idx = idx
                break
        
        if removable_idx == -1:
            break
        
        # Remove the codeword and update coverage
        cw = codewords.pop(removable_idx)
        cw_tuple = tuple(cw)
        for delta in deltas:
            word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
            coverage[word] -= 1
            if coverage[word] == 0:
                del coverage[word]
    
    # Phase 2: Try to find better codewords by replacing existing ones
    # For each codeword, try swapping it with a nearby word that covers more uncovered words
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            cw = codewords[idx]
            cw_tuple = tuple(cw)
            
            # Try replacing with a nearby word
            best_new_cw = None
            best_gain = 0
            
            # Consider words within distance 2 of cw
            for delta1 in deltas:
                for delta2 in deltas:
                    # Combined delta (not exactly right, but heuristic)
                    pass
            
            # Simpler: try random perturbations
            for _ in range(20):
                new_cw = list(cw)
                # Change 1-2 positions
                num_changes = random.randint(1, 2)
                positions = random.sample(range(n), num_changes)
                for pos in positions:
                    new_cw[pos] = (new_cw[pos] + random.randint(1, q-1)) % q
                
                new_tuple = tuple(new_cw)
                if new_tuple in [tuple(c) for c in codewords]:
                    continue
                
                # Calculate gain: how many words would have coverage decrease from 2 to 1
                # and how many would have coverage increase from 1 to 2
                gain = 0
                for delta in deltas:
                    word = tuple((new_tuple[i] + delta[i]) % q for i in range(n))
                    if word not in coverage:
                        gain += 1  # Newly covered
                    elif coverage[word] == 1:
                        gain += 1  # Was only covered by this, now covered by new too
                    else:
                        gain -= 1  # Was covered by multiple, now one less
                
                # Also consider words that were covered by old but not new
                for delta in deltas:
                    old_word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
                    if old_word not in [tuple((new_tuple[i] + d[i]) % q for i in range(n)) for d in deltas]:
                        if coverage.get(old_word, 0) <= 1:
                            gain -= 2  # Would become uncovered or less covered
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_cw = new_cw
            
            if best_new_cw is not None and best_gain > 0:
                # Replace
                old_tuple = tuple(cw)
                new_tuple = tuple(best_new_cw)
                
                # Update coverage for old word's sphere
                for delta in deltas:
                    word = tuple((old_tuple[i] + delta[i]) % q for i in range(n))
                    coverage[word] -= 1
                    if coverage[word] == 0:
                        del coverage[word]
                
                # Update coverage for new word's sphere
                for delta in deltas:
                    word = tuple((new_tuple[i] + delta[i]) % q for i in range(n))
                    coverage[word] += 1
                
                codewords[idx] = best_new_cw
                improved = True
        
        if not improved:
            break
    
    # Phase 3: Try to add codewords that cover many words with low coverage
    # Then prune
    for iteration in range(30):
        if time.time() - start_time > time_limit:
            break
        
        # Find words with coverage == 1 (most vulnerable)
        # Sample from the space
        sample_size = min(500, 3**n // 20)
        low_coverage_words = []
        for _ in range(sample_size):
            word = tuple(random.randint(0, q-1) for _ in range(n))
            if coverage.get(word, 0) <= 1:
                low_coverage_words.append(word)
        
        if not low_coverage_words:
            break
        
        # Find a candidate that covers many of these
        best_candidate = None
        best_score = 0
        existing_set = set(tuple(c) for c in codewords)
        
        candidates = set()
        for lw in low_coverage_words[:30]:
            for delta in deltas:
                cand = tuple((lw[i] + delta[i]) % q for i in range(n))
                if cand not in existing_set:
                    candidates.add(cand)
        
        for cand in list(candidates)[:100]:
            score = 0
            for lw in low_coverage_words:
                if _hamming_dist(list(cand), list(lw)) <= radius:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = list(cand)
        
        if best_candidate is not None and best_score > 5:
            # Add the codeword
            codewords.append(best_candidate)
            cand_tuple = tuple(best_candidate)
            for delta in deltas:
                word = tuple((cand_tuple[i] + delta[i]) % q for i in range(n))
                coverage[word] += 1
            
            # Prune: remove any codeword that is now redundant
            for idx in range(len(codewords) - 1, -1, -1):
                if time.time() - start_time > time_limit:
                    break
                
                cw = codewords[idx]
                cw_tuple = tuple(cw)
                removable = True
                for delta in deltas:
                    word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
                    if coverage[word] <= 1:
                        removable = False
                        break
                
                if removable:
                    for delta in deltas:
                        word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
                        coverage[word] -= 1
                        if coverage[word] == 0:
                            del coverage[word]
                    codewords.pop(idx)
    
    return codewords


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q
    with covering radius radius.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for the 120s limit
    
    # Handle edge cases
    if radius >= n:
        return [[0] * n]
    
    if n == 0:
        return [[]]
    
    if q == 1:
        return [[0] * n]
    
    # For small cases, use exact or near-exact methods
    if n <= 4:
        return _greedy_cover(q, n, radius, time_limit)
    
    # Use greedy optimization
    result = _greedy_cover(q, n, radius, time_limit)
    
    # Safety check: verify coverage if time permits
    if time.time() - start_time < time_limit - 5:
        # Quick verification on a sample
        deltas = _deltas_within_radius(q, n, radius)
        sample_size = min(200, 3**n // 10)
        for _ in range(sample_size):
            word = [random.randint(0, q-1) for _ in range(n)]
            if not _is_covered(word, result, q, n, radius):
                # Coverage failed, fall back to direct sum
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                if n >= 4:
                    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
                else:
                    return result
    
    return result
# EVOLVE-BLOCK-END
