"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

# Fixed deadline: 115s (5s safety margin below 120s limit)
_DEADLINE = 115.0


def _time_left(deadline):
    return deadline - time.time()


def _hamming_distances_to_codewords(word, codewords):
    """Compute Hamming distance from word to each codeword. word is int (base-3), codewords are list of ints."""
    # Decode word to array
    w = np.array([int(word // (3 ** (n - 1 - i))) % 3 for i in range(n)])
    # This is slow; better to precompute
    pass


def _is_covered(word_arr, codeword_set):
    """Check if word_arr (numpy array) is within distance 1 of any codeword in codeword_set (set of tuples)."""
    # Check exact match
    if tuple(word_arr) in codeword_set:
        return True
    # Check all words at distance 1
    for i in range(len(word_arr)):
        orig = word_arr[i]
        for v in range(3):
            if v == orig:
                continue
            word_arr[i] = v
            if tuple(word_arr) in codeword_set:
                word_arr[i] = orig
                return True
        word_arr[i] = orig
    return False


def _uncovered_count(codewords, n):
    """Count how many words are NOT covered by codewords. Returns count."""
    codeword_set = set(tuple(c) for c in codewords)
    uncovered = 0
    for word in itertools.product(range(3), repeat=n):
        if not _is_covered(np.array(word), codeword_set):
            uncovered += 1
    return uncovered


def _greedy_cover(q, n, radius, deadline, seed=0):
    """Greedy set-cover heuristic with symmetry-aware tie-breaking."""
    rng = random.Random(seed)
    
    # Precompute all words as numpy arrays for speed
    all_words = list(itertools.product(range(q), repeat=n))
    all_words = [np.array(w) for w in all_words]
    
    # Build coverage: for each candidate codeword, which words does it cover?
    # For radius 1, a codeword covers itself and all words at distance 1.
    # Number of words covered by one codeword: 1 + n*(q-1)
    # For q=3, n=6: 1 + 6*2 = 13 words per codeword
    
    # Map each word to its index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (words within distance 1) for each word
    # neighbors[i] = list of indices of words within distance 1 of all_words[i]
    neighbors = []
    for i, w in enumerate(all_words):
        covered = [i]
        for pos in range(n):
            orig = w[pos]
            for v in range(q):
                if v == orig:
                    continue
                w[pos] = v
                idx = word_to_idx.get(tuple(w))
                if idx is not None:
                    covered.append(idx)
                w[pos] = orig
        neighbors.append(covered)
    
    # Greedy set cover
    covered_mask = np.zeros(len(all_words), dtype=bool)
    codewords = []
    
    while not covered_mask.all():
        if time.time() > deadline - 5:
            break
        
        # Find the codeword that covers the most uncovered words
        best_gain = -1
        best_idx = -1
        
        # Sample a subset of candidates for speed (all 729 is small enough to scan)
        for i in range(len(all_words)):
            if covered_mask[i]:
                # Skip if this word is already covered (it could still be a good codeword)
                # Actually, we should consider all candidates
                pass
            
            gain = 0
            for j in neighbors[i]:
                if not covered_mask[j]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            elif gain == best_gain and gain > 0:
                # Tie-breaking: prefer codewords that are "spread out" from existing ones
                # Use symmetry: prefer words with more distinct coordinate patterns
                # Simple heuristic: prefer words that differ from existing codewords in more positions
                if codewords:
                    min_dist = min(np.sum(all_words[i] != np.array(c)) for c in codewords)
                    if best_idx >= 0:
                        prev_min_dist = min(np.sum(all_words[best_idx] != np.array(c)) for c in codewords)
                        if min_dist > prev_min_dist:
                            best_idx = i
                    else:
                        best_idx = i
        
        if best_idx < 0 or best_gain <= 0:
            # Can't improve; break to avoid infinite loop
            break
        
        # Add this codeword
        codewords.append([int(x) for x in all_words[best_idx]])
        for j in neighbors[best_idx]:
            covered_mask[j] = True
    
    return codewords


def _local_search(codewords, n, q, deadline, seed=0):
    """Try to reduce the number of codewords by swapping or removing."""
    rng = random.Random(seed)
    all_words = list(itertools.product(range(q), repeat=n))
    all_words_arr = [np.array(w) for w in all_words]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words_arr):
        covered = [i]
        for pos in range(n):
            orig = w[pos]
            for v in range(q):
                if v == orig:
                    continue
                w[pos] = v
                idx = word_to_idx.get(tuple(w))
                if idx is not None:
                    covered.append(idx)
                w[pos] = orig
        neighbors.append(covered)
    
    best_codewords = list(codewords)
    best_len = len(best_codewords)
    
    while time.time() < deadline - 10:
        improved = False
        
        # Try removing each codeword
        for i in range(len(best_codewords)):
            if time.time() > deadline - 10:
                break
            
            trial = best_codewords[:i] + best_codewords[i+1:]
            if not trial:
                continue
            
            # Check coverage
            covered_mask = np.zeros(len(all_words), dtype=bool)
            for c in trial:
                idx = word_to_idx.get(tuple(c))
                if idx is not None:
                    for j in neighbors[idx]:
                        covered_mask[j] = True
            
            if covered_mask.all():
                best_codewords = trial
                best_len -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: replace one codeword with another
        if len(best_codewords) > 1:
            # Sample some swaps
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                
                i = rng.randrange(len(best_codewords))
                new_word = rng.choice(all_words)
                new_tuple = tuple(new_word)
                
                # Skip if it's already in the set
                if any(tuple(c) == new_tuple for c in best_codewords):
                    continue
                
                trial = best_codewords[:i] + [list(new_word)] + best_codewords[i+1:]
                
                covered_mask = np.zeros(len(all_words), dtype=bool)
                for c in trial:
                    idx = word_to_idx.get(tuple(c))
                    if idx is not None:
                        for j in neighbors[idx]:
                            covered_mask[j] = True
                
                if covered_mask.all():
                    best_codewords = trial
                    improved = True
                    break
        
        if not improved:
            # No improvement found in this pass; do a few more random restarts of local search
            # Try perturbation: remove a random codeword and re-add greedily
            if len(best_codewords) > 2:
                i = rng.randrange(len(best_codewords))
                perturbed = best_codewords[:i] + best_codewords[i+1:]
                
                # Greedily add back
                covered_mask = np.zeros(len(all_words), dtype=bool)
                for c in perturbed:
                    idx = word_to_idx.get(tuple(c))
                    if idx is not None:
                        for j in neighbors[idx]:
                            covered_mask[j] = True
                
                while not covered_mask.all():
                    if time.time() > deadline - 10:
                        break
                    
                    best_gain = -1
                    best_idx = -1
                    for k in range(len(all_words)):
                        gain = 0
                        for j in neighbors[k]:
                            if not covered_mask[j]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = k
                    
                    if best_idx < 0 or best_gain <= 0:
                        break
                    
                    perturbed.append([int(x) for x in all_words[best_idx]])
                    for j in neighbors[best_idx]:
                        covered_mask[j] = True
                
                if covered_mask.all() and len(perturbed) < len(best_codewords):
                    best_codewords = perturbed
    
    return best_codewords


def _solve_q3_n6_r1(deadline):
    """Solve the specific case q=3, n=6, radius=1."""
    best_result = None
    
    # Try multiple greedy initializations with different seeds
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        
        codewords = _greedy_cover(3, 6, 1, deadline, seed=seed)
        
        if not codewords:
            continue
        
        # Local search to improve
        improved = _local_search(codewords, 6, 3, deadline, seed=seed)
        
        if best_result is None or len(improved) < len(best_result):
            best_result = improved
    
    # Fallback: tetracode construction
    if best_result is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    return best_result


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # General fallback: tetracode-based or greedy
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return result
    
    # Generic greedy fallback
    codewords = _greedy_cover(q, n, radius, deadline, seed=0)
    if codewords:
        return codewords
    
    # Last resort: all-zero word
    return [[0] * n]
# EVOLVE-BLOCK-END
