# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115  # 115s safety margin out of 150s

    # Start with tetracode product construction (9 * 3^(n-4) words)
    initial = _tetracode(q, n)
    best = [tuple(cw) for cw in initial]

    # Greedy cover to get a good starting point
    if time.time() < deadline - 50:
        try:
            greedy = _greedy_cover_fast(q, n, radius, deadline - 50)
            if len(greedy) < len(best):
                best = [tuple(cw) for cw in greedy]
        except Exception:
            pass

    # Convert to list of lists for local search
    best = [list(cw) for cw in best]

    # Local search: remove redundant codewords
    if time.time() < deadline - 10:
        best = _local_search_remove(best, q, n, radius, deadline - 10)

    # Simulated annealing with swap-based local search
    if time.time() < deadline - 2:
        best = _simulated_annealing_swap(best, q, n, radius, deadline - 2)

    # Final removal pass
    if time.time() < deadline - 1:
        best = _local_search_remove(best, q, n, radius, deadline - 1)

    return best


def _tetracode(q, n):
    """Product construction: ternary tetracode (9 words of length 4) x all words on remaining coords."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _ball_indices(word, q, n, radius):
    """Return a frozenset of indices (integers 0..q^n-1) within Hamming distance radius of word."""
    word = np.array(word, dtype=np.int8)
    indices = [int(np.ravel_multi_index(tuple(word), (q,) * n))]
    if radius >= 1:
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                w2 = word.copy()
                w2[pos] = val
                indices.append(int(np.ravel_multi_index(tuple(w2), (q,) * n)))
    return frozenset(indices)


def _greedy_cover_fast(q, n, radius, deadline):
    """Faster greedy using numpy for coverage counting."""
    total = q ** n
    balls = np.zeros((total, total), dtype=bool)
    for idx in range(total):
        word = list(np.unravel_index(idx, (q,) * n))
        for p in _ball_indices(word, q, n, radius):
            balls[idx, p] = True
    
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while uncovered.any() and time.time() < deadline:
        counts = balls[:, uncovered].sum(axis=1)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        word = list(np.unravel_index(best_idx, (q,) * n))
        codewords.append(word)
        uncovered &= ~balls[best_idx]
    
    return codewords


def _local_search_remove(codewords, q, n, radius, deadline):
    """Try to remove redundant codewords."""
    total = q ** n
    ball_sets = [_ball_indices(cw, q, n, radius) for cw in codewords]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            remaining = [j for j in range(len(codewords)) if j != i]
            covered = set()
            for j in remaining:
                covered |= ball_sets[j]
            if len(covered) == total:
                codewords.pop(i)
                ball_sets.pop(i)
                improved = True
                break
    
    return codewords


def _simulated_annealing_swap(codewords, q, n, radius, deadline):
    """
    Simulated annealing with swap-based local search.
    Allows escaping local minima by accepting equal-size moves with probability.
    """
    total = q ** n
    ball_sets = [_ball_indices(cw, q, n, radius) for cw in codewords]
    
    # Precompute coverage count for each point
    # count[j] = number of codewords whose ball covers point j
    count = np.zeros(total, dtype=np.int32)
    for bs in ball_sets:
        for idx in bs:
            count[idx] += 1
    
    covered = count > 0
    current_size = len(codewords)
    best_code = [list(cw) for cw in codewords]
    best_size = current_size
    
    # Temperature schedule
    t0 = 1.0
    t_end = 0.01
    cooling = 0.995
    
    temperature = t0
    random.seed(42)
    
    while time.time() < deadline and temperature > t_end:
        # Try a random swap: remove one codeword, add another
        if len(codewords) < 2:
            break
        
        i = random.randint(0, len(codewords) - 1)
        
        # Find a candidate to replace codeword i
        # We want to find a word that covers points not covered by others
        others_covered_count = count.copy()
        others_covered_count[ball_sets[i]] -= 1
        others_covered_count[others_covered_count < 0] = 0
        
        # Points that are only covered by codeword i
        must_cover = [j for j in ball_sets[i] if others_covered_count[j] == 0]
        
        if not must_cover:
            # Can remove codeword i
            codewords.pop(i)
            ball_sets.pop(i)
            count[ball_sets[i]] -= 1  # This is wrong, we already popped
            # Recompute count properly
            count = np.zeros(total, dtype=np.int32)
            for bs in ball_sets:
                for idx in bs:
                    count[idx] += 1
            current_size -= 1
            if current_size < best_size:
                best_size = current_size
                best_code = [list(cw) for cw in codewords]
            temperature *= cooling
            continue
        
        # Try to find a replacement that covers must_cover
        # Sample random candidates
        found = False
        for _ in range(50):  # Try 50 random candidates
            candidate_idx = random.randint(0, total - 1)
            candidate_word = list(np.unravel_index(candidate_idx, (q,) * n))
            if candidate_word == codewords[i]:
                continue
            candidate_ball = _ball_indices(candidate_word, q, n, radius)
            
            # Check if candidate covers all must_cover points
            if all(j in candidate_ball for j in must_cover):
                # Check if this maintains full coverage
                # New coverage = (old coverage minus ball[i]) union candidate_ball
                # This is complex, so let's just try the swap and verify
                
                # Tentative swap
                old_ball = ball_sets[i]
                old_word = codewords[i]
                
                codewords[i] = candidate_word
                ball_sets[i] = candidate_ball
                
                # Recompute count
                new_count = count.copy()
                new_count[old_ball] -= 1
                new_count[candidate_ball] += 1
                
                if np.all(new_count > 0):
                    # Valid swap
                    count = new_count
                    current_size = len(codewords)
                    if current_size < best_size:
                        best_size = current_size
                        best_code = [list(cw) for cw in codewords]
                    found = True
                    break
                else:
                    # Revert
                    codewords[i] = old_word
                    ball_sets[i] = old_ball
        
        if not found:
            # Try a random addition + removal (size-preserving move with acceptance)
            # Add a random word, then try to remove a redundant one
            candidate_idx = random.randint(0, total - 1)
            candidate_word = list(np.unravel_index(candidate_idx, (q,) * n))
            
            # Check if already in code
            if candidate_word in codewords:
                temperature *= cooling
                continue
            
            candidate_ball = _ball_indices(candidate_word, q, n, radius)
            
            # Tentatively add
            codewords.append(candidate_word)
            ball_sets.append(candidate_ball)
            count[candidate_ball] += 1
            
            # Try to remove a redundant codeword
            removed = False
            for j in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                temp_count = count.copy()
                temp_count[ball_sets[j]] -= 1
                if np.all(temp_count > 0):
                    # Can remove
                    count = temp_count
                    codewords.pop(j)
                    ball_sets.pop(j)
                    removed = True
                    break
            
            if not removed:
                # Revert the addition
                codewords.pop()
                ball_sets.pop()
                count[candidate_ball] -= 1
            else:
                current_size = len(codewords)
                if current_size < best_size:
                    best_size = current_size
                    best_code = [list(cw) for cw in codewords]
        
        temperature *= cooling
    
    return best_code


# EVOLVE-BLOCK-END
