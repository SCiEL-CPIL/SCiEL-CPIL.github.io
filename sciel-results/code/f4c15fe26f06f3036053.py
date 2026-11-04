"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0


def _word_to_idx(word, q):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _tetracode_solution(q, n, radius):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _build_balls(q, n):
    """Precompute balls (distance-1 neighborhoods) for all words as list of index arrays."""
    space_size = q ** n
    balls = []
    for i in range(space_size):
        word = _idx_to_word(i, q, n)
        ball = [i]
        for pos in range(n):
            orig = word[pos]
            for d in range(q):
                if d == orig:
                    continue
                word[pos] = d
                ball.append(_word_to_idx(word, q))
            word[pos] = orig
        balls.append(np.array(ball, dtype=np.int32))
    return balls


def _prune_until_stable(active_set, counts, balls, q, n, time_limit):
    """Iteratively remove redundant codewords until no more can be removed."""
    removed_any = True
    while removed_any:
        if time.time() - _start_time > time_limit - 5:
            break
        removed_any = False
        to_check = list(active_set)
        random.shuffle(to_check)
        for idx in to_check:
            if idx not in active_set:
                continue
            # Check if all words in this ball have count > 1
            if np.min(counts[balls[idx]]) > 1:
                for b in balls[idx]:
                    counts[b] -= 1
                active_set.remove(idx)
                removed_any = True
    return active_set, counts


def _try_swaps(active_set, counts, balls, q, n, time_limit, max_swaps=200):
    """Try 1-opt swaps: remove one codeword, add another to maintain coverage."""
    swaps_done = 0
    for _ in range(max_swaps):
        if time.time() - _start_time > time_limit - 5:
            break
        if not active_set:
            break
        
        current = random.choice(list(active_set))
        curr_word = _idx_to_word(current, q, n)
        
        # Generate candidates: neighbors of current (distance 1)
        candidates = []
        for pos in range(n):
            orig = curr_word[pos]
            for d in range(q):
                if d == orig:
                    continue
                curr_word[pos] = d
                candidates.append(_word_to_idx(curr_word, q))
            curr_word[pos] = orig
        
        random.shuffle(candidates)
        
        swapped = False
        for cand in candidates[:8]:
            if cand in active_set:
                continue
            
            # Simulate: remove current, add cand
            for b in balls[current]:
                counts[b] -= 1
            
            uncovered = np.argwhere(counts == 0).flatten()
            
            if len(uncovered) == 0:
                # Free swap
                for b in balls[cand]:
                    counts[b] += 1
                active_set.remove(current)
                active_set.add(cand)
                swapped = True
                swaps_done += 1
                break
            else:
                # Check if cand covers all uncovered
                cand_ball_set = set(balls[cand].tolist())
                if all(u in cand_ball_set for u in uncovered):
                    for b in balls[cand]:
                        counts[b] += 1
                    active_set.remove(current)
                    active_set.add(cand)
                    swapped = True
                    swaps_done += 1
                    break
                else:
                    # Revert
                    for b in balls[current]:
                        counts[b] += 1
        
        if swapped:
            # After swap, try to prune again
            active_set, counts = _prune_until_stable(active_set, counts, balls, q, n, time_limit)
        else:
            # No swap found, break to avoid infinite loop
            if swaps_done == 0:
                break
    
    return active_set, counts, swaps_done


def _try_double_swaps(active_set, counts, balls, q, n, time_limit, max_attempts=50):
    """Try 2-opt: remove two codewords, add one that covers the gap."""
    swaps_done = 0
    for _ in range(max_attempts):
        if time.time() - _start_time > time_limit - 5:
            break
        if len(active_set) < 2:
            break
        
        # Pick two random active codewords
        sample = random.sample(list(active_set), 2)
        a, b = sample
        
        # Simulate removing both
        for x in balls[a]:
            counts[x] -= 1
        for x in balls[b]:
            counts[x] -= 1
        
        uncovered = np.argwhere(counts == 0).flatten()
        
        if len(uncovered) == 0:
            # Both were redundant (shouldn't happen after pruning, but just in case)
            for x in balls[a]:
                counts[x] += 1
            for x in balls[b]:
                counts[x] += 1
            continue
        
        # Find a candidate that covers all uncovered words
        # Try neighbors of a and b
        candidates = set()
        for src in [a, b]:
            w = _idx_to_word(src, q, n)
            for pos in range(n):
                orig = w[pos]
                for d in range(q):
                    if d == orig:
                        continue
                    w[pos] = d
                    candidates.add(_word_to_idx(w, q))
                w[pos] = orig
        
        candidates.discard(a)
        candidates.discard(b)
        candidates = [c for c in candidates if c not in active_set]
        random.shuffle(candidates)
        
        found = False
        for cand in candidates[:15]:
            cand_ball_set = set(balls[cand].tolist())
            if all(u in cand_ball_set for u in uncovered):
                # Accept: add cand
                for x in balls[cand]:
                    counts[x] += 1
                active_set.remove(a)
                active_set.remove(b)
                active_set.add(cand)
                found = True
                swaps_done += 1
                break
        
        if not found:
            # Revert
            for x in balls[a]:
                counts[x] += 1
            for x in balls[b]:
                counts[x] += 1
    
    return active_set, counts, swaps_done


def _local_search(q, n, radius, initial_words, time_limit):
    space_size = q ** n
    balls = _build_balls(q, n)
    
    active_set = set()
    for w in initial_words:
        active_set.add(_word_to_idx(w, q))
    
    counts = np.zeros(space_size, dtype=np.int32)
    for idx in active_set:
        for b in balls[idx]:
            counts[b] += 1
    
    # Phase 1: Initial pruning
    active_set, counts = _prune_until_stable(active_set, counts, balls, q, n, time_limit)
    
    # Phase 2: Swaps
    active_set, counts, _ = _try_swaps(active_set, counts, balls, q, n, time_limit, max_swaps=300)
    
    # Phase 3: Double swaps
    active_set, counts, _ = _try_double_swaps(active_set, counts, balls, q, n, time_limit, max_attempts=100)
    
    # Phase 4: More pruning after swaps
    active_set, counts = _prune_until_stable(active_set, counts, balls, q, n, time_limit)
    
    # Phase 5: Another round of swaps
    active_set, counts, _ = _try_swaps(active_set, counts, balls, q, n, time_limit, max_swaps=200)
    active_set, counts = _prune_until_stable(active_set, counts, balls, q, n, time_limit)
    
    # Convert back
    result = [_idx_to_word(idx, q, n) for idx in active_set]
    return result


def _verify_coverage(q, n, radius, words):
    space_size = q ** n
    covered = np.zeros(space_size, dtype=bool)
    for cw in words:
        idx = _word_to_idx(cw, q)
        covered[idx] = True
        w = _idx_to_word(idx, q, n)
        for pos in range(n):
            orig = w[pos]
            for d in range(q):
                if d == orig:
                    continue
                w[pos] = d
                covered[_word_to_idx(w, q)] = True
            w[pos] = orig
    return np.all(covered)


def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 8  # 8s safety margin

    if q != 3 or n != 8 or radius != 1:
        return _tetracode_solution(q, n, radius)

    seed_words = _tetracode_solution(q, n, radius)
    
    # Run local search with multiple restarts
    best_result = seed_words
    best_size = len(seed_words)
    
    num_restarts = 3
    per_restart_time = (time_limit - 10) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - _start_time > time_limit - 5:
            break
        
        # Add random perturbation for restarts > 0
        if r == 0:
            init = seed_words
        else:
            # Randomly swap some words to create diversity
            init = list(seed_words)
            num_swaps = random.randint(20, 50)
            for _ in range(num_swaps):
                i = random.randint(0, len(init) - 1)
                w = init[i]
                pos = random.randint(0, n - 1)
                old = w[pos]
                w[pos] = random.choice([d for d in range(q) if d != old])
                # Check if this creates a duplicate
                if w in init:
                    w[pos] = old
                else:
                    init[i] = w
        
        try:
            result = _local_search(q, n, radius, init, time_limit)
            if result and _verify_coverage(q, n, radius, result):
                if len(result) < best_size:
                    best_size = len(result)
                    best_result = result
        except Exception:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
