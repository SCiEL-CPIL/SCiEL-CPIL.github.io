"""
Structural change: exploit the algebraic structure of the ternary space F_3^8
via a direct-sum construction (tetracode x trivial) to get a highly symmetric
starting point, then refine with a vectorized greedy set-cover that uses
precomputed neighborhood bitsets, followed by aggressive redundancy removal
and a targeted swap-based local search using an inverse coverage index.
"""
import time
import random
import itertools
import numpy as np

_DEADLINE = 112.0


def _hamming_dist(a, b):
    """Hamming distance between two words (lists of ints)."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _direct_sum_code(q, n, radius):
    """
    Build a covering code using direct sum of a small base code.
    For q=3, radius=1, n=8: use the ternary tetracode (9 words, length 4, radius 1)
    as the base, and take direct sum with {0,1,2}^{n-4}.
    
    This gives 9 * 3^4 = 9 * 81 = 729 words initially.
    """
    if q == 3 and radius == 1 and n >= 4:
        # Ternary tetracode: 9 words of length 4 covering all 3^4=81 words with radius 1
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Direct sum with {0,1,2}^{n-4}
        code = []
        for tail in itertools.product(range(3), repeat=n - 4):
            for h in tetracode:
                code.append(h + list(tail))
        return code
    else:
        return None


def _precompute_neighbors(q, n, radius):
    """
    Precompute the neighborhood (ball of given radius) for every word in the space.
    Returns a list of lists, where neighbors[i] is the list of indices within
    Hamming distance 'radius' of word i.
    """
    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    neighbors = [None] * total_words
    
    if radius == 1:
        for w in range(total_words):
            word = idx_to_word(w)
            neigh = [w]
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_idx(word))
                word[pos] = cur
            neighbors[w] = neigh
    else:
        # General case for radius >= 1
        for w in range(total_words):
            word = idx_to_word(w)
            neigh = [w]
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_idx(word))
                    word[pos] = cur
            if radius >= 2:
                word = idx_to_word(w)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_idx(word))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = list(set(neigh))
    
    return neighbors


def _greedy_complete(code, q, n, radius, neighbors, deadline):
    """
    Greedy set-cover completion: add codewords to ensure full coverage.
    Uses precomputed neighbors for efficiency.
    """
    total_words = q ** n
    covered = np.zeros(total_words, dtype=np.uint8)
    
    for cw in code:
        idx = 0
        for v in cw:
            idx = idx * q + v
        for u in neighbors[idx]:
            covered[u] = 1
    
    remaining = total_words - int(np.sum(covered))
    
    if remaining <= 0:
        return code
    
    codewords = list(code)
    
    # Track uncovered words for sampling
    uncovered_set = set(np.nonzero(covered == 0)[0].tolist())
    
    while uncovered_set:
        if time.perf_counter() > deadline:
            break
        
        # Sample from uncovered to find good candidates
        k = min(len(uncovered_set), 64)
        samp = random.sample(list(uncovered_set), k)
        
        # Collect candidate codewords: neighbors of sampled uncovered words
        candidates = set()
        for u in samp:
            candidates.update(neighbors[u])
        
        best_cand = None
        best_gain = -1
        
        for w in candidates:
            gain = 0
            for u in neighbors[w]:
                if covered[u] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cand = w
                if gain == len(uncovered_set):
                    break
        
        if best_cand is None:
            break
        
        # Add best candidate
        cw_idx = best_cand
        for u in neighbors[cw_idx]:
            if covered[u] == 0:
                covered[u] = 1
                uncovered_set.discard(u)
        
        # Convert index to word
        word = [0] * n
        tmp = cw_idx
        for i in range(n - 1, -1, -1):
            word[i] = tmp % q
            tmp //= q
        codewords.append(word)
    
    return codewords


def _remove_redundant(code, q, n, radius, neighbors, deadline):
    """
    Remove codewords that are redundant (their removal doesn't uncover any word).
    Uses coverage counting for efficiency.
    """
    total_words = q ** n
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    for cw in code:
        idx = word_to_idx(cw)
        for u in neighbors[idx]:
            coverage_count[u] += 1
    
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        # Shuffle to avoid bias
        indices = list(range(len(code)))
        random.shuffle(indices)
        for i in indices:
            if time.perf_counter() > deadline:
                break
            cw = code[i]
            idx = word_to_idx(cw)
            # Check if any word is only covered by this codeword
            can_remove = True
            for u in neighbors[idx]:
                if coverage_count[u] == 1:
                    can_remove = False
                    break
            if can_remove:
                for u in neighbors[idx]:
                    coverage_count[u] -= 1
                code.pop(i)
                improved = True
                break
    
    return code


def _swap_local_search(code, q, n, radius, neighbors, deadline):
    """
    Local search with swaps: try to replace one codeword with another to
    maintain coverage while potentially enabling further removals.
    """
    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw in code:
        idx = word_to_idx(cw)
        for u in neighbors[idx]:
            coverage_count[u] += 1
    
    start_time = time.perf_counter()
    
    while time.perf_counter() - start_time < deadline:
        if not code:
            break
        
        # Try to remove redundant codewords first
        removed = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        for i in indices:
            if time.perf_counter() > deadline:
                break
            cw = code[i]
            idx = word_to_idx(cw)
            can_remove = True
            for u in neighbors[idx]:
                if coverage_count[u] == 1:
                    can_remove = False
                    break
            if can_remove:
                for u in neighbors[idx]:
                    coverage_count[u] -= 1
                code.pop(i)
                removed = True
                break
        
        if removed:
            continue
        
        # Try a swap: remove one codeword and add another
        idx_to_try = random.randrange(len(code))
        old_cw = code[idx_to_try]
        old_idx = word_to_idx(old_cw)
        
        # Words that would be uncovered
        would_uncover = [u for u in neighbors[old_idx] if coverage_count[u] == 1]
        
        if not would_uncover:
            # Can just remove
            for u in neighbors[old_idx]:
                coverage_count[u] -= 1
            code.pop(idx_to_try)
            continue
        
        # Need to find a replacement that covers all would_uncover words
        # Candidate pool: intersection of neighbors of all would_uncover words
        if len(would_uncover) > 5:
            # Too many constraints, skip this swap
            continue
        
        first_uncovered = would_uncover[0]
        candidates = list(neighbors[first_uncovered])
        
        # Filter candidates that cover all would_uncover words
        valid_candidates = []
        for cand in candidates:
            if cand == old_idx:
                continue
            covers_all = True
            for u in would_uncover:
                if u not in neighbors[cand]:
                    covers_all = False
                    break
            if covers_all:
                # Check if this candidate adds new coverage
                new_gain = 0
                for u in neighbors[cand]:
                    if coverage_count[u] == 0:
                        new_gain += 1
                if new_gain > 0:
                    valid_candidates.append((cand, new_gain))
        
        if valid_candidates:
            # Pick the one with max new coverage
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            new_idx, _ = valid_candidates[0]
            
            # Perform the swap
            for u in neighbors[old_idx]:
                coverage_count[u] -= 1
            code[idx_to_try] = idx_to_word(new_idx)
            for u in neighbors[new_idx]:
                coverage_count[u] += 1
    
    return code


def construct(q, n, radius):
    t0 = time.perf_counter()
    deadline = t0 + _DEADLINE
    
    total_words = q ** n
    
    # Precompute neighbors
    neighbors = _precompute_neighbors(q, n, radius)
    
    # Try direct sum construction first
    code = _direct_sum_code(q, n, radius)
    
    if code is None or len(code) == 0:
        # Fallback: start with a single word and greedily complete
        code = [[0] * n]
    
    # Greedy complete to ensure full coverage
    code = _greedy_complete(code, q, n, radius, neighbors, deadline)
    
    # Remove redundant codewords
    code = _remove_redundant(code, q, n, radius, neighbors, deadline)
    
    # Swap-based local search
    if time.perf_counter() < deadline:
        code = _swap_local_search(code, q, n, radius, neighbors, deadline)
    
    # Final redundancy removal pass
    if time.perf_counter() < deadline:
        code = _remove_redundant(code, q, n, radius, neighbors, deadline)
    
    # Ensure distinct codewords
    seen = set()
    unique_code = []
    for cw in code:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_code.append(cw)
    
    return unique_code
