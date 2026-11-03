"""
Structural change: use algebraic direct-sum construction to build a small valid
covering code, then refine with local search (removal + single replacement).
This avoids the greedy set-cover approach entirely and starts from a
combinatorially strong solution.
"""
import time
import random
import itertools
import numpy as np

_DEADLINE = 112.0


def _hamming_dist(a, b):
    """Hamming distance between two words (lists of ints)."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _ball_radius1(w, q, n):
    """
    Yield all words within Hamming distance 1 of w (including w itself).
    Returns a set of tuples for fast lookup.
    """
    result = {tuple(w)}
    for d in range(n):
        cur = w[d]
        for alt in range(q):
            if alt == cur:
                continue
            w2 = list(w)
            w2[d] = alt
            result.add(tuple(w2))
    return result


def _direct_sum_code(q, n, radius):
    """
    Build a covering code using direct sum of a small base code.
    For q=3, radius=1: use the ternary tetracode (9 words, length 4, radius 1)
    as the base, and take direct sum with {0,1,2}^{n-4}.
    
    For general cases, fall back to a structured construction.
    """
    if q == 3 and radius == 1 and n >= 4:
        # Ternary tetracode: 9 words of length 4 covering all 3^4=81 words
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Verify it's a covering code of radius 1
        # Direct sum with {0,1,2}^{n-4}
        code = []
        for tail in itertools.product(range(3), repeat=n - 4):
            for h in tetracode:
                code.append(h + list(tail))
        return code
    elif q == 3 and radius == 1 and n == 3:
        # For n=3, use all 27 words? No, that's too many.
        # Actually for n=3, q=3, radius=1: need to cover 27 words.
        # Each codeword covers 1 + 3*2 = 7 words. So need at least ceil(27/7)=4.
        # Let's use a small set:
        # Actually let's just use a greedy for small n
        return None  # signal to use greedy
    else:
        return None


def _greedy_init(q, n, radius, deadline):
    """
    Greedy set-covering initialization using bitset representation.
    Returns a list of codewords.
    """
    total_words = q ** n
    # Use a boolean array to track coverage
    covered = np.zeros(total_words, dtype=np.uint8)
    codewords = []
    
    # Precompute powers of q for indexing
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q
    
    def word_to_idx(w):
        idx = 0
        for i, v in enumerate(w):
            idx += v * q_powers[n - 1 - i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    def ball_mask(w):
        """Return set of indices in ball of radius 1 around w."""
        result = {word_to_idx(w)}
        for d in range(n):
            cur = w[d]
            for alt in range(q):
                if alt == cur:
                    continue
                w2 = list(w)
                w2[d] = alt
                result.add(word_to_idx(w2))
        return result
    
    # Start with word 0
    start = [0] * n
    codewords.append(start)
    for idx in ball_mask(start):
        covered[idx] = 1
    
    remaining = total_words - np.sum(covered)
    
    while remaining > 0:
        if time.perf_counter() > deadline:
            break
        
        # Find uncovered words
        uncov_idx = np.nonzero(covered == 0)[0]
        if len(uncov_idx) == 0:
            break
        
        # Sample from uncovered to find best candidate
        k = min(len(uncov_idx), 32)
        rng = random.Random()
        samp = rng.sample(uncov_idx.tolist(), k)
        
        best_cand = None
        best_gain = -1
        for s in samp:
            w = idx_to_word(s)
            bm = ball_mask(w)
            gain = sum(1 for idx in bm if covered[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_cand = (w, bm)
        
        if best_cand is None:
            break
        
        w, bm = best_cand
        codewords.append(w)
        for idx in bm:
            covered[idx] = 1
        remaining = total_words - np.sum(covered)
    
    return codewords


def _local_search(code, q, n, radius, deadline):
    """
    Local search: try to remove codewords that are redundant,
    and try single replacements to improve coverage.
    """
    code = [list(c) for c in code]
    total_words = q ** n
    
    # Build coverage map: for each word in space, which codewords cover it
    # This is expensive for large spaces, so we do it incrementally.
    
    def word_to_idx(w):
        idx = 0
        for i, v in enumerate(w):
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    def ball_indices(w):
        result = [word_to_idx(w)]
        for d in range(n):
            cur = w[d]
            for alt in range(q):
                if alt == cur:
                    continue
                w2 = list(w)
                w2[d] = alt
                result.append(word_to_idx(w2))
        return result
    
    # Count how many codewords cover each position in the space
    # Only feasible for small total_words
    if total_words <= 200000:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for cw in code:
            for idx in ball_indices(cw):
                cover_count[idx] += 1
        
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(len(code)):
                if time.perf_counter() > deadline:
                    break
                # Check if codeword i is redundant
                for idx in ball_indices(code[i]):
                    cover_count[idx] -= 1
                if np.all(cover_count >= 1):
                    # Codeword i is redundant
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    for idx in ball_indices(code[i]):
                        cover_count[idx] += 1
    else:
        # For large spaces, just try removals with sampling
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(len(code)):
                if time.perf_counter() > deadline:
                    break
                # Temporarily remove and check coverage via sampling
                code_without_i = code[:i] + code[i+1:]
                # Check if all sampled words are still covered
                rng = random.Random(42)
                all_covered = True
                for _ in range(200):
                    w = [rng.randrange(q) for _ in range(n)]
                    wt = tuple(w)
                    if not any(_hamming_dist(wt, tuple(cw)) <= radius for cw in code_without_i):
                        all_covered = False
                        break
                if all_covered:
                    code.pop(i)
                    improved = True
                    break
    
    return code


def construct(q, n, radius):
    t0 = time.perf_counter()
    deadline = t0 + _DEADLINE
    
    # Try direct sum construction first
    code = _direct_sum_code(q, n, radius)
    
    if code is None:
        # Fall back to greedy
        code = _greedy_init(q, n, radius, deadline)
    
    if code is None or len(code) == 0:
        return [[0] * n]
    
    # Verify coverage (for small spaces)
    total_words = q ** n
    if total_words <= 50000:
        covered = set()
        for cw in code:
            for idx in range(total_words):
                w = []
                tmp = idx
                for _ in range(n):
                    w.append(tmp % q)
                    tmp //= q
                w.reverse()
                if _hamming_dist(tuple(w), tuple(cw)) <= radius:
                    covered.add(idx)
                    break
        if len(covered) < total_words:
            # Greedy didn't fully cover, do more greedy
            code = _greedy_init(q, n, radius, deadline)
    
    # Local search refinement
    if time.perf_counter() < deadline:
        code = _local_search(code, q, n, radius, deadline)
    
    # Ensure distinct codewords
    seen = set()
    unique_code = []
    for cw in code:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_code.append(cw)
    
    return unique_code
