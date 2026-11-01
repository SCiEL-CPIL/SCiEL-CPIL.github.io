import time
import itertools
import random

_DEADLINE = 150.0
_MARGIN = 1.5


def _all_words(q, n):
    """Yield all q^n words as tuples of ints in 0..q-1."""
    return itertools.product(range(q), repeat=n)


def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists of same length."""
    return sum(x != y for x, y in zip(a, b))


def _neighbors(word, q):
    """Yield all words at Hamming distance exactly 1 from word."""
    n = len(word)
    w = list(word)
    for i in range(n):
        orig = w[i]
        for v in range(q):
            if v != orig:
                w[i] = v
                yield tuple(w)
        w[i] = orig


def _greedy_cover(q, n, radius, deadline):
    """
    Greedy set-cover: repeatedly pick the word that covers the most uncovered words.
    We track covered words in a set. Each codeword covers itself + all words at dist <= radius.
    For efficiency, we maintain the set of uncovered words and for each candidate codeword,
    count how many uncovered words are within distance <= radius.
    """
    all_words = list(_all_words(q, n))
    index = {w: i for i, w in enumerate(all_words)}
    total = len(all_words)
    
    # Precompute: for each word, the set of words within Hamming distance <= radius
    # This is expensive for large spaces, so we do it lazily or use a different approach.
    
    # For q=3, n=6, radius=1: total = 729 words. Each word has 1 + 6*2 = 13 neighbors within dist 1.
    # We can precompute the ball for each word.
    
    balls = {}
    for w in all_words:
        ball = set()
        ball.add(w)
        for nb in _neighbors(w, q):
            if _hamming_dist(w, nb) <= radius:
                ball.add(nb)
        # For radius=1, the ball is just w + its 6*(q-1) neighbors
        # Let's be more precise:
        if radius == 1:
            ball = {w}
            for nb in _neighbors(w, q):
                ball.add(nb)
        balls[w] = ball
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        time_start = time.time()
        best_word = None
        best_count = 0
        for w in all_words:
            if w in uncovered:
                cnt = len(balls[w] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
        if best_word is None:
            break
        code.append(list(best_word))
        uncovered -= balls[best_word]
        if time.time() - time_start > _MARGIN:
            break
        if time.time() > _DEADLINE - _MARGIN:
            break
    
    return code


def _local_search(code, q, n, radius, deadline):
    """
    Try to reduce the code size by removing codewords or replacing pairs.
    A codeword can be removed if all words it covers are also covered by other codewords.
    """
    code = [tuple(c) for c in code]
    if not code:
        return code
    
    # Build coverage: for each word in the space, which codewords cover it?
    # For efficiency, track which codewords are needed for each word.
    all_words = list(_all_words(q, n))
    
    def is_covered(word, code_list):
        """Check if word is within distance radius of any codeword in code_list."""
        for cw in code_list:
            if _hamming_dist(word, cw) <= radius:
                return True
        return False
    
    improved = True
    while improved and time.time() < _DEADLINE - _MARGIN:
        improved = False
        # Try removing each codeword
        for i in range(len(code) - 1, -1, -1):
            if time.time() > _DEADLINE - _MARGIN:
                return [list(c) for c in code]
            candidate = code[:i] + code[i+1:]
            # Check if all words are still covered
            all_covered = True
            for w in all_words:
                if not is_covered(w, candidate):
                    all_covered = False
                    break
            if all_covered:
                code = candidate
                improved = True
                break
    
    # Try replacing one codeword with another that covers more
    # (This is more complex; skip for now)
    
    return [list(c) for c in code]


def construct(q, n, radius):
    """
    Construct a covering code for the space {0..q-1}^n with covering radius.
    Uses greedy set-cover followed by local search to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    # For the specific case q=3, n=6, radius=1, use a more targeted approach
    # The known best is 73 words (van Laarhoven et al., 1989)
    
    # Start with a reasonable initial code
    # Direct product of tetracode: 9 * 3^(n-4)
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: greedy from scratch
        initial_code = _greedy_cover(q, n, radius, deadline)
    
    # Verify the initial code covers everything
    all_words = list(_all_words(q, n))
    
    def is_valid(code):
        for w in all_words:
            if not any(_hamming_dist(w, c) <= radius for c in code):
                return False
        return True
    
    if not is_valid(initial_code):
        # If invalid, use greedy cover
        initial_code = _greedy_cover(q, n, radius, deadline)
    
    # Local search to reduce size
    if time.time() < deadline:
        improved_code = _local_search(initial_code, q, n, radius, deadline)
        if is_valid(improved_code):
            initial_code = improved_code
    
    return initial_code
