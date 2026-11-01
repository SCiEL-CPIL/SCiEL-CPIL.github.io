"""Improved ternary covering code K_3(7,1) via vectorized greedy + local search."""
import time
import numpy as np
from itertools import product

# Deadline guard
_DEADLINE = 150.0  # seconds; evaluator kills at 150s, use 145s safety margin

def _time_left():
    """Return remaining time budget."""
    return _DEADLINE - (time.time() - _START_TIME)

_START_TIME = time.time()

def _ball_mask(code, q, n):
    """
    Return a boolean mask of all words within Hamming distance 1 of `code`.
    For q=3, n=7: 1 + n*(q-1) = 1 + 7*2 = 15 words.
    We encode each word as an integer in [0, q^n) using base-q digits.
    """
    # code is a list/array of n ints
    # Generate all words at distance 0 and 1
    # Distance 0: the code itself
    # Distance 1: change one position to any other value
    words = []
    code = list(code)
    words.append(code)
    for i in range(n):
        for v in range(q):
            if v == code[i]:
                continue
            new_word = code[:i] + [v] + code[i+1:]
            words.append(new_word)
    # Convert to base-q integers
    powers = [q ** (n - 1 - i) for i in range(n)]
    ints = []
    for w in words:
        val = 0
        for j in range(n):
            val = val * q + w[j]
        ints.append(val)
    return np.array(ints, dtype=np.int64)

def _encode(code, q, n):
    """Encode a codeword (list of n ints) to a base-q integer."""
    val = 0
    for c in code:
        val = val * q + c
    return val

def _decode(val, q, n):
    """Decode a base-q integer to a list of n ints."""
    digits = []
    for _ in range(n):
        digits.append(val % q)
        val //= q
    digits.reverse()
    return digits

def _all_ball_masks(codes, q, n):
    """
    Precompute ball masks for a list of codes.
    Returns a list of arrays, each of shape (15,) for q=3,n=7.
    Also returns a flat array of all covered word indices.
    """
    masks = []
    for code in codes:
        masks.append(_ball_mask(code, q, n))
    return masks

def construct(q, n, radius):
    global _START_TIME
    _START_TIME = time.time()
    
    total_words = q ** n
    # For q=3, n=7: total_words = 2187
    
    # Step 1: Greedy construction
    # Maintain a covered array of size total_words
    covered = np.zeros(total_words, dtype=bool)
    
    # Precompute all words as base-q integers
    # and precompute ball membership for each word
    # ball_members[w] = list of words in the ball of radius `radius` around w
    # Actually, we need: for each candidate codeword c, which words does it cover?
    # That's _ball_mask(c, q, n)
    
    # For efficiency, precompute the ball mask for every possible word
    # There are 2187 words, each ball has 15 entries, so 2187*15 = 32805 entries - fine
    all_ball_masks = {}
    for w in range(total_words):
        code = _decode(w, q, n)
        all_ball_masks[w] = _ball_mask(code, q, n)
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    codes = []
    uncovered_count = total_words
    
    # Precompute for each word, the set of words it covers (as a set of ints)
    # Actually we have all_ball_masks[w] as an array of ints
    
    while uncovered_count > 0 and _time_left() > 5.0:
        best_w = -1
        best_gain = 0
        # To speed up, we can sample or use a heuristic
        # For 2187 words, checking all each iteration is 2187 * 15 = ~33k ops per iteration
        # We need ~186 iterations, so ~6M ops total - fast enough
        for w in range(total_words):
            if covered[w]:
                # If the center is already covered, it might still cover new words
                # But typically we prefer uncovered centers
                pass
            # Count how many words in this ball are uncovered
            ball = all_ball_masks[w]
            gain = np.sum(~covered[ball])
            if gain > best_gain:
                best_gain = gain
                best_w = w
            # Early exit if we found a perfect gain
            if best_gain == 15:  # max possible for q=3,n=7,radius=1
                break
        
        if best_w < 0 or best_gain == 0:
            break
        
        codes.append(best_w)
        ball = all_ball_masks[best_w]
        covered[ball] = True
        uncovered_count = int(np.sum(~covered))
    
    # Step 2: Local search - try to reduce the number of codewords
    # Strategy: for each codeword, check if removing it still leaves all words covered
    # If so, remove it. Repeat until no more can be removed.
    # Then try 2-for-1: replace two codewords with one if possible.
    
    codes_set = set(codes)
    
    def _is_valid(codes_list):
        """Check if codes_list covers all words."""
        cov = np.zeros(total_words, dtype=bool)
        for c in codes_list:
            cov[all_ball_masks[c]] = True
        return np.all(cov)
    
    # Phase 2a: Remove redundant codewords
    changed = True
    while changed and _time_left() > 5.0:
        changed = False
        new_codes = []
        for c in codes:
            # Try removing c
            remaining = [x for x in codes if x != c]
            if _is_valid(remaining):
                codes = remaining
                changed = True
                break  # Restart the scan
        if not changed:
            codes = new_codes if new_codes else codes
    
    # Phase 2b: 2-for-1 moves - try to replace two codewords with one
    # For each pair (i,j), check if there exists a codeword k such that
    # ball(k) covers ball(i) ∪ ball(j)
    # This is expensive: O(m^2 * total_words) where m = num codes
    # m ~ 186, so 186^2 * 2187 ~ 76M - too slow for repeated checks
    
    # Better: for each pair, compute the union of their balls, then check if any single ball covers it
    # Union size is at most 29, so we can check quickly
    
    codes_list = list(codes)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        m = len(codes_list)
        for i in range(m):
            if _time_left() < 3.0:
                break
            for j in range(i + 1, m):
                if _time_left() < 3.0:
                    break
                # Union of balls i and j
                union = np.union1d(all_ball_masks[codes_list[i]], all_ball_masks[codes_list[j]])
                # Check if any word's ball covers this union
                # union has at most 29 elements
                # For each candidate k, check if union ⊆ ball(k)
                # ball(k) has 15 elements, so union must be subset of it
                # But union has up to 29 elements, ball has 15, so this can only work if union ≤ 15
                if len(union) > 15:
                    continue
                # Check if there's a k whose ball contains union
                for k in range(total_words):
                    ball_k = all_ball_masks[k]
                    # Check if union ⊆ ball_k
                    # Use set intersection
                    if set(union).issubset(set(ball_k)):
                        # Replace i and j with k
                        new_codes = []
                        for idx, c in enumerate(codes_list):
                            if idx == i or idx == j:
                                continue
                            new_codes.append(c)
                        new_codes.append(k)
                        # Verify validity
                        if _is_valid(new_codes):
                            codes_list = new_codes
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    # Phase 2c: Final polish - try to add a codeword that allows removing two
    # This is the 1-for-2 move
    
    # Convert back to list of lists
    result = [_decode(c, q, n) for c in codes_list]
    
    # Verify validity one more time
    if not _is_valid(codes_list):
        # Fallback to original
        result = [_decode(c, q, n) for c in codes]
    
    return result
