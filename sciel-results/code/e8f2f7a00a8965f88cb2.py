"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _ball_mask(q, n, radius, word):
    """Generate all words within Hamming distance radius of word."""
    if radius == 0:
        yield word
        return
    if radius == 1:
        yield word
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    w = list(word)
                    w[i] = v
                    yield tuple(w)
        return
    # General case for radius > 1 (not expected in this task but for completeness)
    def _generate(pos, current, remaining):
        if remaining == 0:
            yield tuple(current)
            return
        # Option 1: keep current value
        yield from _generate(pos + 1, current, remaining)
        # Option 2: change current value
        orig = current[pos]
        for v in range(q):
            if v != orig:
                current[pos] = v
                yield from _generate(pos + 1, current, remaining - 1)
        current[pos] = orig
    
    current = list(word)
    yield from _generate(0, current, radius)


def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover: repeatedly pick the word that covers the most uncovered words."""
    total = q ** n
    
    # For large spaces, we need a smarter approach
    # Start with a random or structured initial code
    # For this task, we'll use a greedy approach with random restarts
    
    best_code = None
    best_size = total  # Upper bound
    
    # Generate candidate words
    # We'll use a heuristic: start with a random subset and improve
    # But for efficiency, let's use a structured approach
    
    # For q=3, n=6, radius=1, the space is 729
    # We can enumerate all words and use greedy
    
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute balls for each word
    balls = {}
    for w in all_words:
        balls[w] = set(_ball_mask(q, n, radius, w))
    
    # Greedy set cover
    uncovered = set(all_words)
    code = []
    
    while uncovered and time.time() < deadline:
        # Find the word that covers the most uncovered words
        best_w = None
        best_count = -1
        for w in all_words:
            if w in uncovered or True:  # We can pick any word
                count = len(balls[w] & uncovered)
                if count > best_count:
                    best_count = count
                    best_w = w
                if time.time() >= deadline:
                    break
        if time.time() >= deadline:
            break
        if best_w is None or best_count <= 0:
            break
        code.append(list(best_w))
        uncovered -= balls[best_w]
    
    if uncovered:
        # Fallback: add all remaining uncovered words
        for w in uncovered:
            code.append(list(w))
    
    if best_code is None or len(code) < best_size:
        best_code = code
        best_size = len(code)
    
    return best_code


def _local_improve(q, n, radius, code, deadline):
    """Try to remove redundant codewords or swap for better coverage."""
    code = [tuple(w) for w in code]
    code_set = set(code)
    
    # Check if we can remove any codeword
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i, w in enumerate(list(code)):
            # Check if removing w still covers everything
            new_code_set = code_set - {w}
            # Check if all words in ball(w) are covered by other codewords
            for v in _ball_mask(q, n, radius, w):
                if v not in new_code_set and not any(
                    sum(1 for a, b in zip(v, cw) if a != b) <= radius
                    for cw in new_code_set
                ):
                    break
            else:
                # w is redundant
                code.pop(i)
                code_set.remove(w)
                changed = True
                break
    
    return [list(w) for w in code]


def construct(q, n, radius):
    deadline = time.time() + 120 - 5  # Safety margin
    
    # Special case: known optimal for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Try to find a good code using greedy + local search
        # Start with the tetracode-based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Try greedy from scratch
        greedy_code = _greedy_cover(q, n, radius, deadline)
        
        # Try local improvement on initial
        improved_initial = _local_improve(q, n, radius, initial, deadline)
        
        # Pick the best
        candidates = [initial, greedy_code, improved_initial]
        best = min(candidates, key=len)
        
        # Try more local search
        if time.time() < deadline:
            best = _local_improve(q, n, radius, best, deadline)
        
        return best
    
    # General case
    # Use greedy cover
    return _greedy_cover(q, n, radius, deadline)
# EVOLVE-BLOCK-END
