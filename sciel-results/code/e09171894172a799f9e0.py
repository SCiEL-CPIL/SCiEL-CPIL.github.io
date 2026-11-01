"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import deque


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius radius.
    Score is the number of codewords (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin

    # Special case: n=6, q=3, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Try to find a good covering code using a greedy approach with local search
        # Start with the tetracode-based construction (81 words) and try to improve
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
        
        # Try greedy removal: remove codewords that are still covered by others
        code = _try_greedy_removal(initial_code, q, n, radius, deadline)
        
        # Try local search to add better codewords
        code = _local_search(code, q, n, radius, deadline)
        
        return code
    
    # General case: use greedy set cover
    return _greedy_set_cover(q, n, radius, deadline)


def _hamming_dist(a, b):
    """Compute Hamming distance between two lists."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _ball(word, q, n, radius):
    """Generate all words within Hamming distance radius of word."""
    if radius == 0:
        yield list(word)
        return
    if radius == 1:
        yield list(word)
        for i in range(n):
            orig = word[i]
            for v in range(q):
                if v != orig:
                    new_word = list(word)
                    new_word[i] = v
                    yield new_word
        return
    # For general radius, use recursive generation
    def _gen(pos, remaining, current):
        if remaining == 0:
            yield list(current)
            return
        if pos == n:
            return
        # Don't change position pos
        current[pos] = word[pos]
        yield from _gen(pos + 1, remaining, current)
        # Change position pos
        for v in range(q):
            if v != word[pos]:
                current[pos] = v
                yield from _gen(pos + 1, remaining - 1, current)
    
    current = [0] * n
    yield from _gen(0, radius, current)


def _build_coverage_map(code, q, n, radius):
    """Build a map from each word to the set of codeword indices that cover it."""
    # For efficiency, use a dictionary mapping word tuple to count
    coverage = {}
    for idx, cw in enumerate(code):
        for w in _ball(cw, q, n, radius):
            key = tuple(w)
            coverage[key] = coverage.get(key, 0) + 1
    return coverage


def _try_greedy_removal(code, q, n, radius, deadline):
    """Try to remove codewords that are redundant (their ball is covered by others)."""
    code = list(code)
    if not code:
        return code
    
    # Build coverage count
    coverage = _build_coverage_map(code, q, n, radius)
    
    # Try to remove each codeword if possible
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for i in range(len(code)):
            if time.time() > deadline - 20:
                break
            # Check if removing code[i] still covers everything
            # A word is still covered if it's covered by at least one other codeword
            # i.e., coverage[word] > 1 for all words in ball(code[i])
            removable = True
            for w in _ball(code[i], q, n, radius):
                key = tuple(w)
                if coverage.get(key, 0) <= 1:
                    removable = False
                    break
            if removable:
                # Remove code[i]
                for w in _ball(code[i], q, n, radius):
                    key = tuple(w)
                    coverage[key] -= 1
                code.pop(i)
                changed = True
                break  # Restart from beginning
    return code


def _local_search(code, q, n, radius, deadline):
    """Try to improve the code by replacing codewords with better ones."""
    code = list(code)
    if not code:
        return code
    
    best_code = list(code)
    best_size = len(code)
    
    while time.time() < deadline - 10:
        # Try to find a replacement that reduces the code size
        # or at least maintains it while improving coverage
        improved = False
        
        # Try removing one codeword and adding a better one
        for i in range(len(code)):
            if time.time() > deadline - 10:
                break
            
            # Temporarily remove code[i]
            remaining = code[:i] + code[i+1:]
            
            # Find uncovered words
            covered = set()
            for cw in remaining:
                for w in _ball(cw, q, n, radius):
                    covered.add(tuple(w))
            
            # Check if all words are covered
            all_words_covered = True
            for w in itertools.product(range(q), repeat=n):
                if tuple(w) not in covered:
                    all_words_covered = False
                    break
            
            if all_words_covered:
                # We can remove code[i]
                code = remaining
                improved = True
                break
        
        if not improved:
            # Try swapping a codeword with a better one
            for i in range(len(code)):
                if time.time() > deadline - 10:
                    break
                
                # Try to find a codeword that covers more uncovered words
                # when combined with the rest of the code
                remaining = code[:i] + code[i+1:]
                
                covered_by_others = set()
                for cw in remaining:
                    for w in _ball(cw, q, n, radius):
                        covered_by_others.add(tuple(w))
                
                # Find words not covered by others
                uncovered = []
                for w in itertools.product(range(q), repeat=n):
                    if tuple(w) not in covered_by_others:
                        uncovered.append(tuple(w))
                
                if not uncovered:
                    code = remaining
                    improved = True
                    break
                
                # Find the best replacement codeword
                # Heuristic: try words near the center of uncovered region
                # For simplicity, try a few random candidates
                import random
                random.seed(42)
                candidates = []
                for _ in range(50):
                    cand = [random.randint(0, q-1) for _ in range(n)]
                    candidates.append(cand)
                
                best_cand = None
                best_covered = 0
                for cand in candidates:
                    covered_by_cand = set()
                    for w in _ball(cand, q, n, radius):
                        covered_by_cand.add(tuple(w))
                    newly_covered = len(covered_by_cand & set(uncovered))
                    if newly_covered > best_covered:
                        best_covered = newly_covered
                        best_cand = cand
                
                if best_cand and best_covered == len(uncovered):
                    code[i] = best_cand
                    improved = True
                    break
        
        if not improved:
            break
    
    if len(code) < best_size:
        best_code = code
        best_size = len(code)
    
    return best_code


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general cases."""
    # Start with empty code
    code = []
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    
    while uncovered and time.time() < deadline:
        # Find the codeword that covers the most uncovered words
        best_cw = None
        best_count = 0
        best_ball = None
        
        # Sample some candidate positions
        import random
        random.seed(0)
        candidates = list(uncovered)
        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)
        
        for cand in candidates:
            ball = set(_ball(cand, q, n, radius))
            count = len(ball & uncovered)
            if count > best_count:
                best_count = count
                best_cw = cand
                best_ball = ball
        
        if best_cw is None:
            break
        
        code.append(list(best_cw))
        uncovered -= best_ball
    
    return code
# EVOLVE-BLOCK-END
