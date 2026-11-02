"""Ternary covering code K_3(7,1) via randomized local search with iterative compression."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict

_DEADLINE = 150.0  # wall-clock limit
_SAFETY = 10.0     # safety margin


def _neighbors(codeword, q, n):
    """Yield all words within Hamming distance <= 1 of codeword."""
    yield tuple(codeword)
    for i in range(n):
        orig = codeword[i]
        for v in range(q):
            if v != orig:
                lst = list(codeword)
                lst[i] = v
                yield tuple(lst)


def _ball(codeword, q, n, radius=1):
    """Return the set of all words within Hamming distance <= radius of codeword."""
    if radius == 0:
        return {tuple(codeword)}
    if radius == 1:
        return set(_neighbors(codeword, q, n))
    # For general radius (not needed here but safe)
    result = set()
    for i in range(n):
        for v in range(q):
            if v != codeword[i]:
                lst = list(codeword)
                lst[i] = v
                result |= _ball(lst, q, n, radius - 1)
    result.add(tuple(codeword))
    return result


def _is_covered(code, q, n, radius=1):
    """Check if every word in {0..q-1}^n is within distance radius of some codeword."""
    total = q ** n
    covered = set()
    for cw in code:
        covered |= _ball(cw, q, n, radius)
    return len(covered) == total


def _coverage_count(code, q, n, radius=1):
    """Count how many words are covered by the code."""
    covered = set()
    for cw in code:
        covered |= _ball(cw, q, n, radius)
    return len(covered)


def _greedy_cover(q, n, radius=1, time_limit=30.0):
    """Greedy set cover: repeatedly add the codeword that covers the most uncovered words."""
    start = time.time()
    total = q ** n
    
    # Precompute balls for all possible codewords? Too many (3^7=2187), but manageable
    # Actually 2187 balls of size 1+7*2=15 each = ~32805 entries. Fine.
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    all_set = set(all_words)
    
    # Precompute ball for each word
    balls = {}
    for w in all_words:
        balls[w] = _ball(w, q, n, radius)
    
    uncovered = set(all_words)
    code = []
    
    while uncovered and (time.time() - start) < time_limit:
        best_cw = None
        best_gain = -1
        # Sample a subset of candidates for speed
        candidates = list(uncovered)
        if len(candidates) > 200:
            random.shuffle(candidates)
            candidates = candidates[:200]
        for cw in candidates:
            gain = len(balls[cw] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cw = cw
        if best_cw is None:
            break
        code.append(list(best_cw))
        uncovered -= balls[best_cw]
    
    return code


def _try_remove(code, q, n, radius=1):
    """Try to remove a codeword if coverage is still complete. Returns (new_code, removed_idx) or (code, -1)."""
    total = q ** n
    for i in range(len(code)):
        remaining = code[:i] + code[i+1:]
        covered = set()
        for cw in remaining:
            covered |= _ball(cw, q, n, radius)
        if len(covered) == total:
            return remaining, i
    return code, -1


def _try_replace(code, q, n, radius=1):
    """Try to replace a codeword with a better one that allows removal of another.
    Simple heuristic: for each pair, try merging."""
    total = q ** n
    # Try: for each codeword c, try moving it slightly to see if we can remove another
    # Simpler: try all single-coordinate changes of each codeword and check if we can drop another
    for i in range(len(code)):
        for pos in range(n):
            for v in range(q):
                if v == code[i][pos]:
                    continue
                new_cw = list(code[i])
                new_cw[pos] = v
                # Check if replacing code[i] with new_cw still covers everything
                # (we don't remove anything, just modify)
                # This doesn't reduce size, so skip for now
                pass
    return code


def _iterative_compression(code, q, n, radius=1, time_limit=60.0):
    """Iteratively try to compress the code by removing redundant codewords."""
    start = time.time()
    code = [list(c) for c in code]
    
    while (time.time() - start) < time_limit:
        changed = False
        # Try removing each codeword
        for i in range(len(code)):
            if (time.time() - start) >= time_limit:
                break
            remaining = code[:i] + code[i+1:]
            covered = set()
            for cw in remaining:
                covered |= _ball(cw, q, n, radius)
            if len(covered) == q ** n:
                code = remaining
                changed = True
                break
        if not changed:
            # Try pair replacement: replace two codewords with one
            for i in range(len(code)):
                if (time.time() - start) >= time_limit:
                    break
                for j in range(i + 1, len(code)):
                    if (time.time() - start) >= time_limit:
                        break
                    # Try all words as potential replacement for code[i] and code[j]
                    # Only try words that are in the union of their balls
                    ball_i = _ball(tuple(code[i]), q, n, radius)
                    ball_j = _ball(tuple(code[j]), q, n, radius)
                    union = ball_i | ball_j
                    # The new codeword must cover union \ (coverage from others)
                    others = code[:i] + code[i+1:j] + code[j+1:]
                    others_cover = set()
                    for cw in others:
                        others_cover |= _ball(cw, q, n, radius)
                    
                    # Words only covered by i or j
                    only_by_ij = union - others_cover
                    
                    # Try each word in only_by_ij as candidate new codeword
                    # Actually, we need a single word whose ball covers only_by_ij
                    # That's very restrictive. Let's just try words in ball_i ∪ ball_j
                    candidates = list(union)
                    if len(candidates) > 50:
                        random.shuffle(candidates)
                        candidates = candidates[:50]
                    
                    for new_cw in candidates:
                        new_ball = _ball(tuple(new_cw), q, n, radius)
                        if only_by_ij <= new_ball:
                            new_code = others + [list(new_cw)]
                            # Verify full coverage
                            full_cover = set()
                            for cw in new_code:
                                full_cover |= _ball(cw, q, n, radius)
                            if len(full_cover) == q ** n:
                                code = new_code
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
        if not changed:
            break
    
    return code


def construct(q, n, radius):
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + _DEADLINE - _SAFETY
    
    # Start with a greedy cover
    code = _greedy_cover(q, n, radius, time_limit=30.0)
    
    # Verify it's actually a cover
    if not _is_covered(code, q, n, radius):
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Iterative compression
    code = _iterative_compression(code, q, n, radius, time_limit=45.0)
    
    # Final check
    if not _is_covered(code, q, n, radius):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return code
# EVOLVE-BLOCK-END
