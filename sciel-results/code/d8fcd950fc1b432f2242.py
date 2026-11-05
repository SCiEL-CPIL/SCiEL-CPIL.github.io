"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _ball_indices(word, q, n, radius):
    """Return list of integer indices (base-q) of all words within Hamming distance <= radius."""
    # For small n and q, enumerate all words in the ball
    word = tuple(word)
    results = [int(''.join(str(d) for d in word), base=q) if n < 20 else _encode(word, q)]
    # Add neighbors at distance 1
    for pos in range(n):
        orig = word[pos]
        for val in range(q):
            if val == orig:
                continue
            new_word = list(word)
            new_word[pos] = val
            results.append(_encode(tuple(new_word), q))
    return results


def _encode(word, q):
    """Encode a tuple of digits to an integer."""
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _decode(idx, q, n):
    """Decode an integer to a tuple of n digits in base q."""
    digits = []
    for _ in range(n):
        digits.append(idx % q)
        idx //= q
    return tuple(reversed(digits))


def construct(q, n, radius):
    global _START_TIME
    _START_TIME = time.time()
    
    deadline = 120.0  # seconds
    safety_margin = 5.0
    
    total_space = q ** n
    all_words = list(range(total_space))
    
    # Precompute balls for all words
    # ball[word_idx] = set of word indices within distance <= radius
    # For efficiency, store as numpy arrays
    balls = []
    for idx in range(total_space):
        word = _decode(idx, q, n)
        ball = _ball_indices(word, q, n, radius)
        balls.append(set(ball))
    
    # Greedy set cover
    uncovered = set(all_words)
    chosen = []
    
    while uncovered:
        # Pick the word whose ball covers the most uncovered words
        best_word = -1
        best_count = -1
        for idx in uncovered:
            count = len(balls[idx] & uncovered)
            if count > best_count:
                best_count = count
                best_word = idx
            # Early stop if we found perfect coverage
            if count == len(uncovered):
                break
        
        if best_word == -1:
            break
        
        chosen.append(best_word)
        uncovered -= balls[best_word]
        
        if _elapsed() > deadline - safety_margin:
            break
    
    if not chosen:
        # Fallback to tetracode-based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Local improvement: try to remove words that are redundant
    def is_covered_without(word_idx, chosen_set, balls):
        """Check if removing word_idx still leaves everything covered."""
        if not chosen_set:
            return False
        # For each word in the ball of word_idx, check if it's covered by another chosen word
        for w in balls[word_idx]:
            covered = False
            for other in chosen_set:
                if other == word_idx:
                    continue
                if w in balls[other]:
                    covered = True
                    break
            if not covered:
                return False
        return True
    
    # Try to remove redundant words
    changed = True
    while changed and _elapsed() < deadline - safety_margin:
        changed = False
        for i in range(len(chosen)):
            if _elapsed() > deadline - safety_margin:
                break
            word_idx = chosen[i]
            # Check if this word is redundant
            redundant = True
            for w in balls[word_idx]:
                covered_by_other = False
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    if w in balls[chosen[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            if redundant:
                chosen.pop(i)
                changed = True
    
    # Try swap improvements: replace one word with another to cover more
    # This is expensive, so do a limited number of iterations
    for iteration in range(50):
        if _elapsed() > deadline - safety_margin:
            break
        improved = False
        for i in range(len(chosen)):
            if _elapsed() > deadline - safety_margin:
                break
            old_word = chosen[i]
            # Try replacing with a word from the uncovered set or a word not in chosen
            # For simplicity, try a few random candidates
            for attempt in range(20):
                if _elapsed() > deadline - safety_margin:
                    break
                new_word = np.random.randint(0, total_space)
                if new_word in chosen:
                    continue
                # Check if replacing old_word with new_word maintains coverage
                # All words covered by old_word must be covered by new_word or other chosen words
                valid = True
                for w in balls[old_word]:
                    covered = False
                    if w in balls[new_word]:
                        covered = True
                    else:
                        for j in range(len(chosen)):
                            if j == i:
                                continue
                            if w in balls[chosen[j]]:
                                covered = True
                                break
                    if not covered:
                        valid = False
                        break
                if valid:
                    # Check if new_word covers something old_word didn't (for future removals)
                    chosen[i] = new_word
                    improved = True
                    break
        if not improved:
            break
    
    # Convert back to lists of integers
    result = []
    for idx in chosen:
        word = _decode(idx, q, n)
        result.append(list(word))
    
    # Verify validity
    covered = set()
    for idx in chosen:
        covered |= balls[idx]
    if len(covered) < total_space:
        # Fallback to tetracode-based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return result
# EVOLVE-BLOCK-END
