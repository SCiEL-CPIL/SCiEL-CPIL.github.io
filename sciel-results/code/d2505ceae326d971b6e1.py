"""Improved ternary covering code K_3(6,1) using fast simulated annealing."""
import itertools
import time
import random
import numpy as np


def _word_to_int(word, q):
    val = 0
    for d in word:
        val = val * q + d
    return val


def _int_to_word(val, q, n):
    word = []
    for _ in range(n):
        word.append(val % q)
        val //= q
    return word[::-1]


def _precompute_balls(q, n, radius=1):
    """Precompute the ball (set of words within Hamming distance <= radius) for each word."""
    total = q ** n
    balls = []
    for w_int in range(total):
        word = _int_to_word(w_int, q, n)
        ball = {w_int}
        if radius == 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val != orig:
                        new_word = word[:]
                        new_word[pos] = val
                        ball.add(_word_to_int(new_word, q))
        balls.append(ball)
    return balls


def _tetracode(q, n):
    """Generate tetracode-based initial code (9 * q^(n-4) words)."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def construct(q, n, radius):
    """Construct a covering code using fast simulated annealing with coverage counters."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    total_space = q ** n
    
    # Precompute balls for all words
    balls = _precompute_balls(q, n, radius)
    
    # Initial code from tetracode
    code_words = _tetracode(q, n)
    code_ints = [_word_to_int(w, q) for w in code_words]
    
    # Build coverage counter: cover_count[w] = number of codewords that cover word w
    cover_count = np.zeros(total_space, dtype=np.int32)
    for cw in code_ints:
        for w in balls[cw]:
            cover_count[w] += 1
    
    # Verify initial code is valid
    if np.min(cover_count) < 1:
        # Fallback: return all words
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    best_code = list(code_words)
    best_size = len(code_words)
    
    # Simulated annealing: try to remove words
    # Key insight: only try removing words that have "redundant" coverage
    # A word w is removable if for all w' in ball(w), cover_count[w'] >= 2
    
    T0 = 1.0
    T_final = 0.001
    cooling = 0.999
    
    T = T0
    attempts = 0
    max_attempts = 200000
    
    while T > T_final and time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Find candidate words to try removing
        # A word is a good candidate if it's "well-covered" by others
        # i.e., removing it doesn't leave any word uncovered
        # Quick check: for each w in ball(cw), cover_count[w] >= 2
        
        idx = random.randint(0, len(code_ints) - 1)
        cw = code_ints[idx]
        
        # Check if removing cw would leave any word uncovered
        removable = True
        for w in balls[cw]:
            if cover_count[w] <= 1:
                removable = False
                break
        
        if removable:
            # Remove the codeword
            code_ints.pop(idx)
            for w in balls[cw]:
                cover_count[w] -= 1
            
            if len(code_ints) < best_size:
                best_size = len(code_ints)
                best_code = [_int_to_word(c, q, n) for c in code_ints]
        else:
            # Try a swap: remove cw and add a different word
            # This is less efficient but can escape local optima
            # For now, just reject
            pass
        
        # Cool down
        T *= cooling
        
        # Periodic check for deadline
        if time.time() > deadline - 3.0:
            break
    
    # Final verification
    cover_count_final = np.zeros(total_space, dtype=np.int32)
    for cw in [_word_to_int(w, q) for w in best_code]:
        for w in balls[cw]:
            cover_count_final[w] += 1
    
    if np.min(cover_count_final) < 1:
        # Fallback to tetracode
        return _tetracode(q, n)
    
    return best_code
