"""Football pool / ternary covering code K_3(6,1): minimize number of codewords covering {0,1,2}^6 within Hamming distance 1."""
import time
import random
import numpy as np
from itertools import product

# Precompute all words in {0,1,2}^6 as integers 0..728
# Encoding: word w = sum w[i] * 3**i
Q = 3
N = 6
R = 1
TOTAL = Q ** N  # 729

# Precompute the "ball" of radius 1 around each word (word + all single-position changes)
# For word w (as int), the ball contains w and w +/- 1 at each position (mod 3)
# We precompute for each word index, the list of word indices in its ball.
_POWERS = [1, 3, 9, 27, 81, 243]

def _word_to_int(word):
    """Convert a list of 6 ternary digits to an integer 0..728."""
    v = 0
    for i, d in enumerate(word):
        v += d * _POWERS[i]
    return v

def _int_to_word(v):
    """Convert integer 0..728 to a list of 6 ternary digits."""
    return [(v // _POWERS[i]) % 3 for i in range(N)]

# Precompute balls: for each word index, list of word indices within Hamming distance <= 1
_BALLS = []
for w in range(TOTAL):
    word = _int_to_word(w)
    ball = {w}
    for i in range(N):
        for d in range(Q):
            if d != word[i]:
                new_word = word[:]
                new_word[i] = d
                ball.add(_word_to_int(new_word))
    _BALLS.append(sorted(ball))

# For each word index, precompute which codeword positions (0..728) could cover it
# i.e., _COVERERS[word_idx] = list of codeword indices c such that word_idx is in ball(c)
# This is the transpose of _BALLS.
_COVERERS = [[] for _ in range(TOTAL)]
for c in range(TOTAL):
    for w in _BALLS[c]:
        _COVERERS[w].append(c)


def construct(q, n, radius):
    """Construct a covering code for q=3, n=6, radius=1. Returns list of codewords."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    # Start with a good initial solution: direct product construction gives 81 codewords.
    # We'll improve it via iterated local search.
    
    # Initial solution: all-zero word plus greedy
    # Actually, let's start with a known good structure: the tetracode product gives 81.
    # Let's start with a random set and repair, or use greedy.
    
    # Greedy construction:
    uncovered = set(range(TOTAL))
    codewords = []
    
    while uncovered:
        # Pick the codeword that covers the most uncovered words
        best_c = None
        best_count = -1
        # Sample if too many candidates
        candidates = list(uncovered)
        if len(candidates) > 200:
            candidates = random.sample(candidates, 200)
        for c in candidates:
            count = 0
            for w in _BALLS[c]:
                if w in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_c = c
        codewords.append(best_c)
        for w in _BALLS[best_c]:
            uncovered.discard(w)
    
    best_set = set(codewords)
    best_count = len(best_set)
    
    # Iterated Local Search: try to remove redundant codewords and swap to improve
    rng = random.Random(42)
    
    def is_covered(codeword_set):
        """Check if all words are covered by the given set of codewords."""
        covered = np.zeros(TOTAL, dtype=bool)
        for c in codeword_set:
            for w in _BALLS[c]:
                covered[w] = True
        return np.all(covered)
    
    def get_uncovered(codeword_set):
        """Return set of uncovered word indices."""
        covered = np.zeros(TOTAL, dtype=bool)
        for c in codeword_set:
            for w in _BALLS[c]:
                covered[w] = True
        return set(np.where(~covered)[0])
    
    # Local search: try removing each codeword, see if still covered
    current_set = set(best_set)
    current_count = len(current_set)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing each codeword
        for c in list(current_set):
            if time.time() > deadline:
                break
            trial_set = current_set - {c}
            if len(trial_set) == 0:
                continue
            if is_covered(trial_set):
                current_set = trial_set
                current_count -= 1
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add another
            if time.time() > deadline:
                break
            # Try a random swap
            if len(current_set) > 5:
                c_remove = rng.choice(list(current_set))
                # Find candidates to add that cover the most words not covered by others
                without_c = current_set - {c_remove}
                uncovered_without = get_uncovered(without_c)
                if uncovered_without:
                    # Find best addition
                    best_add = None
                    best_gain = 0
                    # Sample candidates
                    all_candidates = list(range(TOTAL))
                    if len(all_candidates) > 100:
                        all_candidates = rng.sample(all_candidates, 100)
                    for c_add in all_candidates:
                        if c_add in current_set:
                            continue
                        gain = 0
                        for w in _BALLS[c_add]:
                            if w in uncovered_without:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_add = c_add
                    if best_add is not None and best_gain > 0:
                        new_set = without_c | {best_add}
                        if is_covered(new_set):
                            current_set = new_set
                            improved = True
        
        if current_count < best_count:
            best_set = set(current_set)
            best_count = current_count
    
    # Convert best_set to list of codeword lists
    result = [_int_to_word(c) for c in sorted(best_set)]
    return result
