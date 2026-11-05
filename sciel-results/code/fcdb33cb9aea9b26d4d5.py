"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Precompute the ternary tetracode (9 words of length 4, perfect covering radius 1)
_TETRACODE = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
_TET_SET = frozenset(tuple(t) for t in _TETRACODE)

# For the "tail" part (last n-4 coordinates), we can use a greedy covering code
# For n=8, tail length is 4. A perfect code of length 4 over q=3 with radius 1 has size 9 (the tetracode itself).
# So the direct sum gives 9 * 9 = 81 words? No, the direct sum construction in the seed uses ALL tails: 9 * 3^(n-4).
# For n=8, that's 9 * 81 = 729 words.
# The tetracode is a perfect code, so every word of length 4 is within distance 1 of exactly one tetracode word.
# The direct sum: codeword = (t, tail) where t in tetracode, tail in {0,1,2}^(n-4).
# A word (u, v) is covered if there exists t in tetracode with d(u,t) <= 1 (always true, perfect) AND d(v, tail) <= 1.
# Wait, the distance in the direct sum is d((t,tail),(u,v)) = d(t,u) + d(tail,v).
# For this to be <= radius=1, we need d(t,u) + d(tail,v) <= 1.
# Since d(t,u) >= 0 and d(tail,v) >= 0, this means either:
#   d(t,u) = 0 and d(tail,v) <= 1, OR
#   d(t,u) = 1 and d(tail,v) = 0.
# So for a given (u,v), it's covered if:
#   There exists t in tetracode with t=u (i.e., u is in tetracode) and tail=v, OR
#   There exists t in tetracode with d(t,u)=1 and tail=v.
# Since the tetracode is perfect, for any u, there is exactly one t with d(t,u)<=1.
# If u is in tetracode, t=u, and we need tail=v (d=0). So (u,v) is covered by (u,v) itself if u in tetracode.
# If u is not in tetracode, there's a unique t with d(t,u)=1, and we need tail=v (d=0). So (u,v) is covered by (t,v).
# Thus, EVERY (u,v) is covered by the direct sum code! The direct sum is a valid covering code.
# Size: 9 * 3^(n-4). For n=8: 9*81=729.

# But the best known is 486. So we need a better construction.
# Idea: Instead of using ALL tails, use a covering code for the tail part too.
# If we use a covering code C_tail of size M for the tail (length n-4, radius r_tail),
# and the tetracode (radius 1), the combined radius is 1 + r_tail.
# We want total radius <= 1, so r_tail must be 0. That means C_tail must be the full space.
# So the direct sum with radius 1 requires the full tail space.

# Alternative: Use a different construction.
# The van Laarhoven et al. code of size 486 = 2 * 243 = 2 * 3^5.
# Or 486 = 6 * 81 = 6 * 3^4.
# Actually, 486 = 2 * 3^5 = 2 * 243.
# Another known construction: the ternary code of length 8, size 486, radius 1.
# This can be constructed using the "doubling" or "product" method.

# Let's try a greedy algorithm with bitset acceleration.
# For n=8, q=3, the space has 3^8 = 6561 words.
# Each codeword covers 1 + 8*2 = 17 words (itself + 8 positions * 2 changes).
# So we need at least ceil(6561/17) = 386 codewords.
# The best known is 486.

# Greedy algorithm:
# 1. Start with an empty set of codewords.
# 2. Maintain a bitset of uncovered words.
# 3. At each step, choose the codeword that covers the most uncovered words.
# 4. Add it to the code, update the uncovered set.
# 5. Repeat until all words are covered.

# Then use local search to improve:
# - Try removing a codeword and see if the code is still valid.
# - Try swapping a codeword with another to reduce the size or improve coverage.

# For efficiency, we'll use numpy arrays and bit operations.

def _build_cover_table(q, n, radius):
    """Build a table mapping each word to the set of codeword indices that cover it.
    But this is too large. Instead, we'll compute coverage on the fly."""
    pass

def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover for the covering code problem."""
    total_words = q ** n
    # Represent each word as an integer in [0, total_words)
    # A word w = (w0, w1, ..., w_{n-1}) maps to w0 + w1*q + ... + w_{n-1}*q^(n-1)
    
    # Precompute for each word, the list of words within distance <= radius
    # This is expensive for n=8, q=3, radius=1: 6561 * 17 = ~111k entries. Manageable.
    
    # Actually, for the greedy algorithm, we need to know which words each candidate codeword covers.
    # Let's precompute the "ball" around each word: the set of words within distance <= radius.
    
    # For radius=1, the ball around w is {w} union {w with one coordinate changed}.
    
    # Let's represent the uncovered set as a boolean array of size total_words.
    uncovered = np.ones(total_words, dtype=bool)
    codewords = []
    
    # Precompute the balls
    # ball[w] = list of words within distance <= radius of w
    balls = []
    for w in range(total_words):
        # Convert w to a word
        word = [(w // (q ** i)) % q for i in range(n)]
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for new_val in range(q):
                    if new_val != word[i]:
                        new_word = word[:]
                        new_word[i] = new_val
                        idx = sum(new_word[j] * (q ** j) for j in range(n))
                        ball.add(idx)
        balls.append(list(ball))
    
    # For efficiency, convert balls to numpy arrays
    balls_np = [np.array(b, dtype=np.int64) for b in balls]
    
    while uncovered.any() and time.time() < deadline - 10:
        # Find the word that covers the most uncovered words
        best_idx = -1
        best_count = -1
        # This is O(total_words * ball_size) per step, which is 6561 * 17 = ~111k per step.
        # With ~400 steps, that's ~44M operations. Might be slow in Python.
        # Let's optimize by using numpy.
        
        # For each candidate word, count how many uncovered words it covers.
        # uncovered is a boolean array. balls_np[i] is an array of indices.
        # count_i = uncovered[balls_np[i]].sum()
        
        # We can vectorize this, but it's memory intensive.
        # Let's just loop, but skip words that are already covered (they might still be good candidates).
        # Actually, any word can be a codeword, covered or not.
        
        # Optimization: only consider words that are uncovered or near uncovered.
        # For now, let's do a simple loop with numpy.
        
        counts = np.array([uncovered[b].sum() for b in balls_np])
        best_idx = int(np.argmax(counts))
        best_count = counts[best_idx]
        
        if best_count <= 0:
            break
        
        codewords.append(best_idx)
        # Mark covered words
        uncovered[balls_np[best_idx]] = False
    
    # Convert indices back to words
    result = []
    for idx in codewords:
        word = [(idx // (q ** i)) % q for i in range(n)]
        result.append(word)
    
    return result

def _is_valid_code(q, n, radius, codewords):
    """Check if the code is valid: every word is within distance <= radius of some codeword."""
    total_words = q ** n
    covered = np.zeros(total_words, dtype=bool)
    
    for cw in codewords:
        # Add all words within distance <= radius of cw
        for d in range(radius + 1):
            if d == 0:
                idx = sum(cw[i] * (q ** i) for i in range(n))
                covered[idx] = True
            else:
                # Choose d positions to change
                for positions in itertools.combinations(range(n), d):
                    # For each position, try all other values
                    # This is expensive. Let's use a different approach.
                    pass
    
    # This is too slow. Let's use the ball precomputation.
    # Actually, for the local search, we'll recompute coverage efficiently.
    return covered.all()

def _local_search(q, n, radius, codewords, deadline):
    """Local search to improve the code."""
    total_words = q ** n
    
    # Precompute balls
    balls = []
    for w in range(total_words):
        word = [(w // (q ** i)) % q for i in range(n)]
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for new_val in range(q):
                    if new_val != word[i]:
                        new_word = word[:]
                        new_word[i] = new_val
                        idx = sum(new_word[j] * (q ** j) for j in range(n))
                        ball.add(idx)
        balls.append(list(ball))
    balls_np = [np.array(b, dtype=np.int64) for b in balls]
    
    # Convert codewords to indices
    cw_indices = [sum(cw[i] * (q ** i) for i in range(n)) for cw in codewords]
    
    # Coverage count: how many times each word is covered
    coverage = np.zeros(total_words, dtype=np.int32)
    for idx in cw_indices:
        coverage[balls_np[idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing each codeword
        for i in range(len(cw_indices)):
            if time.time() > deadline - 5:
                break
            idx = cw_indices[i]
            # Temporarily remove
            coverage[balls_np[idx]] -= 1
            if coverage.min() >= 1:
                # Can remove
                cw_indices.pop(i)
                improved = True
                break
            else:
                # Restore
                coverage[balls_np[idx]] += 1
    
    # Convert back
    result = []
    for idx in cw_indices:
        word = [(idx // (q ** i)) % q for i in range(n)]
        result.append(word)
    
    return result

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with the direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        codewords = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Greedy improvement
    if time.time() < deadline - 30:
        codewords = _greedy_cover(q, n, radius, deadline)
    
    # Local search
    if time.time() < deadline - 5:
        codewords = _local_search(q, n, radius, codewords, deadline)
    
    return codewords
# EVOLVE-BLOCK-END
