"""Structural change: greedy set-cover (greedy maximal independent set style) with a
tight deadline guard. The seed returns 9*3**(n-4) words; for n=8 that is 9*81=729.
We try to beat that with a fast greedy covering-code construction.

Key idea:
- Represent the space {0..q-1}^n as integers in [0, q**n).
- Precompute, for each codeword, the set of words it covers (Hamming ball of radius r).
- Greedy: repeatedly pick the codeword that covers the most currently-uncovered words.
- To keep this feasible within ~120s, we:
  * Use a randomized greedy with multiple restarts and early termination.
  * Use a bitset (Python int) for covered set to make "newly covered" count fast.
  * Limit the number of candidate codewords we consider per step (e.g., sample).
  * Track a deadline and return the best solution found.

For n=8, q=3, radius=1:
  Total words = 3^8 = 6561.
  Each ball has 1 + 8*2 = 17 words.
  Lower bound ~403, best known 486.
  Greedy often gets within ~1.5x of optimum for covering codes, so ~486-730 is plausible.
  We aim to get below 729 (the seed's answer).

Optimization:
- Precompute for each word w (0..6560) the list of words in its radius-1 ball.
- Represent each ball as a Python int bitmask (6561 bits ~ 820 bytes, fine).
- Maintain a global 'covered' bitmask.
- At each step, for a set of candidate codewords, compute popcount(ball & ~covered),
  pick the max, add it, update covered.
- To speed up: instead of checking all 6561 candidates each step, we can:
  * First pass: pick the best among all (6561) – this is 6561 popcounts per step.
    6561 * ~500 steps * popcount cost... popcount of 6561-bit int is fast in Python?
    Actually int.bit_count() is O(number of machine words) = O(6561/64) ~ 103 ops.
    6561 * 500 * 103 ≈ 338M ops – too slow in pure Python.

  Better approach: 
  * Use a priority queue / heap where we lazily update. But updates are expensive.
  * Alternative: use a smaller candidate pool. E.g., in each step, sample S random 
    uncovered-adjacent codewords and pick the best among them.
  * Or: use a two-level approach: partition the space, do greedy within blocks.

Given the 120s limit, let's try a simpler but effective heuristic:
  1. Generate a random ordering of all 6561 words.
  2. Greedy scan: for each word in that order, if it covers at least T new words, add it.
     T can be adaptive: start high, decrease.
  3. Multiple random restarts, keep the best (fewest codewords) complete cover.

Even simpler and often effective: "greedy maximal" where we pick the word with max
new coverage from the full set, but we approximate by only checking a subset.

Let me try: 
  - Precompute balls as frozensets (for intersection) – but 6561 balls * 17 elements = ~111k elements, fine.
  - Greedy: at each step, compute for each candidate c: len(ball[c] - covered_set).
    This is 6561 set differences per step, each O(17). 6561*17 ≈ 111k ops per step.
    ~500 steps → 55M ops. In Python, ~5-10s. Feasible!

Let's do that with a deadline guard.
"""

import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    total_words = q ** n
    # Map word (tuple) to int index
    def to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def from_idx(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word[::-1]

    # Generate all words and their indices
    all_words = [from_idx(i) for i in range(total_words)]

    # Precompute balls: for each index, list of indices in radius-r Hamming ball
    # For radius=1: self + all single-position changes
    def get_ball_indices(word, q, n, radius):
        """Return list of indices within Hamming distance <= radius."""
        # For radius=1, we can do it directly
        if radius == 0:
            return [to_idx(word)]
        elif radius == 1:
            indices = [to_idx(word)]
            for pos in range(n):
                orig = word[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    new_word = list(word)
                    new_word[pos] = v
                    indices.append(to_idx(new_word))
            return indices
        else:
            # General case: BFS or combinatorial – for this task radius=1
            # Fallback: brute force (slow but correct)
            indices = []
            for i in range(total_words):
                w2 = all_words[i]
                dist = sum(1 for a, b in zip(word, w2) if a != b)
                if dist <= radius:
                    indices.append(i)
            return indices

    # Precompute all balls
    balls = [get_ball_indices(all_words[i], q, n, radius) for i in range(total_words)]
    # Convert to sets for fast intersection/difference
    ball_sets = [set(b) for b in balls]

    # Greedy set cover
    def greedy_cover(seed_rng=None):
        rng = seed_rng or random
        covered = set()
        code = []
        remaining = total_words
        # We'll iterate: at each step find the word with max new coverage
        # To speed up, we can use a candidate list
        for step in range(total_words):  # upper bound on steps
            if time.time() > deadline:
                return None
            best_idx = -1
            best_new = 0
            # Check all candidates
            for idx in range(total_words):
                if idx in covered:
                    continue
                new_cover = len(ball_sets[idx] - covered)
                if new_cover > best_new:
                    best_new = new_cover
                    best_idx = idx
            if best_idx == -1 or best_new == 0:
                break
            code.append(best_idx)
            covered.update(ball_sets[best_idx])
            remaining -= best_new
            if remaining <= 0:
                break
        if remaining <= 0:
            return code
        return None

    # Try greedy with a few restarts
    best_code = None
    for trial in range(5):
        if time.time() > deadline:
            break
        code = greedy_cover()
        if code is not None:
            if best_code is None or len(code) < len(best_code):
                best_code = code
            if len(best_code) <= 403:  # lower bound, can't do better
                break

    if best_code is not None:
        return [all_words[i] for i in best_code]

    # Fallback: return the seed's construction
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # For n < 4, just return all words
        return [from_idx(i) for i in range(total_words)]
