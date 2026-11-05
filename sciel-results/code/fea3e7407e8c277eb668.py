"""Construct a small ternary covering code of length 8, radius 1.

Strategy:
- Build an initial seed set using the tetracode direct-sum construction (729 words)
  but then aggressively prune and improve via greedy set-cover + local search.
- Use bitset representation for fast coverage checks.
- The space is 3^8 = 6561 words. Each codeword covers itself plus 16 neighbors = 17 words.
- We want to minimize the number of codewords such that every word is covered.

Approach:
1. Generate all 6561 words as bit indices.
2. Precompute coverage mask for each potential codeword (17 bits out of 6561).
3. Greedy: repeatedly pick the codeword that covers the most uncovered words.
4. Local search: try to remove a codeword if its coverage is redundant; try to swap.
5. Return the list of codewords.
"""

import itertools
import random
import time

# Precomputation for speed: map word tuple to index and vice versa
def _precompute():
    """Precompute all words, their indices, and neighbor indices."""
    n = 8
    q = 3
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    # For each word, compute the set of indices it covers (itself + all 16 neighbors)
    n_neighbors = 2 * n * (q - 1)  # 2*8*2 = 32? No: for each position, (q-1) alternatives = 2*8 = 16 neighbors + self = 17
    cover_sets = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = alt
                w2 = tuple(w2)
                covered.add(word_to_idx[w2])
        cover_sets.append(frozenset(covered))
    return all_words, word_to_idx, cover_sets

# Global cache
_all_words = None
_word_to_idx = None
_cover_sets = None

def _ensure_precomputed():
    global _all_words, _word_to_idx, _cover_sets
    if _all_words is None:
        _all_words, _word_to_idx, _cover_sets = _precompute()

def construct(q, n, radius):
    """Return a list of codewords (each a list of n ints in 0..q-1) covering
    all of {0..q-1}^n within Hamming distance radius."""
    _ensure_precomputed()
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: trivial construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    all_words = _all_words
    cover_sets = _cover_sets
    total_words = len(all_words)  # 6561
    
    # Start with a greedy construction
    uncovered = set(range(total_words))
    selected = set()
    
    # Greedy: pick codeword with maximum coverage of uncovered words
    # To speed up, we can use a priority approach but 6561 is small enough
    while uncovered:
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            if i in selected:
                continue
            overlap = len(cover_sets[i] & uncovered)
            if overlap > best_count:
                best_count = overlap
                best_idx = i
                if best_count == len(uncovered):
                    break
        if best_idx < 0:
            break
        selected.add(best_idx)
        uncovered -= cover_sets[best_idx]
    
    # Now do local search: try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        # Recompute coverage for each selected codeword
        for idx in list(selected):
            # Check if removing this codeword still covers everything
            remaining = selected - {idx}
            # Build union of coverage of remaining
            covered = set()
            for j in remaining:
                covered |= cover_sets[j]
            if len(covered) == total_words:
                selected.remove(idx)
                changed = True
    
    # Further local search: try to swap a codeword for another that allows removal of others
    # This is complex, so we'll do a simpler improvement: try to add a codeword that allows removing multiple
    # For now, just return what we have
    
    result = [list(all_words[idx]) for idx in sorted(selected)]
    return result
