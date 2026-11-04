"""Football pool problem for 7 matches: ternary covering code K_3(7,1).

Strategy:
  1. Direct-sum construction with the ternary tetracode (9 * 3**(n-4) words).
     For q=3, n=7, radius=1 this gives 243 codewords.
  2. Greedy set-cover on top of the tetracode construction to try to reduce
     the number of codewords. We treat the problem as a set cover where each
     possible word in the space is an element and each candidate codeword
     covers its radius-1 neighborhood.
  3. Local search / removal pass to further reduce the count.

Given the time budget, we focus on a fast greedy that works on the full space
for n=7 (2187 words) and then attempts to remove redundant codewords.
"""

import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 110  # leave 10s margin before 120s limit
    
    # Direct sum with tetracode for q=3
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Try to improve via greedy reduction
        if time.time() < deadline - 5:
            improved = _try_reduce(q, n, radius, result, deadline)
            if improved is not None:
                return improved
        return result
    
    # General fallback: greedy set cover
    return _greedy_set_cover(q, n, radius, deadline)


def _try_reduce(q, n, radius, initial_words, deadline):
    """Attempt to reduce the number of codewords using greedy removal and
    possibly local search. Works for the specific case q=3, n=7, radius=1."""
    if q != 3 or n != 7 or radius != 1:
        return None
    
    space_size = 3 ** 7  # 2187
    
    # Precompute: for each word (as index 0..2186), its radius-1 neighborhood
    # as a set of indices.
    # Index = sum(word[i] * 3**i for i in range(7))
    
    powers = [3 ** i for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % 3
            idx //= 3
        return word
    
    # Build neighborhood table
    neighborhoods = [None] * space_size
    for idx in range(space_size):
        w = index_to_word(idx)
        neigh = [idx]
        for pos in range(n):
            for delta in (1, 2):
                w2 = list(w)
                w2[pos] = (w2[pos] + delta) % 3
                neigh.append(word_to_index(w2))
        neighborhoods[idx] = neigh
    
    # Current set of codewords (as indices)
    codeword_indices = set(word_to_index(w) for w in initial_words)
    
    # Coverage count for each element
    coverage = np.zeros(space_size, dtype=np.int32)
    for cw in codeword_indices:
        for nb in neighborhoods[cw]:
            coverage[nb] += 1
    
    current_count = len(codeword_indices)
    best_count = current_count
    best_codewords = set(codeword_indices)
    
    # Try removing redundant codewords
    # A codeword is redundant if removing it doesn't leave any element with coverage 0
    # We do this greedily: repeatedly try to remove codewords that are "least useful"
    
    # For efficiency, we'll track for each codeword how many elements it uniquely covers
    # (elements where coverage == 1 and this codeword is the only one covering it)
    
    def unique_coverage_count(cw):
        """Count elements uniquely covered by cw."""
        count = 0
        for nb in neighborhoods[cw]:
            if coverage[nb] == 1:
                count += 1
        return count
    
    # Greedy removal: try to remove codewords with 0 unique coverage
    # If no such, try to remove codewords that are most "redundant"
    
    removal_attempts = 0
    max_removal_attempts = 500
    
    while time.time() < deadline - 3 and removal_attempts < max_removal_attempts:
        removal_attempts += 1
        
        # Find codewords with minimum unique coverage
        min_unique = None
        candidate = None
        for cw in list(codeword_indices):
            ucc = unique_coverage_count(cw)
            if ucc == 0:
                candidate = cw
                break
            if min_unique is None or ucc < min_unique:
                min_unique = ucc
                candidate = cw
        
        if candidate is None:
            break
        
        # Try removing candidate
        cw = candidate
        # Check if removal is safe
        safe = True
        for nb in neighborhoods[cw]:
            if coverage[nb] == 1:
                safe = False
                break
        
        if safe:
            # Remove it
            codeword_indices.discard(cw)
            for nb in neighborhoods[cw]:
                coverage[nb] -= 1
            current_count -= 1
            if current_count < best_count:
                best_count = current_count
                best_codewords = set(codeword_indices)
        else:
            # Can't remove this one; try another strategy
            # Maybe try a different codeword with same unique coverage
            # For now, just break if we can't find any removable
            # Actually, let's try to find ANY removable
            found = False
            for cw2 in list(codeword_indices):
                if cw2 == cw:
                    continue
                safe2 = True
                for nb in neighborhoods[cw2]:
                    if coverage[nb] == 1:
                        safe2 = False
                        break
                if safe2:
                    codeword_indices.discard(cw2)
                    for nb in neighborhoods[cw2]:
                        coverage[nb] -= 1
                    current_count -= 1
                    if current_count < best_count:
                        best_count = current_count
                        best_codewords = set(codeword_indices)
                    found = True
                    break
            if not found:
                break
    
    # Local search: try to replace one codeword with another that covers more uncovered
    # elements, or try swaps to reduce count further.
    
    # For now, return if we improved
    if best_count < len(initial_words):
        return [index_to_word(idx) for idx in best_codewords]
    
    return None


def _greedy_set_cover(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    space_size = q ** n
    
    # For large spaces, use a simpler approach
    if space_size > 50000:
        # Random construction
        random.seed(42)
        words = []
        seen = set()
        while len(words) < 200 and time.time() < deadline - 5:
            w = tuple(random.randint(0, q - 1) for _ in range(n))
            if w not in seen:
                seen.add(w)
                words.append(list(w))
        return words
    
    powers = [q ** i for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
    
    # Build neighborhoods for radius=1
    if radius == 1:
        neighborhoods = [None] * space_size
        for idx in range(space_size):
            w = index_to_word(idx)
            neigh = [idx]
            for pos in range(n):
                for delta in range(1, q):
                    w2 = list(w)
                    w2[pos] = (w2[pos] + delta) % q
                    neigh.append(word_to_index(w2))
            neighborhoods[idx] = neigh
    
    # Greedy: start with empty, repeatedly add the codeword that covers the most
    # currently uncovered elements.
    
    coverage = np.zeros(space_size, dtype=np.int32)
    uncovered = np.ones(space_size, dtype=bool)
    selected = []
    selected_set = set()
    
    while time.time() < deadline - 3 and uncovered.any():
        # Find the codeword that covers the most uncovered elements
        best_cw = -1
        best_count = -1
        
        # Sample a subset of candidates for speed
        candidates = list(range(space_size))
        random.shuffle(candidates)
        candidates = candidates[:min(500, space_size)]
        
        for cw in candidates:
            if cw in selected_set:
                continue
            count = 0
            for nb in neighborhoods[cw]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_cw = cw
        
        if best_cw < 0:
            break
        
        # Add best_cw
        selected.append(index_to_word(best_cw))
        selected_set.add(best_cw)
        for nb in neighborhoods[best_cw]:
            coverage[nb] += 1
            if coverage[nb] == 1:
                uncovered[nb] = False
    
    return selected
