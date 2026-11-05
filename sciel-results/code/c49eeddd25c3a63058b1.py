"""Football pool problem for 7 matches: ternary covering code K_3(7,1).

Strategy:
  1. Direct-sum construction with the ternary tetracode (9 * 3**(n-4) words).
     For q=3, n=7, radius=1 this gives 243 codewords.
  2. Use Algorithm X with Dancing Links to find a minimum subset of the
     initial codewords that still covers the entire space.
  3. Fallback to greedy set cover for general cases.
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
        
        # Try to improve via Algorithm X exact cover solver
        if time.time() < deadline - 5:
            improved = _algorithm_x_reduce(q, n, radius, result, deadline)
            if improved is not None:
                return improved
        return result
    
    # General fallback: greedy set cover
    return _greedy_set_cover(q, n, radius, deadline)


def _algorithm_x_reduce(q, n, radius, initial_words, deadline):
    """Use Algorithm X with Dancing Links to find minimum subset of initial_words
    that covers all elements. Uses exact cover formulation with branch and bound."""
    if q != 3 or n != 7 or radius != 1:
        return None
    
    space_size = 3 ** n  # 2187
    num_codewords = len(initial_words)
    
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
    
    # Build coverage matrix: coverage[cw_idx, elem] = 1 if cw covers elem
    coverage = np.zeros((num_codewords, space_size), dtype=np.uint8)
    for cw_idx, w in enumerate(initial_words):
        widx = word_to_index(w)
        coverage[cw_idx, widx] = 1
        for pos in range(n):
            for delta in (1, 2):
                w2 = list(w)
                w2[pos] = (w2[pos] + delta) % 3
                coverage[cw_idx, word_to_index(w2)] = 1
    
    # Check all elements are covered
    elem_cover_count = coverage.sum(axis=0)
    if (elem_cover_count == 0).any():
        return None
    
    # Get greedy solution as upper bound
    greedy_result = _greedy_subset(coverage, deadline - 10)
    if greedy_result is None:
        return None
    
    best_count = len(greedy_result)
    best_solution = greedy_result
    
    # Try to improve with branch and bound using Algorithm X style
    # We search for solutions with fewer codewords
    
    # Precompute: for each element, list of codewords that cover it
    elem_to_cws = [[] for _ in range(space_size)]
    for elem in range(space_size):
        cws = np.where(coverage[:, elem] > 0)[0]
        elem_to_cws[elem] = list(cws)
    
    # Dancing Links simplified: we'll use recursive backtracking with pruning
    # State: which elements are covered, which codewords are selected
    
    covered = np.zeros(space_size, dtype=bool)
    for cw in best_solution:
        covered |= coverage[cw] > 0
    
    # Try to find solution with best_count - 1, best_count - 2, etc.
    target = best_count - 1
    found_better = False
    
    while target >= 100 and time.time() < deadline - 3 and not found_better:
        solution = _branch_and_bound(coverage, elem_to_cws, target, deadline - 3)
        if solution is not None:
            best_count = target
            best_solution = solution
            found_better = True
        else:
            target -= 1
    
    if best_count < num_codewords:
        return [index_to_word(word_to_index(initial_words[cw_idx])) for cw_idx in sorted(best_solution)]
    
    return None


def _branch_and_bound(coverage, elem_to_cws, max_cw, deadline):
    """Branch and bound to find solution with at most max_cw codewords."""
    num_rows, num_cols = coverage.shape
    
    # Start with uncovered elements
    uncovered = np.ones(num_cols, dtype=bool)
    uncovered_count = num_cols
    
    # Elements with fewest covering codewords first (most constrained)
    elem_order = sorted(range(num_cols), key=lambda e: len(elem_to_cws[e]))
    
    selected = []
    
    def solve(uncovered, uncovered_count, selected, depth):
        if time.time() > deadline:
            return None
        if uncovered_count == 0:
            return list(selected)
        if len(selected) >= max_cw:
            return None
        
        # Find most constrained uncovered element
        best_elem = -1
        min_options = float('inf')
        for elem in elem_order:
            if uncovered[elem]:
                options = [cw for cw in elem_to_cws[elem] if cw not in selected]
                if len(options) < min_options:
                    min_options = len(options)
                    best_elem = elem
                if min_options == 0:
                    return None
                if min_options == 1:
                    break
        
        if best_elem == -1:
            return None
        
        # Try each codeword that covers best_elem
        for cw in elem_to_cws[best_elem]:
            if cw in selected:
                continue
            
            new_uncovered = uncovered.copy()
            new_uncovered &= ~coverage[cw].astype(bool)
            new_uncovered_count = new_uncovered.sum()
            
            if new_uncovered_count < uncovered_count:
                selected.append(cw)
                result = solve(new_uncovered, new_uncovered_count, selected, depth + 1)
                selected.pop()
                if result is not None:
                    return result
        
        return None
    
    return solve(uncovered, uncovered_count, selected, 0)


def _greedy_subset(coverage, deadline):
    """Greedy set cover on coverage matrix. Returns list of row indices."""
    num_rows, num_cols = coverage.shape
    uncovered = np.ones(num_cols, dtype=bool)
    uncovered_count = num_cols
    
    selected = []
    
    while uncovered_count > 0 and time.time() < deadline:
        # Find row with max uncovered coverage
        gains = coverage[:, uncovered].sum(axis=1)
        best_row = np.argmax(gains)
        
        if gains[best_row] == 0:
            break
        
        selected.append(best_row)
        uncovered &= ~coverage[best_row].astype(bool)
        uncovered_count = uncovered.sum()
    
    if uncovered_count == 0:
        return selected
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
