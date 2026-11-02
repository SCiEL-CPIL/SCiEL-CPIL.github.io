"""
Football pool problem: ternary covering code K_3(7,1)
Minimize number of codewords of length 7 over {0,1,2} such that every word
is within Hamming distance 1 of some codeword.

Strategy:
1. Start with a reasonable initial covering (direct sum construction or greedy).
2. Use local search (greedy removal + addition) to minimize the code size.
3. Use simulated annealing or iterative improvement with time budget.

Key insight: A word w is covered by codeword c if Hamming distance <= radius.
A codeword c is "redundant" if every word it covers is also covered by another codeword.
We can greedily remove redundant codewords.

For efficiency:
- Precompute for each codeword the set of words it covers (itself + n*(q-1) neighbors).
- For n=7, q=3, radius=1: each codeword covers 1 + 7*2 = 15 words.
- Total space: 3^7 = 2187 words.
- We can represent the space as a 2187-element boolean array.
- For each codeword, mark its 15 covered words.
- To check if a codeword is redundant: temporarily unmark its words and check if any
  become uncovered. If none become uncovered, it's redundant.

This is O(|code| * 15) per removal check, and we can iterate until no more removals.

Then we can try adding codewords to cover remaining gaps (if any) and removing others.

Actually, since we start with a valid covering, we just need to remove redundant codewords.
The challenge is that removing one might make another redundant, so we iterate.

For a better search: after greedy removal, try swapping - remove a codeword and add
a different one to maintain coverage while potentially enabling further removals.
"""

import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(deadline)
    
    # General fallback: greedy set cover
    return _greedy_cover(q, n, radius, deadline)

def _word_to_int(word, q, n):
    """Convert a word (list of ints) to an integer index."""
    idx = 0
    for i, val in enumerate(word):
        idx = idx * q + val
    return idx

def _int_to_word(idx, q, n):
    """Convert integer index back to word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def _neighbors(word, q, n, radius=1):
    """Yield all words within Hamming distance <= radius from word."""
    # The word itself
    yield tuple(word)
    if radius >= 1:
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val != orig:
                    new_word = list(word)
                    new_word[pos] = val
                    yield tuple(new_word)

def _solve_ternary_7_1(deadline):
    """Specialized solver for q=3, n=7, radius=1."""
    q, n, radius = 3, 7, 1
    space_size = q ** n  # 2187
    
    # Precompute: for each integer index, its word and its neighbors
    all_words = [_int_to_word(i, q, n) for i in range(space_size)]
    
    # For each word index, precompute the list of indices it covers (itself + neighbors)
    cover_sets = []
    for i in range(space_size):
        word = all_words[i]
        covered = set()
        covered.add(i)
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val != orig:
                    new_word = list(word)
                    new_word[pos] = val
                    covered.add(_word_to_int(new_word, q, n))
        cover_sets.append(covered)
    
    # Initialize: use the direct sum construction as starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            code.append(h + list(tail))
    
    code_set = set(tuple(c) for c in code)
    
    # Track coverage: for each word in space, count how many codewords cover it
    coverage = np.zeros(space_size, dtype=np.int32)
    for c in code:
        for idx in cover_sets[_word_to_int(c, q, n)]:
            coverage[idx] += 1
    
    # Greedy removal: repeatedly remove a codeword if it's redundant
    # A codeword is redundant if all words it covers have coverage >= 2
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i, c in enumerate(list(code)):
            if time.time() > deadline:
                break
            c_idx = _word_to_int(c, q, n)
            # Check if all words covered by c have coverage >= 2
            redundant = True
            for idx in cover_sets[c_idx]:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove this codeword
                for idx in cover_sets[c_idx]:
                    coverage[idx] -= 1
                code.pop(i)
                changed = True
                break  # Restart scan from beginning
    
    # Now try to improve: local search with swaps
    # Try removing a codeword and adding a different one
    best_code = list(code)
    best_size = len(code)
    
    # Iterative improvement: try all possible single swaps
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        for i, c in enumerate(list(code)):
            if time.time() > deadline:
                break
            c_idx = _word_to_int(c, q, n)
            # Temporarily remove c
            for idx in cover_sets[c_idx]:
                coverage[idx] -= 1
            
            # Check if coverage is still valid (all >= 1)
            if np.all(coverage >= 1):
                # c was redundant, just remove it
                code.pop(i)
                improvement_found = True
                break
            else:
                # Find which words are now uncovered
                uncovered = set(np.where(coverage == 0)[0])
                # Try to find a codeword that covers all uncovered words
                # and doesn't add too much overhead
                # For each uncovered word, find codewords that cover it
                # We need a codeword that covers ALL uncovered words
                candidate_indices = None
                for u in uncovered:
                    # Find all codeword indices that cover word u
                    # This is expensive; instead, check each possible new codeword
                    if candidate_indices is None:
                        candidate_indices = set(range(space_size))
                    else:
                        # Intersect with codewords that cover u
                        # A codeword c' covers u if u is in cover_sets[c'_idx]
                        # This is O(space_size) per uncovered word, which is too slow
                        # Instead, just try a few random candidates
                        candidate_indices = None
                        break
                
                # Simpler approach: try adding a random codeword and see if it helps
                # Or try the best single codeword that covers the most uncovered
                if candidate_indices is None:
                    # Find codeword that covers most uncovered words
                    best_new = -1
                    best_count = -1
                    for new_idx in range(space_size):
                        if tuple(all_words[new_idx]) in code_set:
                            continue
                        count = 0
                        for u in uncovered:
                            if u in cover_sets[new_idx]:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_new = new_idx
                    
                    if best_new >= 0 and best_count == len(uncovered):
                        # Adding this covers all uncovered
                        new_word = all_words[best_new]
                        code.append(new_word)
                        code_set.add(tuple(new_word))
                        for idx in cover_sets[best_new]:
                            coverage[idx] += 1
                        improvement_found = True
                        break
                    else:
                        # Restore coverage
                        for idx in cover_sets[c_idx]:
                            coverage[idx] += 1
                else:
                    for idx in cover_sets[c_idx]:
                        coverage[idx] += 1
    
    # Final greedy removal pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i, c in enumerate(list(code)):
            if time.time() > deadline:
                break
            c_idx = _word_to_int(c, q, n)
            redundant = True
            for idx in cover_sets[c_idx]:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in cover_sets[c_idx]:
                    coverage[idx] -= 1
                code.pop(i)
                changed = True
                break
    
    # Verify coverage
    assert np.all(coverage >= 1), "Coverage invalid!"
    
    return [list(c) for c in code]

def _greedy_cover(q, n, radius, deadline):
    """General greedy set cover for the covering code problem."""
    space_size = q ** n
    all_words = [_int_to_word(i, q, n) for i in range(space_size)]
    
    # Precompute cover sets
    cover_sets = []
    for i in range(space_size):
        word = all_words[i]
        covered = set()
        # BFS up to radius
        from collections import deque
        visited = {tuple(word): 0}
        queue = deque([(tuple(word), 0)])
        while queue:
            w, dist = queue.popleft()
            covered.add(_word_to_int(list(w), q, n))
            if dist < radius:
                for pos in range(n):
                    orig = w[pos]
                    for val in range(q):
                        if val != orig:
                            new_w = list(w)
                            new_w[pos] = val
                            new_w_tuple = tuple(new_w)
                            if new_w_tuple not in visited:
                                visited[new_w_tuple] = dist + 1
                                queue.append((new_w_tuple, dist + 1))
        cover_sets.append(covered)
    
    # Greedy: start empty, add codewords that cover most uncovered words
    uncovered = set(range(space_size))
    code = []
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        for i in range(space_size):
            count = len(cover_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
            if time.time() > deadline:
                break
        if best_idx < 0 or best_count <= 0:
            break
        code.append(list(all_words[best_idx]))
        uncovered -= cover_sets[best_idx]
    
    # Greedy removal
    coverage = np.zeros(space_size, dtype=np.int32)
    for c in code:
        for idx in cover_sets[_word_to_int(c, q, n)]:
            coverage[idx] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i, c in enumerate(list(code)):
            if time.time() > deadline:
                break
            c_idx = _word_to_int(c, q, n)
            redundant = True
            for idx in cover_sets[c_idx]:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in cover_sets[c_idx]:
                    coverage[idx] -= 1
                code.pop(i)
                changed = True
                break
    
    return code
