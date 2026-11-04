"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0  # safety margin before the 150s kill

    # ---- mapping helpers ----
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * q + x
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    total = q ** n

    # ---- precompute radius balls for every word ----
    neighbor_lists = []
    for i in range(total):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
            w[pos] = orig
        neighbor_lists.append(ball)

    # ---- small-n fallback (exhaustive greedy) ----
    if n <= 4:
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        for w in all_words:
            if not uncovered:
                break
            if w in uncovered:
                code.append(list(w))
                for u in list(uncovered):
                    if sum(1 for i in range(n) if u[i] != w[i]) <= radius:
                        uncovered.remove(u)
        return code

    # ---- initial code: direct sum of the perfect ternary tetracode ----
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_tails = list(itertools.product(range(q), repeat=n - 4))
        code = [h + list(t) for h in tetracode for t in base_tails]
    else:
        # Fallback: greedy initial code
        uncovered = set(range(total))
        code = []
        for i in range(total):
            if not uncovered:
                break
            if i in uncovered:
                code.append(int_to_word(i))
                for nb in neighbor_lists[i]:
                    if nb in uncovered:
                        uncovered.discard(nb)

    code_indices = [word_to_int(c) for c in code]
    code_set = set(code_indices)

    counts = np.zeros(total, dtype=np.int32)
    for ci in code_indices:
        for nb in neighbor_lists[ci]:
            counts[nb] += 1

    # ---- greedy removal ----
    while time.time() < deadline - 5.0:
        found = False
        for idx in range(len(code_indices)):
            ci = code_indices[idx]
            ball = neighbor_lists[ci]
            if np.all(counts[ball] >= 2):
                for nb in ball:
                    counts[nb] -= 1
                code_indices.pop(idx)
                code_set.discard(ci)
                found = True
                break
        if not found:
            break

    # ---- swap local search with slack-based scoring ----
    rng = random.Random(12345)

    def try_swaps():
        nonlocal counts
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            order = list(range(len(code_indices)))
            rng.shuffle(order)
            for i in order:
                if time.time() >= deadline - 5.0:
                    return
                ci = code_indices[i]
                ball_i = neighbor_lists[ci]
                for j in range(len(code_indices)):
                    if j == i:
                        continue
                    cj = code_indices[j]
                    ball_j = neighbor_lists[cj]
                    # Check if removing i and adding j still covers all
                    # Words only covered by i (count==1 in ball_i) must be in ball_j
                    words_only_i = [nb for nb in ball_i if counts[nb] == 1]
                    if words_only_i:
                        # These words must be covered by cj's ball
                        uncovered_by_swap = False
                        for w in words_only_i:
                            if w not in ball_j:
                                uncovered_by_swap = True
                                break
                        if uncovered_by_swap:
                            continue
                    # Compute slack improvement
                    union_set = set(ball_i) | set(ball_j)
                    old_slack = sum(max(counts[w] - 1, 0) for w in union_set)
                    # New counts after swap: remove i, add j
                    new_counts_check = counts.copy()
                    for nb in ball_i:
                        new_counts_check[nb] -= 1
                    for nb in ball_j:
                        new_counts_check[nb] += 1
                    if np.any(new_counts_check < 1):
                        continue
                    new_slack = sum(max(new_counts_check[w] - 1, 0) for w in union_set)
                    if new_slack < old_slack:
                        # Accept the swap
                        for nb in ball_i:
                            counts[nb] -= 1
                        for nb in ball_j:
                            counts[nb] += 1
                        code_indices[i] = cj
                        improved = True
                        break
                if improved:
                    break

    try_swaps()

    # ---- 2-remove-1-add local search (key operation to reduce code size) ----
    def try_two_remove_one_add():
        nonlocal counts
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            m = len(code_indices)
            # Precompute set of each ball for fast lookup
            ball_sets = [set(neighbor_lists[ci]) for ci in code_indices]
            for i in range(m):
                if time.time() >= deadline - 10.0:
                    return
                for j in range(i + 1, m):
                    if time.time() >= deadline - 10.0:
                        return
                    # Compute words that would become uncovered if we remove i and j
                    # A word is uncovered if its count is 1 and it's in ball_i or ball_j
                    # More precisely: words where count == 1 and (word in ball_i XOR word in ball_j)
                    # Actually: words where after removal, count becomes 0
                    # count_after(w) = counts[w] - (1 if w in ball_i else 0) - (1 if w in ball_j else 0)
                    # We need count_after(w) >= 1 for all w
                    # So we need: for all w in ball_i, counts[w] - (1 if w in ball_j else 0) >= 1
                    # i.e., for all w in ball_i: if w in ball_j then counts[w] >= 2, else counts[w] >= 1
                    # And for all w in ball_j: if w in ball_i then counts[w] >= 2, else counts[w] >= 1
                    # Simpler: compute the set of words that would have count 0 after removal
                    uncovered = set()
                    for w in ball_sets[i]:
                        if counts[w] == 1 and w not in ball_sets[j]:
                            uncovered.add(w)
                    for w in ball_sets[j]:
                        if counts[w] == 1 and w not in ball_sets[i]:
                            uncovered.add(w)
                    if not uncovered:
                        # We can remove both i and j without adding anything
                        # This is already handled by greedy, but let's do it
                        for idx in [j, i]:
                            ci = code_indices[idx]
                            for nb in neighbor_lists[ci]:
                                counts[nb] -= 1
                            code_indices.pop(idx)
                            ball_sets.pop(idx)
                        improved = True
                        break
                    # Try to find a single codeword c such that ball(c) covers all words in `uncovered`
                    # and adding c doesn't create any issue (it won't, since we're adding)
                    # c must not already be in code_indices
                    # Optimization: only consider c whose ball intersects all words in uncovered
                    # A necessary condition: for each w in uncovered, c must be within distance radius of w
                    # i.e., c must be in the intersection of all balls centered at words in uncovered
                    # Precompute: for each word w, the set of codewords within distance radius of w
                    # Actually, let's just iterate but with early termination
                    found = False
                    # Build candidate set: intersection of balls of all words in uncovered
                    # A codeword c covers w if dist(c,w) <= radius, i.e., c in ball(w)
                    # So c must be in intersect(ball(w) for w in uncovered)
                    if len(uncovered) == 1:
                        candidates = set(neighbor_lists[list(uncovered)[0]])
                    else:
                        candidates = set(neighbor_lists[list(uncovered)[0]])
                        for w in list(uncovered)[1:]:
                            candidates &= set(neighbor_lists[w])
                            if not candidates:
                                break
                    candidates -= code_set  # exclude existing codewords
                    for c in candidates:
                        ball_c = set(neighbor_lists[c])
                        # Verify (should be guaranteed by construction)
                        if all(w in ball_c for w in uncovered):
                            # Accept: remove i and j, add c
                            for idx in [j, i]:
                                ci = code_indices[idx]
                                for nb in neighbor_lists[ci]:
                                    counts[nb] -= 1
                                code_indices.pop(idx)
                                ball_sets.pop(idx)
                            code_indices.append(c)
                            ball_sets.append(ball_c)
                            code_set.add(c)
                            for nb in neighbor_lists[c]:
                                counts[nb] += 1
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

    try_two_remove_one_add()

    # ---- Another round of greedy removal after 2-1 operations ----
    while time.time() < deadline - 5.0:
        found = False
        for idx in range(len(code_indices)):
            ci = code_indices[idx]
            ball = neighbor_lists[ci]
            if np.all(counts[ball] >= 2):
                for nb in ball:
                    counts[nb] -= 1
                code_indices.pop(idx)
                code_set.discard(ci)
                found = True
                break
        if not found:
            break

    # ---- Second pass of 2-remove-1-add after greedy removal ----
    if time.time() < deadline - 15.0:
        try_two_remove_one_add()
        # Final greedy removal
        while time.time() < deadline - 5.0:
            found = False
            for idx in range(len(code_indices)):
                ci = code_indices[idx]
                ball = neighbor_lists[ci]
                if np.all(counts[ball] >= 2):
                    for nb in ball:
                        counts[nb] -= 1
                    code_indices.pop(idx)
                    code_set.discard(ci)
                    found = True
                    break
            if not found:
                break

    # ---- final greedy removal pass after swaps ----
    while time.time() < deadline - 5.0:
        found = False
        for idx in range(len(code_indices)):
            ci = code_indices[idx]
            ball = neighbor_lists[ci]
            if np.all(counts[ball] >= 2):
                for nb in ball:
                    counts[nb] -= 1
                code_indices.pop(idx)
                code_set.discard(ci)
                found = True
                break
        if not found:
            break

    # ---- Branch-and-bound: find minimum-size subset of code_indices that covers all ----
    m = len(code_indices)
    if m > 200 or time.time() > deadline - 30.0:
        # Too many codewords for B&B; skip
        return [int_to_word(ci) for ci in code_indices]
    
    # Precompute which words are covered by each codeword (as bitsets)
    # Use numpy boolean arrays for coverage
    cover_masks = []
    for ci in code_indices:
        mask = np.zeros(total, dtype=bool)
        mask[neighbor_lists[ci]] = True
        cover_masks.append(mask)
    
    # Current coverage
    current_cover = np.zeros(total, dtype=bool)
    for cm in cover_masks:
        current_cover |= cm
    
    # B&B: iteratively try to remove each codeword
    # Sort by "criticality": codewords that cover fewer unique words first (easier to remove)
    # Actually, for minimum cover, we want to remove redundant codewords.
    # A codeword is redundant if its ball is subset of the union of others' balls.
    
    # Precompute pairwise coverage relationships
    # For efficiency, use the count array
    # A codeword i can be removed if for all words in ball(i), count[word] >= 2
    
    # Use a more aggressive B&B:
    # 1. Find all removable codewords (ball subset of others)
    # 2. Remove the one whose removal causes the least "damage" (minimizes max count)
    # 3. Repeat until no removable codeword
    
    # But this is what greedy removal already does. For true B&B, we need to consider
    # combinations. Let's do a limited-depth B&B:
    # Try all single removals, then all double removals, etc., up to depth 3.
    
    def can_remove(indices_to_remove):
        """Check if removing these codewords still leaves full coverage."""
        if not indices_to_remove:
            return True
        remove_set = set(indices_to_remove)
        new_counts = counts.copy()
        for idx in remove_set:
            ci = code_indices[idx]
            for nb in neighbor_lists[ci]:
                new_counts[nb] -= 1
        return np.all(new_counts >= 1)
    
    # Depth-limited B&B: try to remove up to 3 codewords at a time
    # This is expensive, so limit the search space
    if time.time() < deadline - 10.0:
        # Try single removals (already done by greedy, but verify)
        for idx in range(m):
            if time.time() > deadline - 10.0:
                break
            if can_remove([idx]):
                # Remove it
                ci = code_indices[idx]
                for nb in neighbor_lists[ci]:
                    counts[nb] -= 1
                code_indices.pop(idx)
                m -= 1
                # Recompute cover_masks
                cover_masks.pop(idx)
                break  # Restart the loop
        
        # Try double removals
        if time.time() < deadline - 10.0 and m > 0:
            for i in range(m):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline - 10.0:
                        break
                    if can_remove([i, j]):
                        # Remove both
                        for idx in [j, i]:  # Remove in reverse order
                            ci = code_indices[idx]
                            for nb in neighbor_lists[ci]:
                                counts[nb] -= 1
                            code_indices.pop(idx)
                            cover_masks.pop(idx)
                        m -= 2
                        break
                if m < len(code_indices):
                    break  # Something was removed, restart
        
        # Try triple removals
        if time.time() < deadline - 10.0 and m > 0:
            for i in range(m):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline - 10.0:
                        break
                    for k in range(j + 1, m):
                        if time.time() > deadline - 10.0:
                            break
                        if can_remove([i, j, k]):
                            # Remove all three
                            for idx in [k, j, i]:
                                ci = code_indices[idx]
                                for nb in neighbor_lists[ci]:
                                    counts[nb] -= 1
                                code_indices.pop(idx)
                                cover_masks.pop(idx)
                            m -= 3
                            break
                    if m < len(code_indices):
                        break
                if m < len(code_indices):
                    break
    
    # Final verification and return
    return [int_to_word(ci) for ci in code_indices]
# EVOLVE-BLOCK-END
