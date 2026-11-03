"""Combined approach for K_3(6,1):
- Structural init: greedy independent set on distance-2 graph
- Refinement: coverage-count based redundancy pruning + perturbation/repair
- Optional: time-bounded swap local search using marginal coverage heuristics
"""
import itertools
import random
import time

def _build_space_indexed(q, n):
    total = q ** n
    # words: list of tuples in lex order (same as itertools.product)
    words = list(itertools.product(range(q), repeat=n))
    # index helpers
    pow_q = [1] * (n + 1)
    for i in range(1, n + 1):
        pow_q[i] = pow_q[i - 1] * q
    return words, total, pow_q

def _precompute_neighbors(q, n, radius):
    total = q ** n
    pow_q = [1] * (n + 1)
    for i in range(1, n + 1):
        pow_q[i] = pow_q[i - 1] * q

    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        if radius >= 1:
            for i in range(n):
                digit = (idx // pow_q[i]) % q
                for nd in range(q):
                    if nd == digit:
                        continue
                    nidx = idx + (nd - digit) * pow_q[i]
                    neigh[idx].append(nidx)
                    neigh_set[idx].add(nidx)
    return neigh, neigh_set, total, pow_q

def _precompute_dist2_adj(q, n):
    total = q ** n
    pow_q = [1] * (n + 1)
    for i in range(1, n + 1):
        pow_q[i] = pow_q[i - 1] * q
    dist2_adj = [set() for _ in range(total)]
    for idx in range(total):
        for i in range(n):
            digit_i = (idx // pow_q[i]) % q
            for j in range(i + 1, n):
                digit_j = (idx // pow_q[j]) % q
                for nd_i in range(q):
                    if nd_i == digit_i:
                        continue
                    for nd_j in range(q):
                        if nd_j == digit_j:
                            continue
                        nidx = idx + (nd_i - digit_i) * pow_q[i] + (nd_j - digit_j) * pow_q[j]
                        if nidx != idx:
                            dist2_adj[idx].add(nidx)
                            dist2_adj[nidx].add(idx)
    return dist2_adj

def _coverage_counts(code_idx, neigh, total):
    counts = [0] * total
    for ci in code_idx:
        for w in neigh[ci]:
            counts[w] += 1
    return counts

def _verify(code_idx, neigh, total):
    counts = [0] * total
    for ci in code_idx:
        for w in neigh[ci]:
            counts[w] += 1
    for c in counts:
        if c == 0:
            return False
    return True

def _local_search_prune(code_idx, neigh, total):
    # Repeatedly delete any redundant codeword until fixed point.
    counts = _coverage_counts(list(code_idx), neigh, total)
    code = sorted(set(code_idx))
    changed = True
    while changed:
        changed = False
        new_code = []
        for ci in code:
            redundant = True
            for w in neigh[ci]:
                if counts[w] <= 1:
                    redundant = False
                    break
            if redundant:
                for w in neigh[ci]:
                    counts[w] -= 1
                changed = True
            else:
                new_code.append(ci)
        code = new_code
    return code

def _greedy_from_uncovered(uncovered_set, neigh, total):
    import numpy as np
    chosen = []
    # Build bitset representation of neighbors for fast intersection
    # Each neigh[c] is a list of indices; we need to count overlaps with uncov
    # For efficiency with 729 words, use numpy
    uncov_arr = np.zeros(total, dtype=np.int8)
    for w in uncovered_set:
        uncov_arr[w] = 1
    uncov_count = len(uncovered_set)
    
    # Precompute neighbor arrays as numpy
    neigh_arrays = []
    for c in range(total):
        neigh_arrays.append(np.array(neigh[c], dtype=np.int32))
    
    while uncov_count > 0:
        best_c = -1
        best_cnt = -1
        for c in range(total):
            # Count how many uncovered words are in neigh[c]
            cnt = int(np.sum(uncov_arr[neigh_arrays[c]]))
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c
        if best_cnt <= 0:
            break
        chosen.append(best_c)
        # Update uncov_arr
        for w in neigh[best_c]:
            if uncov_arr[w] == 1:
                uncov_arr[w] = 0
                uncov_count -= 1
    return chosen

def _repair_and_local_search(code_idx, uncovered, neigh, total):
    added = _greedy_from_uncovered(uncovered, neigh, total)
    combined = list(code_idx) + added
    return _local_search_prune(combined, neigh, total)

def _swap_local_search(code_idx, neigh, neigh_set, total, deadline, rng, max_time=10.0):
    # Time-bounded swap: replace a center with another that better covers singly-covered words.
    code = list(code_idx)
    counts = _coverage_counts(code, neigh, total)
    start = time.time()
    improved = True
    while improved and (time.time() < deadline - max_time):
        improved = False
        order = list(range(len(code)))
        rng.shuffle(order)
        for pos in order:
            if time.time() > deadline - max_time:
                break
            old_c = code[pos]
            critical = []
            for w in neigh[old_c]:
                if counts[w] == 1:
                    critical.append(w)
            if not critical:
                code.pop(pos)
                for w in neigh[old_c]:
                    counts[w] -= 1
                improved = True
                break
            # Try to find a replacement that covers all critical words
            best_new = -1
            best_score = -1
            candidates = list(range(total))
            rng.shuffle(candidates)
            for c in candidates:
                if c == old_c:
                    continue
                # Check if c is already in code (excluding old_c)
                if any(c == code[j] for j in range(len(code)) if j != pos):
                    continue
                score = 0
                for w in critical:
                    if w in neigh_set[c]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_new = c
                if best_score == len(critical):
                    break
            if best_new != -1 and best_score == len(critical):
                for w in neigh[old_c]:
                    counts[w] -= 1
                for w in neigh[best_new]:
                    counts[w] += 1
                code[pos] = best_new
                improved = True
                break
        # After a pass, prune redundancies
        code = _local_search_prune(code, neigh, total)
        counts = _coverage_counts(code, neigh, total)
    return code

def _two_for_one_search(code_idx, neigh, neigh_set, total, deadline, rng, max_time=5.0):
    """Try to replace two codewords with one that covers their union better."""
    code = list(code_idx)
    counts = _coverage_counts(code, neigh, total)
    start = time.time()
    improved = True
    while improved and (time.time() < deadline - max_time):
        improved = False
        k = len(code)
        if k < 2:
            break
        # Try random pairs
        for _ in range(50):
            if time.time() > deadline - max_time:
                break
            i, j = rng.sample(range(k), 2)
            ci, cj = code[i], code[j]
            # Words only covered by ci or cj (i.e., would be uncovered if both removed)
            critical = []
            for w in range(total):
                if counts[w] == 1 and (w in neigh_set[ci] or w in neigh_set[cj]):
                    critical.append(w)
            if not critical:
                # Can remove both
                # Remove j first (higher index)
                for w in neigh[cj]:
                    counts[w] -= 1
                code.pop(j)
                for w in neigh[ci]:
                    counts[w] -= 1
                code.pop(i)
                improved = True
                break
            # Try to find a single replacement covering all critical
            best_new = -1
            best_score = -1
            candidates = list(range(total))
            rng.shuffle(candidates)
            for c in candidates:
                if c == ci or c == cj:
                    continue
                if any(c == code[m] for m in range(len(code)) if m != i and m != j):
                    continue
                score = 0
                for w in critical:
                    if w in neigh_set[c]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_new = c
                if best_score == len(critical):
                    break
            if best_new != -1 and best_score == len(critical):
                # Replace both with best_new
                for w in neigh[ci]:
                    counts[w] -= 1
                for w in neigh[cj]:
                    counts[w] -= 1
                for w in neigh[best_new]:
                    counts[w] += 1
                # Remove j, then i, add best_new
                code.pop(j)
                code.pop(i)
                code.append(best_new)
                improved = True
                break
        if improved:
            code = _local_search_prune(code, neigh, total)
            counts = _coverage_counts(code, neigh, total)
    return code

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 110.0  # safety margin under 120/150s limits

    # Build space and precompute
    words, total, pow_q = _build_space_indexed(q, n)
    neigh, neigh_set, _, _ = _precompute_neighbors(q, n, radius)
    dist2_adj = _precompute_dist2_adj(q, n)

    # Seed: tetracode direct sum -> 81 words (if n >= 4)
    seed_idx = []
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                w = h + list(tail)
                # Convert to index
                idx = 0
                for i, digit in enumerate(w):
                    idx += digit * (q ** i)
                seed_idx.append(idx)
    else:
        # Fallback: all words
        seed_idx = list(range(total))

    best_code = sorted(set(seed_idx))
    best_size = len(best_code)

    # Structural init: greedy independent set on distance-2 graph
    # This avoids pairs at Hamming distance 2, which are candidates for removal
    rng = random.Random(20240517)
    remaining = set(range(total))
    code_is = []
    while remaining and time.time() < DEADLINE - 30:
        # Pick vertex with minimum degree in remaining graph
        best_v = min(remaining, key=lambda v: len(dist2_adj[v] & remaining))
        code_is.append(best_v)
        remaining.discard(best_v)
        for u in dist2_adj[best_v]:
            if u in remaining:
                remaining.discard(u)
    # Repair to full coverage
    counts = _coverage_counts(code_is, neigh, total)
    uncovered = set()
    for w in range(total):
        if counts[w] == 0:
            uncovered.add(w)
    if uncovered:
        code_is = _repair_and_local_search(code_is, uncovered, neigh, total)
    # Local search prune
    code_is = _local_search_prune(code_is, neigh, total)
    if len(code_is) < best_size:
        best_size = len(code_is)
        best_code = code_is

    # Pass 2: full greedy from scratch (random tie-breaks) + local search, multiple times
    for trial in range(30):
        if time.time() > DEADLINE - 30:
            break
        # build greedy order over centers by random weight on coverage
        centers = list(range(total))
        rng.shuffle(centers)
        uncov = set(range(total))
        code = []
        # greedy: repeatedly pick center covering most uncovered (tie by random shuffle already)
        while uncov and len(code) < 90:
            best_c = -1
            best_cnt = -1
            for c in centers:
                if not uncov:
                    break
                cnt = 0
                for w in neigh[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            code.append(best_c)
            for w in neigh[best_c]:
                uncov.discard(w)
            # rebuild centers order? cheaper to just re-shuffle remaining cheaply by re-listing
            centers = [c for c in centers if c != best_c]
            rng.shuffle(centers)
        if uncov:
            continue
        code2 = _local_search_prune(code, neigh, total)
        if len(code2) < best_size:
            best_size = len(code2)
            best_code = code2

    # Pass 3: perturbation from best_code: randomly drop one center, repair, local search
    iterations = 0
    while time.time() < DEADLINE - 25:
        iterations += 1
        code = list(best_code)
        k = len(code)
        if k < 2:
            break
        # pick a random center to drop
        drop_idx = rng.randrange(k)
        new_code = [c for j, c in enumerate(code) if j != drop_idx]
        counts = _coverage_counts(new_code, neigh, total)
        uncovered = set()
        for w in range(total):
            if counts[w] == 0:
                uncovered.add(w)
        repaired = _repair_and_local_search(new_code, uncovered, neigh, total)
        if len(repaired) < best_size:
            best_size = len(repaired)
            best_code = repaired

        # every few iterations try dropping two centers
        if iterations % 3 == 0 and k >= 2:
            d1 = rng.randrange(k)
            d2 = rng.randrange(k)
            if d1 == d2:
                d2 = (d2 + 1) % k
            new_code = [c for j, c in enumerate(code) if j != d1 and j != d2]
            counts = _coverage_counts(new_code, neigh, total)
            uncovered = set()
            for w in range(total):
                if counts[w] == 0:
                    uncovered.add(w)
            repaired = _repair_and_local_search(new_code, uncovered, neigh, total)
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired

        # every 5 iterations try dropping three centers
        if iterations % 5 == 0 and k >= 3:
            drops = rng.sample(range(k), 3)
            drop_set = set(drops)
            new_code = [c for j, c in enumerate(code) if j not in drop_set]
            counts = _coverage_counts(new_code, neigh, total)
            uncovered = set()
            for w in range(total):
                if counts[w] == 0:
                    uncovered.add(w)
            repaired = _repair_and_local_search(new_code, uncovered, neigh, total)
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired

    # Pass 4: two-for-one search on best_code
    if time.time() < DEADLINE - 15:
        two_for_one = _two_for_one_search(best_code, neigh, neigh_set, total, DEADLINE, rng, max_time=5.0)
        if len(two_for_one) < best_size:
            best_size = len(two_for_one)
            best_code = two_for_one

    # Pass 5: swap local search on best_code
    if time.time() < DEADLINE - 10:
        swapped = _swap_local_search(best_code, neigh, neigh_set, total, DEADLINE, rng, max_time=5.0)
        if len(swapped) < best_size:
            best_size = len(swapped)
            best_code = swapped

    # Final: ensure best_code is valid (it should be by construction), return it
    if not _verify(best_code, neigh, total):
        # fallback to seed
        best_code = sorted(set(seed_idx))

    # Convert indices -> lists of ints
    result = [list(words[i]) for i in sorted(set(best_code))]
    return result
