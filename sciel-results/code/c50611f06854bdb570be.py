import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety buffer before 120s kill

    N = q ** n

    # --- Build neighbor masks and inverse mapping ---
    # Little-endian: digit j has weight q**j
    weights = [q ** j for j in range(n)]

    # Precompute digits for every word
    # digits[i, j] = j-th digit of word i
    digits = np.zeros((N, n), dtype=np.int32)
    for i in range(N):
        x = i
        for j in range(n):
            digits[i, j] = x % q
            x //= q

    # Build neighbor masks (Python int bitmasks) and neighbor lists
    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]

    if radius == 1:
        for c_idx in range(N):
            mask = 1 << c_idx
            nbrs = [c_idx]
            w = digits[c_idx]
            for j in range(n):
                base = w[j]
                wt = weights[j]
                for d in range(q):
                    if d != base:
                        nidx = c_idx + (d - base) * wt
                        nbrs.append(nidx)
                        mask |= (1 << nidx)
            neighbor_masks[c_idx] = mask
            neighbor_lists[c_idx] = nbrs
    else:
        # General radius (N is small for the actual task)
        from itertools import combinations, product
        for c_idx in range(N):
            mask = 1 << c_idx
            nbrs = [c_idx]
            w = digits[c_idx].copy()
            for dist in range(1, radius + 1):
                for positions in combinations(range(n), dist):
                    for vals in product(range(q), repeat=dist):
                        if all(vals[k] == w[positions[k]] for k in range(dist)):
                            continue
                        new_w = w.copy()
                        for k, pos in enumerate(positions):
                            new_w[pos] = vals[k]
                        nidx = sum(new_w[j] * weights[j] for j in range(n))
                        nbrs.append(nidx)
                        mask |= (1 << nidx)
            neighbor_masks[c_idx] = mask
            neighbor_lists[c_idx] = nbrs

    # Inverse: for each word, which codewords cover it
    word_to_coders = [[] for _ in range(N)]
    for c in range(N):
        for w in neighbor_lists[c]:
            word_to_coders[w].append(c)

    # --- Greedy strategies ---

    def greedy_criticality(seed=0):
        """Greedy prioritizing hardest uncovered word (fewest coders),
        then max new coverage. Optional random tie-breaking via seed."""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        rng = np.random.default_rng(seed)

        while uncovered:
            if time.time() > deadline:
                break
            # Find uncovered word with min criticality
            unc_idx = np.nonzero(uncovered)[0]
            if len(unc_idx) == 0:
                break
            # Sample for speed if many uncovered
            if len(unc_idx) > 200:
                sample = rng.choice(unc_idx, size=50, replace=False)
            else:
                sample = unc_idx
            min_crit = N + 1
            target = -1
            for w in sample:
                if not (uncovered >> w) & 1:
                    continue
                crit = len(word_to_coders[w])
                if crit < min_crit:
                    min_crit = crit
                    target = w
                    if min_crit == 1:
                        break
            if target == -1:
                # fallback: any uncovered
                target = int(unc_idx[0])
            # Pick best codeword covering target
            best_c = -1
            best_new = -1
            for c in word_to_coders[target]:
                if c in code_set:
                    continue
                new_count = (uncovered & neighbor_masks[c]).bit_count()
                if new_count > best_new:
                    best_new = new_count
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
        return code

    def greedy_safety_margin(seed=0):
        """Farthest-point greedy: pick word with max distance to nearest
        existing center for uniform density. Then local search."""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        rng = np.random.default_rng(seed)

        while uncovered:
            if time.time() > deadline:
                break
            # Find uncovered word with max "safety margin" = min distance to any codeword
            # Approximate: pick uncovered word that is farthest from all chosen codewords
            # For speed, sample uncovered words
            unc_idx = np.nonzero(uncovered)[0]
            if len(unc_idx) == 0:
                break
            if len(unc_idx) > 300:
                sample = rng.choice(unc_idx, size=100, replace=False)
            else:
                sample = unc_idx

            best_target = -1
            best_margin = -1
            for w in sample:
                if not (uncovered >> w) & 1:
                    continue
                # safety margin = min Hamming distance to any codeword
                # = 1 if covered by some codeword (radius 1), else larger
                # For radius 1: margin = 0 if covered, 1 if not
                # Use: is w in any codeword's sphere? If not, it's "far"
                # Cheaper: count how many codewords are at distance 1 from w
                # = len(word_to_coders[w]) but only count those in code_set
                margin = 0
                for c in word_to_coders[w]:
                    if c in code_set:
                        margin += 1
                        break
                # margin 0 means not covered -> high priority
                # We want MIN margin (hardest to cover)
                if margin < best_margin or best_target == -1:
                    best_margin = margin
                    best_target = w
                if best_margin == 0:
                    break

            if best_target == -1:
                best_target = int(unc_idx[0])

            # Pick best codeword covering target
            best_c = -1
            best_new = -1
            for c in word_to_coders[best_target]:
                if c in code_set:
                    continue
                new_count = (uncovered & neighbor_masks[c]).bit_count()
                if new_count > best_new:
                    best_new = new_count
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
        return code

    # --- Local search ---

    def local_search(code):
        """Redundant removal + swap-based improvement."""
        code = list(code)
        if not code:
            return code
        code_set = set(code)
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pass 1: removal
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                c = code[i]
                has_unique = False
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    code.pop(i)
                    code_set.discard(c)
                    improved = True
                    break

            # Pass 2: swap (remove c, add c2, then maybe remove another)
            if not improved:
                for i in range(len(code)):
                    if time.time() > deadline:
                        break
                    c = code[i]
                    newly_uncovered = [w for w in neighbor_lists[c] if cover_counts[w] == 1]
                    if not newly_uncovered:
                        continue  # would have been caught by removal
                    # Find c2 that covers all newly_uncovered
                    candidates = set()
                    for w in newly_uncovered:
                        for c2 in word_to_coders[w]:
                            if c2 not in code_set:
                                candidates.add(c2)
                    best_c2 = -1
                    best_gain = 0
                    for c2 in candidates:
                        # Check c2 covers all newly_uncovered
                        ok = True
                        for w in newly_uncovered:
                            if w not in neighbor_lists[c2]:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Count how many codewords become removable after swap
                        gain = 0
                        for c3 in code:
                            if c3 == c:
                                continue
                            removable = True
                            for w in neighbor_lists[c3]:
                                cc = cover_counts[w]
                                cc -= 1 if w in neighbor_lists[c] else 0
                                cc += 1 if w in neighbor_lists[c2] else 0
                                if cc <= 0:
                                    removable = False
                                    break
                            if removable:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c2 = c2
                    if best_c2 != -1 and best_gain >= 1:
                        # Apply swap
                        for w in neighbor_lists[c]:
                            cover_counts[w] -= 1
                        code.pop(i)
                        code_set.discard(c)
                        code.append(best_c2)
                        code_set.add(best_c2)
                        for w in neighbor_lists[best_c2]:
                            cover_counts[w] += 1
                        # Remove one newly-removable
                        for j in range(len(code)):
                            cj = code[j]
                            removable = True
                            for w in neighbor_lists[cj]:
                                if cover_counts[w] <= 1:
                                    removable = False
                                    break
                            if removable:
                                for w in neighbor_lists[cj]:
                                    cover_counts[w] -= 1
                                code.pop(j)
                                code_set.discard(cj)
                                break
                        improved = True
                        break
        return code

    # --- Main: run multiple greedy strategies, keep best ---
    best_code = greedy_criticality(seed=0)
    best_code = local_search(best_code)

    # Try safety-margin greedy
    if time.time() < deadline - 10:
        sm_code = greedy_safety_margin(seed=0)
        sm_code = local_search(sm_code)
        if len(sm_code) < len(best_code):
            best_code = sm_code

    # Randomized restarts with criticality greedy
    rng = np.random.default_rng(42)
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        rcode = greedy_criticality(seed=100 + attempt)
        rcode = local_search(rcode)
        if len(rcode) < len(best_code):
            best_code = rcode

    # Convert indices to base-q digit lists (big-endian)
    result = []
    for idx in best_code:
        word = [0] * n
        x = idx
        for j in range(n):
            word[n - 1 - j] = x % q
            x //= q
        result.append(word)

    return result
