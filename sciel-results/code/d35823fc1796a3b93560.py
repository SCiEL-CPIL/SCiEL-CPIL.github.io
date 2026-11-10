import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Ternary covering code K_3(8,1) via structural pair-merge local search
    on the tetracode direct-sum seed, plus greedy hole repair.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback greedy (rare)
        N = q ** n
        powers = [q ** i for i in range(n)]
        neigh = []
        for w in range(N):
            nb = [w]
            for i in range(n):
                d = (w // powers[i]) % q
                for nd in range(q):
                    if nd != d:
                        nb.append(w + (nd - d) * powers[i])
            neigh.append(nb)
        uncovered = set(range(N))
        code = []
        while uncovered:
            best_w, best_gain = -1, -1
            for w in range(N):
                g = sum(1 for x in neigh[w] if x in uncovered)
                if g > best_gain:
                    best_gain, best_w = g, w
            if best_gain <= 0:
                break
            code.append(best_w)
            for x in neigh[best_w]:
                uncovered.discard(x)
        out = []
        for w in code:
            word = []
            t = w
            for _ in range(n):
                word.append(t % q)
                t //= q
            out.append(word)
        return out

    N = 3 ** 8  # 6561
    powers = [3 ** i for i in range(n)]
    deadline = time.time() + 113.0  # safety margin under 150s kill

    # --- Precompute neighborhood (17 ints each) and membership bitsets ---
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            neighbors[w, idx] = w + ((d + 1) % 3 - d) * powers[i]
            idx += 1
            neighbors[w, idx] = w + ((d + 2) % 3 - d) * powers[i]
            idx += 1

    # Coverage bitsets: cov[w] is a 6561-bit array (103 uint64 words)
    n64 = (N + 63) // 64
    cov = np.zeros((N, n64), dtype=np.uint64)
    for w in range(N):
        for nb in neighbors[w]:
            cov[w, nb >> 6] |= (1 << (nb & 63))

    # --- Build tetracode direct-sum seed (486 codewords) ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = list(h) + list(tail)
            w_int = sum(word[i] * powers[i] for i in range(n))
            seed.append(w_int)
    best_code = seed[:]
    best_size = len(best_code)

    def is_covered(code):
        """Return True if the union of neighborhoods of `code` covers all N words."""
        union = np.zeros(n64, dtype=np.uint64)
        for w in code:
            union |= cov[w]
        for b in range(n64):
            if union[b] != 0xFFFFFFFFFFFFFFFF:
                return False
        return True

    # --- Greedy add: cover all currently-uncovered words, return list of added codewords ---
    def greedy_add(code):
        union = np.zeros(n64, dtype=np.uint64)
        for w in code:
            union |= cov[w]
        # uncovered bitset
        uncov = np.zeros(n64, dtype=np.uint64)
        for b in range(n64):
            uncov[b] = (~union[b]) & 0xFFFFFFFFFFFFFFFF
        # mask off bits beyond N in last word
        extra = n64 * 64 - N
        if extra > 0:
            uncov[-1] &= (1 << (64 - extra)) - 1
        added = []
        while True:
            if not uncov.any():
                break
            # For each candidate, gain = popcount(cov[w] & uncov)
            best_w, best_gain = -1, 0
            for w in range(N):
                g = 0
                cw = cov[w]
                for b in range(n64):
                    g += np.uint64(cw[b] & uncov[b]).view(np.uint64).item()
                    # use int popcount
                # recompute properly with popcount
                g = sum(bin(int(cw[b] & uncov[b])).count('1') for b in range(n64))
                if g > best_gain:
                    best_gain, best_w = g, w
            if best_gain <= 0:
                break
            added.append(best_w)
            for b in range(n64):
                uncov[b] &= ~cov[best_w, b]
        return added

    # --- Pair-merge local search on the seed ---
    # A "merge" of two seed codewords c1,c2 (same tail) whose tetracode heads differ
    # in exactly one coordinate is valid: the merged head covers both heads' 4-nbhds.
    # We precompute, for each tail, the 9 heads present and which pairs are mergeable.
    rng = random.Random(12345)

    # Group seed by tail (last 4 coords)
    from collections import defaultdict
    by_tail = defaultdict(list)
    for w in seed:
        tail = tuple((w // powers[i]) % 3 for i in range(4, n))
        head = tuple((w // powers[i]) % 3 for i in range(4))
        by_tail[tail].append((w, head))

    # For a given tail, given the set of active heads, find all mergeable pairs
    # (heads at Hamming distance 1) and the merged head.
    def heads_of(w):
        return tuple((w // powers[i]) % 3 for i in range(4))

    def make_word(head, tail):
        word = list(head) + list(tail)
        return sum(word[i] * powers[i] for i in range(n))

    def try_merge_pass(code_set, tail):
        """Attempt to merge pairs within one tail group; return (new_code_set, improved)."""
        active_heads = {}
        for w in code_set:
            t = tuple((w // powers[i]) % 3 for i in range(4, n))
            if t != tail:
                continue
            active_heads[w] = heads_of(w)
        if len(active_heads) < 2:
            return code_set, False
        changed = False
        new_set = set(code_set)
        # Try all pairs
        ws = list(active_heads.keys())
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                w1, w2 = ws[i], ws[j]
                if w1 not in new_set or w2 not in new_set:
                    continue
                h1, h2 = active_heads[w1], active_heads[w2]
                # Hamming distance of heads
                dist = sum(1 for a, b in zip(h1, h2) if a != b)
                if dist != 1:
                    continue
                # merged head: at the differing coordinate, the third symbol
                merged = list(h1)
                for k in range(4):
                    if h1[k] != h2[k]:
                        merged[k] = (3 - h1[k] - h2[k]) % 3
                # Verify merged head is a valid tetracode word
                a, b, c, d = merged
                if (a + b) % 3 != c or (a + 2 * b) % 3 != d:
                    continue
                merged_w = make_word(tuple(merged), tail)
                # Only merge if merged word is NOT already in the set (else pure removal)
                # and the merge actually reduces count: we remove w1,w2 and add merged_w
                if merged_w in new_set:
                    # pure removal possible
                    new_set.discard(w1)
                    new_set.discard(w2)
                    changed = True
                else:
                    new_set.discard(w1)
                    new_set.discard(w2)
                    new_set.add(merged_w)
                    changed = True
        if changed:
            return new_set, True
        return code_set, False

    # --- Main local-search loop ---
    all_tails = list(by_tail.keys())
    while time.time() < deadline:
        code_set = set(best_code)
        improved_this_round = False
        # Shuffle tail order for diversity
        tail_order = all_tails[:]
        rng.shuffle(tail_order)
        for tail in tail_order:
            if time.time() >= deadline:
                break
            new_set, ch = try_merge_pass(code_set, tail)
            if ch:
                # Check coverage of the new set
                if is_covered(new_set):
                    code_set = new_set
                    improved_this_round = True
                # else: revert (do not apply)
        if improved_this_round:
            new_list = list(code_set)
            if len(new_list) < best_size:
                best_size = len(new_list)
                best_code = new_list[:]
                if best_size <= 403:
                    break
        else:
            # No structural merge improved; try a randomized perturbation:
            # pick a random tail, randomly remove one redundant codeword, repair.
            if time.time() < deadline - 2.0:
                tail = rng.choice(all_tails)
                members = [w for w in best_code
                           if tuple((w // powers[i]) % 3 for i in range(4, n)) == tail]
                if len(members) > 1:
                    victim = rng.choice(members)
                    cand = [w for w in best_code if w != victim]
                    if is_covered(cand):
                        best_code = cand
                        best_size = len(cand)
                        continue
                    # Repair: greedy add
                    added = greedy_add(cand)
                    repaired = cand + added
                    if len(repaired) < best_size:
                        best_code = repaired
                        best_size = len(repaired)

    # Final safety: ensure coverage
    if not is_covered(best_code):
        added = greedy_add(best_code)
        best_code = best_code + added

    # Convert to list-of-lists (coordinate order 0..n-1)
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        final_code.append(word)
    return final_code
