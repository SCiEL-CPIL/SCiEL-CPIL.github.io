import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """Construct a ternary covering code K_3(6,1). Returns list of codewords."""
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin vs 120s / 150s kill

    # Fallback for non-target parameters
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    rng = random.Random(12345)

    # --- Precomputation ---
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    def word_to_idx_fast(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    # covers[i] = frozenset of word indices within distance 1 of word i
    covers = [None] * total_words
    for i, w in enumerate(all_words):
        cov = {i}
        for pos in range(n):
            ov = w[pos]
            for nv in range(3):
                if nv == ov:
                    continue
                lst = list(w)
                lst[pos] = nv
                cov.add(word_to_idx_fast(tuple(lst)))
                lst[pos] = ov
        covers[i] = frozenset(cov)

    # dist2 adjacency (undirected), words at Hamming distance exactly 2
    dist2 = [set() for _ in range(total_words)]
    for i in range(total_words):
        w1 = all_words[i]
        for j in range(i + 1, total_words):
            w2 = all_words[j]
            d = 0
            for k in range(n):
                if w1[k] != w2[k]:
                    d += 1
                    if d > 2:
                        break
            if d == 2:
                dist2[i].add(j)
                dist2[j].add(i)

    # covered_by[w] = frozenset of codeword indices that cover word w
    covered_by = [set() for _ in range(total_words)]
    for c in range(total_words):
        for w in covers[c]:
            covered_by[w].add(c)
    covered_by = [frozenset(s) for s in covered_by]

    # --- Initial cover: greedy independent set on distance-2 graph ---
    def build_independent_set(seed):
        r = random.Random(seed)
        remaining = set(range(total_words))
        indep = []
        while remaining:
            cands = r.sample(list(remaining), min(40, len(remaining)))
            best_v, best_deg = None, None
            for v in cands:
                deg = len(dist2[v] & remaining)
                if best_deg is None or deg < best_deg:
                    best_deg, best_v = deg, v
            indep.append(best_v)
            remaining.discard(best_v)
            remaining -= dist2[best_v]
        return indep

    best_iset = []
    for s in range(6):
        if time.time() > deadline - 35:
            break
        iset = build_independent_set(1000 + s * 7)
        if len(iset) < len(best_iset):
            best_iset = iset

    # Fallback: tetracode direct sum (81 words) if independent set is poor
    if len(best_iset) > 81 or time.time() > deadline - 25:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_iset = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                best_iset.append(word_to_idx[tuple(h + list(tail))])

    # Ensure the initial cover is valid (covers all words); add if needed
    def make_valid(active):
        active = set(active)
        cc = np.zeros(total_words, dtype=int)
        for idx in active:
            for w in covers[idx]:
                cc[w] += 1
        uncovered = set(np.where(cc == 0)[0])
        while uncovered and time.time() < deadline - 5:
            best_c, best_cnt = -1, 0
            for c in range(total_words):
                if c in active:
                    continue
                cnt = 0
                for w in covers[c]:
                    if w in uncovered:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt, best_c = cnt, c
            if best_c < 0 or best_cnt == 0:
                break
            active.add(best_c)
            for w in covers[best_c]:
                cc[w] += 1
                if cc[w] == 1:
                    uncovered.discard(w)
        return active

    active = make_valid(best_iset)

    cc = np.zeros(total_words, dtype=int)
    for idx in active:
        for w in covers[idx]:
            cc[w] += 1

    def is_redundant(idx):
        for w in covers[idx]:
            if cc[w] == 1:
                return False
        return True

    # --- CSP-style arc-consistency polish: iteratively drop non-essential words ---
    def ac_polish(active):
        active = set(active)
        cc = np.zeros(total_words, dtype=int)
        for idx in active:
            for w in covers[idx]:
                cc[w] += 1
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for idx in list(active):
                essential = False
                for w in covers[idx]:
                    if cc[w] == 1:
                        essential = True
                        break
                if not essential:
                    for w in covers[idx]:
                        cc[w] -= 1
                    active.discard(idx)
                    changed = True
        return active, cc

    active, cc = ac_polish(active)
    best_size = len(active)
    best_state = set(active)

    if best_size <= 73:
        return [list(all_words[i]) for i in sorted(best_state)]

    # --- Local search: removals, 2-for-1 merges, swaps to escape local minima ---
    steps = 0
    max_steps = 400000
    while time.time() < deadline - 3 and steps < max_steps:
        steps += 1
        cur_size = len(active)

        # 1) try a redundant removal
        if cur_size > 1:
            i = rng.randint(0, cur_size - 1)
            idx = list(active)[i]
            if is_redundant(idx):
                for w in covers[idx]:
                    cc[w] -= 1
                active.discard(idx)
                if len(active) < best_size:
                    best_size = len(active)
                    best_state = set(active)
                continue

        # 2) 2-for-1 merge: find c1,c2 whose union is covered by a single c3
        if cur_size > 2 and steps % 3 == 0:
            al = list(active)
            c1, c2 = rng.sample(al, 2)
            union = covers[c1] | covers[c2]
            sample_word = rng.choice(tuple(union))
            found = -1
            for c3 in covered_by[sample_word]:
                if union <= covers[c3]:
                    found = c3
                    break
            if found >= 0:
                for w in covers[c1]:
                    cc[w] -= 1
                for w in covers[c2]:
                    cc[w] -= 1
                active.discard(c1)
                active.discard(c2)
                if found not in active:
                    for w in covers[found]:
                        cc[w] += 1
                    active.add(found)
                if len(active) < best_size:
                    best_size = len(active)
                    best_state = set(active)
                continue

        # 3) swap: remove a redundant word, then if still valid, try adding a word
        #    that lets us drop another (escape local minima)
        if cur_size > 1 and steps % 7 == 0:
            al = list(active)
            c1, c2 = rng.sample(al, 2)
            # temporarily remove c1
            for w in covers[c1]:
                cc[w] -= 1
            active.discard(c1)
            # find c3 that covers a now-uncovered word
            added = False
            for w in covers[c1]:
                if cc[w] == 0:
                    for c3 in covered_by[w]:
                        if c3 not in active:
                            for x in covers[c3]:
                                cc[x] += 1
                            active.add(c3)
                            added = True
                            break
                    if added:
                        break
            if not added:
                # restore
                for w in covers[c1]:
                    cc[w] += 1
                active.add(c1)
                continue
            # now try to drop c2 if redundant
            redundant = True
            for w in covers[c2]:
                if cc[w] == 1:
                    redundant = False
                    break
            if redundant:
                for w in covers[c2]:
                    cc[w] -= 1
                active.discard(c2)
                if len(active) < best_size:
                    best_size = len(active)
                    best_state = set(active)
            else:
                # rollback the add if it didn't help; keep c1
                # (leave the added c3 only if it reduced size; otherwise revert)
                if len(active) >= cur_size:
                    for w in covers[c3]:
                        cc[w] -= 1
                    active.discard(c3)
                    for w in covers[c1]:
                        cc[w] += 1
                    active.add(c1)

    # Final CSP polish on the best state found
    if len(best_state) < len(active):
        active = set(best_state)
        cc = np.zeros(total_words, dtype=int)
        for idx in active:
            for w in covers[idx]:
                cc[w] += 1
    active, cc = ac_polish(active)

    # Final validity check
    if np.any(cc == 0):
        active, cc = make_valid(list(active)), None
        cc = np.zeros(total_words, dtype=int)
        for idx in active:
            for w in covers[idx]:
                cc[w] += 1
        if np.any(cc == 0):
            # ultimate fallback: full space
            return [list(w) for w in all_words]

    return [list(all_words[i]) for i in sorted(active)]
