# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start = time.time()
    deadline = start + 135.0

    total = q ** n
    covers = []
    for c in itertools.product(range(q), repeat=n):
        s = [c]
        for pos in range(n):
            for val in range(q):
                if val == c[pos]:
                    continue
                lst = list(c)
                lst[pos] = val
                s.append(tuple(lst))
        covers.append(s)

    nwords = (total + 63) // 64
    full_mask = [0] * nwords
    for w in range(nwords):
        full_mask[w] = (1 << 64) - 1
    if total & 63:
        full_mask[-1] = (1 << (total & 63)) - 1

    def make_bitset(indices):
        words = [0] * nwords
        for idx in indices:
            words[idx >> 6] |= (1 << (idx & 63))
        return words

    cover_bits = [make_bitset(covers[i]) for i in range(total)]

    def count_uncovered_covered(cb, uncovered):
        c = 0
        for a, b in zip(cb, uncovered):
            c += bin(a & b).count("1")
        return c

    def is_full(cov):
        for w in range(nwords):
            if cov[w] != full_mask[w]:
                return False
        return True

    def full_covered(chosen):
        cov = [0] * nwords
        for i in chosen:
            cb = cover_bits[i]
            for w in range(nwords):
                cov[w] |= cb[w]
        return is_full(cov)

    def compute_count_only(chosen):
        cnt = [0] * total
        last = [-1] * total
        for i in chosen:
            for v in covers[i]:
                cnt[v] += 1
                last[v] = i
        co = [0] * total
        for v in range(total):
            if cnt[v] == 1:
                co[last[v]] += 1
        return co

    def try_improve(chosen):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            co = compute_count_only(chosen)
            # removals
            for idx in range(len(chosen)):
                if time.time() > deadline - 5.0:
                    break
                i = chosen[idx]
                if co[i] == 0:
                    newchosen = chosen[:idx] + chosen[idx+1:]
                    if full_covered(newchosen):
                        chosen = newchosen
                        improved = True
                        break
            if time.time() > deadline - 5.0:
                break
            # swaps to enable more removals
            co2 = compute_count_only(chosen)
            cnt = [0] * total
            last = [-1] * total
            for k in chosen:
                for v in covers[k]:
                    cnt[v] += 1
                    last[v] = k
            for idx in range(len(chosen)):
                if time.time() > deadline - 5.0:
                    break
                i = chosen[idx]
                # vertices uniquely covered by i
                uniq = [v for v in covers[i] if cnt[v] == 1 and last[v] == i]
                for j in range(total):
                    if j in chosen:
                        continue
                    if all(v in covers[j] for v in uniq):
                        newchosen = chosen[:idx] + [j] + chosen[idx+1:]
                        if full_covered(newchosen):
                            co_new = compute_count_only(newchosen)
                            rem_old = sum(1 for k in chosen if co2[k] == 0)
                            rem_new = sum(1 for k in newchosen if co_new[k] == 0)
                            if rem_new > rem_old:
                                chosen = newchosen
                                improved = True
                                break
                if improved:
                    break
        return chosen

    def greedy(seed):
        rng = random.Random(seed)
        uncovered = list(full_mask)
        chosen = []
        chosen_set = set()
        weights = [rng.random() for _ in range(total)]
        while True:
            if all(uncovered[w] == 0 for w in range(nwords)):
                break
            if time.time() > deadline - 5.0:
                break
            best = -1
            best_key = -1.0
            for i in range(total):
                if i in chosen_set:
                    continue
                cnt = count_uncovered_covered(cover_bits[i], uncovered)
                if cnt <= 0:
                    continue
                key = cnt + weights[i] * 0.001
                if key > best_key:
                    best_key = key
                    best = i
            if best < 0:
                break
            chosen.append(best)
            chosen_set.add(best)
            cb = cover_bits[best]
            for w in range(nwords):
                uncovered[w] &= ~cb[w]
        return try_improve(chosen)

    best = greedy(0)
    for seed in range(1, 30):
        if time.time() > deadline - 10.0:
            break
        cand = greedy(seed)
        if len(cand) < len(best):
            best = cand

    return [list(covers[i]) for i in best]
# EVOLVE-BLOCK-END
