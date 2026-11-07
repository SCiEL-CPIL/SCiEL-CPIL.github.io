import time
import random
from math import comb

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    n = 12
    m = 40
    total = 1 << n  # 4096

    # ---- Precompute strict subset / superset bitsets (4096-bit ints) ----
    # popcount for all masks
    pop = [0] * total
    for x in range(1, total):
        pop[x] = pop[x >> 1] + (x & 1)

    # SUB[m]: bitmask of masks that are strict subsets of m
    # SUP[m]: bitmask of masks that are strict supersets of m
    SUB = [0] * total
    SUP = [0] * total
    for s in range(total):
        full = (1 << s) - 1
        sub = full
        while True:
            SUB[s] |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & s
        SUP[full ^ s] |= (1 << s)
    # remove self
    for msk in range(total):
        SUB[msk] &= ~(1 << msk)
        SUP[msk] &= ~(1 << msk)

    CMP = [SUB[m] | SUP[m] for m in range(total)]

    def cmp_score(fam):
        # fam: list of m distinct masks; return # unordered comparable pairs
        # = sum over i of popcount(CMP[fam[i]] & union_{j>i} bit(fam[j]))
        # efficient: iterate, maintain running bitset of "seen" masks
        seen = 0
        total_c = 0
        for msk in fam:
            total_c += (CMP[msk] & seen).bit_count()
            seen |= (1 << msk)
        return total_c

    random.seed(42)

    # ---- Initial family: a chain of 13 (all sizes 0..12) + greedy fill ----
    chain = [(1 << s) - 1 for s in range(13)]
    fam = list(chain)
    fam_set = set(fam)
    # greedy fill: add mask maximizing comparability with current fam
    order = list(range(total))
    random.shuffle(order)
    for _ in range(m - len(fam)):
        best = None
        bestg = -1
        for c in order:
            if c in fam_set:
                continue
            g = CMP[c]
            # gain = number of existing fam comparable with c
            cnt = 0
            for f in fam:
                if (g >> f) & 1:
                    cnt += 1
            if cnt > bestg:
                bestg = cnt
                best = c
        fam.append(best)
        fam_set.add(best)

    best_fam = list(fam)
    best_score = cmp_score(best_fam)

    def try_swap(fam, i, c):
        # return net gain of replacing fam[i] with c (c not in fam)
        old = fam[i]
        others = 0
        for j in range(m):
            if j != i:
                others |= (1 << fam[j])
        gain = (CMP[c] & others).bit_count()
        loss = (CMP[old] & others).bit_count()
        return gain - loss

    # ---- Phase 1: exhaustive single-swap local search (fast) ----
    def local_search(fam, score, budget):
        improved = True
        while improved and time.time() < budget:
            improved = False
            fam_set = set(fam)
            for i in range(m):
                old = fam[i]
                # find best replacement
                bestc = -1
                bestg = 0
                for c in range(total):
                    if c == old or c in fam_set:
                        continue
                    g = try_swap(fam, i, c)
                    if g > bestg:
                        bestg = g
                        bestc = c
                if bestc >= 0:
                    fam_set.remove(old)
                    fam[i] = bestc
                    fam_set.add(bestc)
                    score += bestg
                    improved = True
                    break
        return fam, score

    best_fam, best_score = local_search(best_fam, best_score, deadline - 6.0)

    # ---- Phase 2: 2-opt (two removals + two insertions) ----
    def two_opt(fam, score, budget):
        improved = True
        while improved and time.time() < budget:
            improved = False
            fam_set = set(fam)
            # precompute others for each pair removal is expensive; do smart
            # For each ordered pair (i,j) to remove, find best two insertions
            # Use: base others = all masks not in {i,j}
            for i in range(m):
                if time.time() >= budget:
                    break
                mi = fam[i]
                for j in range(i + 1, m):
                    mj = fam[j]
                    others = 0
                    for k in range(m):
                        if k != i and k != j:
                            others |= (1 << fam[k])
                    # loss of removing both
                    loss = (CMP[mi] & others).bit_count() + (CMP[mj] & others).bit_count()
                    # find best two distinct insertions c1,c2 not in fam
                    best1 = -1
                    best2 = -1
                    bestg1 = 0
                    bestg2 = 0
                    for c in range(total):
                        if c in fam_set:
                            continue
                        g = (CMP[c] & others).bit_count()
                        if g > bestg1:
                            if c == best1:
                                continue
                            bestg2 = bestg1
                            best2 = best1
                            bestg1 = g
                            best1 = c
                        elif g > bestg2:
                            bestg2 = g
                            best2 = c
                    total_gain = bestg1 + bestg2
                    if total_gain - loss > 0 and best2 >= 0:
                        fam_set.discard(mi)
                        fam_set.discard(mj)
                        fam[i] = best1
                        fam[j] = best2
                        fam_set.add(best1)
                        fam_set.add(best2)
                        score += total_gain - loss
                        improved = True
                        break
                if improved:
                    break
        return fam, score

    best_fam, best_score = two_opt(best_fam, best_score, deadline - 3.0)

    # ---- Phase 3: a couple of random restarts if time remains ----
    while time.time() < deadline - 5.0:
        # perturb best solution: random 2-opt style perturbation
        fam = list(best_fam)
        fam_set = set(fam)
        # remove 2 random, insert 2 random good
        idxs = random.sample(range(m), 2)
        removed = [fam[idxs[0]], fam[idxs[1]]]
        others = 0
        for k in range(m):
            if k not in idxs:
                others |= (1 << fam[k])
        # pick best two insertions
        cands = []
        for c in range(total):
            if c in fam_set:
                continue
            g = (CMP[c] & others).bit_count()
            cands.append((g, c))
        cands.sort(reverse=True)
        new1 = cands[0][1]
        new2 = cands[1][1]
        fam[idxs[0]] = new1
        fam[idxs[1]] = new2
        # local search
        sc = cmp_score(fam)
        fam, sc = local_search(fam, sc, deadline - 4.0)
        if sc > best_score:
            best_score = sc
            best_fam = list(fam)

    # ---- Convert to list of lists of ints ----
    result = []
    for msk in best_fam:
        s = []
        for b in range(n):
            if (msk >> b) & 1:
                s.append(int(b))
        result.append(s)
    return result
