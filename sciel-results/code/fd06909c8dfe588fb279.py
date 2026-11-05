import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # STS(15) via GF(2)^4
    sts_triples = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            k = i ^ j
            if k > j:
                sts_triples.append(tuple(sorted([i - 1, j - 1, k - 1])))
    
    triple_pairs = []
    for t in sts_triples:
        triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
    
    pair_to_triple = {}
    for idx, t in enumerate(sts_triples):
        for i in range(3):
            for j in range(i + 1, 3):
                pair_to_triple[tuple(sorted((t[i], t[j])))] = idx
    
    # Precompute
    subsets_8 = list(itertools.combinations(range(15), 8))
    subsets_7 = list(itertools.combinations(range(15), 7))
    
    triples_in_8 = []
    for s in subsets_8:
        ss = set(s)
        triples_in_8.append([i for i, t in enumerate(sts_triples) if t[0] in ss and t[1] in ss and t[2] in ss])
    
    triples_in_7 = []
    for s in subsets_7:
        ss = set(s)
        triples_in_7.append([i for i, t in enumerate(sts_triples) if t[0] in ss and t[1] in ss and t[2] in ss])
    
    def check_alpha7(M):
        Ms = set(M)
        for i in range(len(subsets_8)):
            ok = True
            for ti in triples_in_8[i]:
                if not any(p in Ms for p in triple_pairs[ti]):
                    ok = False
                    break
            if ok:
                return False
        for i in range(len(subsets_7)):
            ss = set(subsets_7[i])
            if any(u in ss and v in ss for u, v in M):
                continue
            ok = True
            for ti in triples_in_7[i]:
                if not any(p in Ms for p in triple_pairs[ti]):
                    ok = False
                    break
            if ok:
                return False
        return True
    
    def make_triples(M):
        final = [[15, u, v] for u, v in M]
        rem = set(pair_to_triple[tuple(sorted((u, v)))] for u, v in M)
        for i, t in enumerate(sts_triples):
            if i not in rem:
                final.append(list(t))
        return final
    
    def linear_ok(trs):
        sp = set()
        for t in trs:
            for i in range(3):
                for j in range(i+1, 3):
                    p = tuple(sorted((t[i], t[j])))
                    if p in sp:
                        return False
                    sp.add(p)
        return True
    
    def rand_match():
        pts = list(range(15))
        random.shuffle(pts)
        return [tuple(sorted((pts[i], pts[i+1]))) for i in range(0, 14, 2)]
    
    best = None
    for r in range(5000):
        if time.time() > deadline - 10:
            break
        M = rand_match()
        if check_alpha7(M):
            best = M
            break
        for s in range(300):
            if time.time() > deadline - 10:
                break
            if best:
                break
            i = random.randint(0, 6)
            M.pop(i)
            used = set(p for pair in M for p in pair)
            av = [x for x in range(15) if x not in used]
            for a in range(len(av)):
                for b in range(a+1, len(av)):
                    np = tuple(sorted((av[a], av[b])))
                    M.append(np)
                    if check_alpha7(M):
                        best = M[:]
                        break
                    M.pop()
                if best:
                    break
            if best:
                break
        if best:
            break
    
    if best is None:
        best = rand_match()
    
    trs = make_triples(best)
    if not linear_ok(trs):
        return [list(t) for t in sts_triples]
    return trs

if __name__ == "__main__":
    t = construct()
    print(f"Constructed {len(t)} triples")
