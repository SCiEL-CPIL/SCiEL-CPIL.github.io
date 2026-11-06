import random
import time

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin under 150s

    n = 16
    # all candidate triples
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a,b,c))

    # S: pair -> set of third points
    def build_S(triples):
        S = {}
        for (a,b,c) in triples:
            for (x,y) in [(a,b),(a,c),(b,c)]:
                S.setdefault(frozenset((x,y)), set()).add(c if (x,y)==(a,b) else (b if (x,y)==(a,c) else a))
        return S

    def is_forbidden(T, S, present):
        a,b,c = T
        # need a' in A' (not a), b' not b, c' not c, all distinct
        others = [x for x in range(n) if x not in (a,b,c)]
        # For efficiency iterate a',b'
        # We need c in S[a'b] & S[a'b'] & S[ab']
        # and c' in S[ab]&S[a'b]&S[ab']&S[a'b']
        # Precompute S_ab etc
        def Sset(x,y):
            return S.get(frozenset((x,y)), set())
        Sab = Sset(a,b)
        for ap in others:
            if ap == b or ap == c: continue
            Sabp = Sset(a,ap)  # S[a b'] where b'=ap? careful: ap is a'
            # ap is a'
            for bp in others:
                if bp == ap or bp == a or bp == b or bp == c: continue
                # bp is b'
                # check c in S[a' b] & S[a' b'] & S[a b']
                if c not in Sset(ap,b): continue
                if c not in Sset(ap,bp): continue
                if c not in Sset(a,bp): continue
                # now need c' in S[a b] & S[a' b] & S[a b'] & S[a' b']
                inter = Sab & Sset(ap,b) & Sset(a,bp) & Sset(ap,bp)
                for cp in inter:
                    if cp not in (a,b,c,ap,bp):
                        return True
        return False

    def greedy(order, S, present, triples_list):
        # triples_list is list of current triples
        for T in order:
            if time.time() > DEADLINE: break
            if T in present: continue
            if not is_forbidden(T, S, present):
                present.add(T)
                triples_list.append(T)
                a,b,c = T
                for (x,y) in [(a,b),(a,c),(b,c)]:
                    S.setdefault(frozenset((x,y)), set()).add(c if (x,y)==(a,b) else (b if (x,y)==(a,c) else a))
        return triples_list

    best = None
    best_len = -1

    # seed: star at 0
    star = [(0,a,b) for a in range(1,n) for b in range(a+1,n)]
    best = list(star)
    best_len = len(best)

    rng = random.Random(12345)

    attempts = 0
    while time.time() < DEADLINE - 5:
        attempts += 1
        # random order of all triples, but start from a random star or empty
        if attempts % 3 == 1:
            init = []
        elif attempts % 3 == 2:
            p = rng.randint(0,n-1)
            init = [(p,a,b) for a in range(n) if a!=p for b in range(a+1,n) if b!=p]
            # dedupe ordering
            init = sorted(set(tuple(sorted(t)) for t in init))
            init = [list(t) for t in init]
            init = [tuple(t) for t in init]
        else:
            # random subset of triples
            init = [tuple(sorted(t)) for t in rng.sample(all_triples, 40)]

        S = build_S(init)
        present = set(init)
        triples_list = list(init)

        order = all_triples[:]
        rng.shuffle(order)
        greedy(order, S, present, triples_list)

        if len(triples_list) > best_len:
            best_len = len(triples_list)
            best = triples_list
            if time.time() > DEADLINE - 2: break

    # final fill pass on best
    if best is not None and time.time() < DEADLINE - 2:
        S = build_S(best)
        present = set(best)
        triples_list = list(best)
        order = all_triples[:]
        rng.shuffle(order)
        greedy(order, S, present, triples_list)
        best = triples_list

    return [list(t) for t in best]
