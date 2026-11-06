# Precompute
TRIPLES = []
ELEM_TO_TRIPLES = {i: [] for i in range(1, 129)}
TRIPLES_WITH_X = {i: [] for i in range(1, 129)} # List of (t_idx, o1, o2)

def precompute():
    for d in range(1, 129):
        multiples = [k for k in range(d, 129, d)]
        if len(multiples) < 3:
            continue
        for a, b, c in combinations(multiples, 3):
            if gcd(a, b) == d and gcd(b, c) == d and gcd(c, a) == d:
                idx = len(TRIPLES)
                TRIPLES.append((a, b, c))
                ELEM_TO_TRIPLES[a].append(idx)
                ELEM_TO_TRIPLES[b].append(idx)
                ELEM_TO_TRIPLES[c].append(idx)
                
                TRIPLES_WITH_X[a].append((idx, b, c))
                TRIPLES_WITH_X[b].append((idx, a, c))
                TRIPLES_WITH_X[c].append((idx, a, b))
                
precompute()
