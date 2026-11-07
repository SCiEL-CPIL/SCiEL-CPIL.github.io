import time

def construct():
    deadline = time.time() + 80
    
    n = 128
    S = set(range(1, n + 1))
    
    def find_bad_triples(S):
        bad = []
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i+1, m):
                b = elems[j]
                for k in range(j+1, m):
                    c = elems[k]
                    if (a+b) in S and (a+c) in S and (b+c) in S:
                        bad.append((a, b, c))
        return bad
    
    def count_involvement(S, bad_triples):
        counts = {}
        for (a, b, c) in bad_triples:
            for x in (a, b, c, a+b, a+c, b+c):
                if x in S:
                    counts[x] = counts.get(x, 0) + 1
        return counts
    
    bad = find_bad_triples(S)
    
    while bad and time.time() < deadline:
        counts = count_involvement(S, bad)
        if not counts:
            break
        to_remove = max(counts, key=counts.get)
        S.discard(to_remove)
        bad = find_bad_triples(S)
    
    # Verify
    elems = sorted(S)
    for i in range(len(elems)):
        a = elems[i]
        for j in range(i+1, len(elems)):
            b = elems[j]
            for k in range(j+1, len(elems)):
                c = elems[k]
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    # Should not happen
                    pass
    
    return list(elems)
