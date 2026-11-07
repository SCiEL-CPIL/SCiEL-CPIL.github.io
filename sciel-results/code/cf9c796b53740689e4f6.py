import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = 128
    S = set(range(1, n + 1))
    
    while time.time() < deadline:
        elems = sorted(S)
        bad_triples = []
        
        for i in range(len(elems)):
            if time.time() > deadline - 3:
                break
            a = elems[i]
            for j in range(i + 1, len(elems)):
                b = elems[j]
                ab = a + b
                if ab not in S:
                    continue
                for k in range(j + 1, len(elems)):
                    c = elems[k]
                    if (a + c) in S and (b + c) in S:
                        bad_triples.append((a, b, c))
        
        if not bad_triples:
            break
        
        count = Counter()
        for (a, b, c) in bad_triples:
            for x in (a, b, c, a + b, a + c, b + c):
                if x in S:
                    count[x] += 1
        
        best_elem = max(count.keys(), key=lambda x: count[x])
        S.remove(best_elem)
    
    return sorted(S)
