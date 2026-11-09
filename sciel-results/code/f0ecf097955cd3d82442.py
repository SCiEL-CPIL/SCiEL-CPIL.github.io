import random
import itertools

def construct():
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    best = []
    best_len = 0
    
    for attempt in range(50):
        random.shuffle(all_triples)
        current = set()
        
        for t in all_triples:
            a, b, c = t
            # Check if adding (a,b,c) creates a K(2,2,2)
            valid = True
            for a2 in points:
                if a2 == a:
                    continue
                for b2 in points:
                    if b2 == b or b2 == a2:
                        continue
                    for c2 in points:
                        if c2 == c or c2 == a2 or c2 == b2:
                            continue
                        # Check all 8 transversals
                        # Pairs: {a,a2}, {b,b2}, {c,c2}
                        ok = True
                        for i in (0, 1):
                            for j in (0, 1):
                                for k in (0, 1):
                                    x = a if i == 0 else a2
                                    y = b if j == 0 else b2
                                    z = c if k == 0 else c2
                                    triple = tuple(sorted((x, y, z)))
                                    if triple == t:
                                        # This is the triple we're adding
                                        continue
                                    if triple not in current:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        if not ok:
                            # Not all 8 present, so this configuration is fine
                            continue
                        # All 8 present (including t which we're about to add)
                        valid = False
                        break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                current.add(t)
        
        if len(current) > best_len:
            best_len = len(current)
            best = list(current)
    
    return [list(t) for t in best]
