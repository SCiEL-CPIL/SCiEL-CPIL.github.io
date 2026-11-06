import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 138
    
    n = 16
    all_triples = [tuple(t) for t in combinations(range(n), 3)]
    
    # Precompute all K(2,2,2) patterns
    # Each pattern is a tuple of 8 frozensets
    k222_patterns = []
    for six in combinations(range(n), 6):
        a, b, c, d, e, f = six
        pairings = [
            ((a,b),(c,d),(e,f)), ((a,b),(c,e),(d,f)), ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)), ((a,c),(b,e),(d,f)), ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)), ((a,d),(b,e),(c,f)), ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)), ((a,e),(b,d),(c,f)), ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)), ((a,f),(b,d),(c,e)), ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in pairings:
            x1, x2 = p1
            y1, y2 = p2
            z1, z2 = p3
            triples = [
                (x1, y1, z1), (x1, y1, z2), (x1, y2, z1), (x1, y2, z2),
                (x2, y1, z1), (x2, y1, z2), (x2, y2, z1), (x2, y2, z2),
            ]
            k222_patterns.append(tuple(frozenset(t) for t in triples))
    
    num_patterns = len(k222_patterns)
    
    # For each triple (as frozenset), which patterns contain it
    triple_to_patterns = {}
    for idx, pattern in enumerate(k222_patterns):
        for t in pattern:
            if t not in triple_to_patterns:
                triple_to_patterns[t] = []
            triple_to_patterns[t].append(idx)
    
    # Seed solution
    seed_triples = [(0, a, b) for a in range(1, n) for b in range(a+1, n)]
    best_list = seed_triples[:]
    best_count = len(seed_triples)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 104729 + 13)
        shuffled = list(all_triples)
        rng.shuffle(shuffled)
        
        current_set = set()
        current_list = []
        pattern_counts = [0] * num_patterns
        
        for t in shuffled:
            if time.time() > deadline:
                break
            tf = frozenset(t)
            would_complete = False
            for pidx in triple_to_patterns.get(tf, []):
                if pattern_counts[pidx] == 7:
                    would_complete = True
                    break
            if not would_complete:
                current_set.add(tf)
                current_list.append(t)
                for pidx in triple_to_patterns.get(tf, []):
                    pattern_counts[pidx] += 1
        
        if len(current_list) > best_count:
            best_count = len(current_list)
            best_list = current_list[:]
        
        if time.time() > deadline:
            break
        
        # Add skipped triples
        skipped = [t for t in all_triples if frozenset(t) not in current_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(skipped)):
                if time.time() > deadline:
                    break
                t = skipped[i]
                tf = frozenset(t)
                would_complete = False
                for pidx in triple_to_patterns.get(tf, []):
                    if pattern_counts[pidx] == 7:
                        would_complete = True
                        break
                if not would_complete:
                    current_set.add(tf)
                    current_list.append(t)
                    for pidx in triple_to_patterns.get(tf, []):
                        pattern_counts[pidx] += 1
                    skipped.pop(i)
                    improved = True
                    if len(current_list) > best_count:
                        best_count = len(current_list)
                        best_list = current_list[:]
                    i -= 1
        
        if time.time() > deadline:
            break
        
        # Swap improvement
        if time.time() < deadline - 3 and skipped and current_list:
            for _ in range(50):
                if time.time() > deadline:
                    break
                rem_idx = rng.randint(0, len(current_list) - 1)
                rem_t = current_list[rem_idx]
                rem_f = frozenset(rem_t)
                add_t = rng.choice(skipped)
                add_f = frozenset(add_t)
                
                ok = True
                for pidx in triple_to_patterns.get(add_f, []):
                    nc = pattern_counts[pidx]
                    if rem_f in k222_patterns[pidx]:
                        nc -= 1
                    nc += 1
                    if nc >= 8:
                        ok = False
                        break
                if ok:
                    for pidx in triple_to_patterns.get(rem_f, []):
                        pattern_counts[pidx] -= 1
                    for pidx in triple_to_patterns.get(add_f, []):
                        pattern_counts[pidx] += 1
                    current_set.discard(rem_f)
                    current_set.add(add_f)
                    current_list[rem_idx] = add_t
                    skipped.remove(add_t)
                    skipped.append(rem_t)
        
        if time.time() > deadline:
            break
    
    return [list(t) for t in best_list]
