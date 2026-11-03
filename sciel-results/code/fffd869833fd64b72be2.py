import itertools
import random
import time

def _to_int(word, q, n):
    val = 0
    for d in word:
        val = val * q + d
    return val

def _to_word(val, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = val % q
        val //= q
    return word

def _build_neighbors(q, n, radius):
    """Build neighbors for all words in the space (for radius=1, q=3, n=6: 729 words)."""
    total = q ** n
    # Precompute digit arrays for all words
    words = [_to_word(v, q, n) for v in range(total)]
    
    neighbors = [None] * total
    for v in range(total):
        w = words[v]
        nbs = [v]
        if radius >= 1:
            for i in range(n):
                for new_val in range(q):
                    if new_val == w[i]:
                        continue
                    new_word = w[:]
                    new_word[i] = new_val
                    nbs.append(_to_int(new_word, q, n))
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    for vi in range(q):
                        for vj in range(q):
                            if vi == w[i] and vj == w[j]:
                                continue
                            if vi == w[i] or vj == w[j]:
                                continue  # already counted in radius=1
                            new_word = w[:]
                            new_word[i] = vi
                            new_word[j] = vj
                            nbs.append(_to_int(new_word, q, n))
        neighbors[v] = nbs
    return neighbors

def _greedy_independent_set(total, q, n, dist2_neighbors, deadline):
    """
    Greedy independent set on the distance-2 graph.
    A covering code of radius 1 requires that no two codewords are within distance 2
    of each other (otherwise they'd be redundant). So we want a maximum independent set
    in the graph where edges connect words at distance <= 2.
    
    Greedy: repeatedly pick the word that covers the most uncovered vertices,
    mark all its neighbors (distance <= 1) as covered, and remove it from candidates.
    This gives an independent set in the distance-2 graph.
    """
    total = q ** n
    covered = [False] * total
    in_code = [False] * total
    candidates = list(range(total))
    
    # We want to maximize coverage while maintaining independence in dist-2 graph.
    # Greedy: pick word with most uncovered neighbors (in dist-1 graph),
    # then mark dist-1 neighbors as covered and dist-2 neighbors as excluded from future picks.
    
    # Precompute dist-1 neighbors (for coverage) and dist-2 neighbors (for exclusion)
    dist1_neighbors = [None] * total
    dist2_neighbors_list = [None] * total
    words = [_to_word(v, q, n) for v in range(total)]
    
    for v in range(total):
        w = words[v]
        d1 = [v]
        d2 = set([v])
        for i in range(n):
            for nv in range(q):
                if nv == w[i]:
                    continue
                nw = w[:]
                nw[i] = nv
                u = _to_int(nw, q, n)
                d1.append(u)
                d2.add(u)
        # distance 2
        for i in range(n):
            for j in range(i + 1, n):
                for vi in range(q):
                    if vi == w[i]:
                        continue
                    for vj in range(q):
                        if vj == w[j]:
                            continue
                        nw = w[:]
                        nw[i] = vi
                        nw[j] = vj
                        u = _to_int(nw, q, n)
                        d2.add(u)
        dist1_neighbors[v] = d1
        dist2_neighbors_list[v] = list(d2)
    
    code = []
    excluded = [False] * total  # words excluded because they're within dist 2 of a chosen word
    
    while candidates:
        if time.time() > deadline:
            break
        # Find candidate with most uncovered dist-1 neighbors
        best_v = -1
        best_count = -1
        for v in candidates:
            if excluded[v]:
                continue
            cnt = 0
            for u in dist1_neighbors[v]:
                if not covered[u]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_v = v
        
        if best_v < 0:
            # No non-excluded candidates
            break
        
        # Choose best_v
        code.append(best_v)
        covered[best_v] = True
        for u in dist1_neighbors[best_v]:
            covered[u] = True
        for u in dist2_neighbors_list[best_v]:
            excluded[u] = True
        
        # Remove covered or excluded from candidates
        new_candidates = []
        for v in candidates:
            if v != best_v and not covered[v] and not excluded[v]:
                new_candidates.append(v)
        candidates = new_candidates
    
    return code, dist1_neighbors

def _construct_ternary_6_1():
    q, n, radius = 3, 6, 1
    total = q ** n  # 729
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build neighbors
    words = [_to_word(v, q, n) for v in range(total)]
    dist1_neighbors = [None] * total
    dist2_neighbors = [None] * total
    
    for v in range(total):
        w = words[v]
        d1 = [v]
        d2 = set([v])
        for i in range(n):
            for nv in range(q):
                if nv == w[i]:
                    continue
                nw = w[:]
                nw[i] = nv
                u = _to_int(nw, q, n)
                d1.append(u)
                d2.add(u)
        for i in range(n):
            for j in range(i + 1, n):
                for vi in range(q):
                    if vi == w[i]:
                        continue
                    for vj in range(q):
                        if vj == w[j]:
                            continue
                        nw = w[:]
                        nw[i] = vi
                        nw[j] = vj
                        u = _to_int(nw, q, n)
                        d2.add(u)
        dist1_neighbors[v] = d1
        dist2_neighbors[v] = list(d2)
    
    # Phase 1: Greedy independent set to get initial code
    initial_code, _ = _greedy_independent_set(total, q, n, dist2_neighbors, start_time + 30.0)
    
    code_set = set(initial_code)
    coverage = [0] * total
    
    def add_word(c):
        for nb in dist1_neighbors[c]:
            coverage[nb] += 1
    
    def remove_word(c):
        for nb in dist1_neighbors[c]:
            coverage[nb] -= 1
    
    for c in code_set:
        add_word(c)
    
    # Verify initial code covers everything
    min_cov = min(coverage)
    
    # Phase 2: Local search
    best_code = set(code_set)
    best_size = len(code_set)
    current_code = list(code_set)
    
    while time.time() < deadline:
        # Try removals
        removed_any = False
        # Compute for each codeword the min coverage in its neighborhood
        removal_candidates = []
        for c in current_code:
            min_cov_c = min(coverage[nb] for nb in dist1_neighbors[c])
            removal_candidates.append((min_cov_c, c))
        
        removal_candidates.sort(key=lambda x: -x[0])
        
        for min_cov_c, c in removal_candidates:
            if time.time() > deadline:
                break
            # Re-check: is c still in current_code?
            if c not in code_set:
                continue
            remove_word(c)
            if min(coverage) > 0:
                code_set.remove(c)
                current_code.remove(c)
                removed_any = True
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = set(code_set)
            else:
                add_word(c)
        
        if not removed_any:
            # Try 1-swap moves
            swap_done = False
            for c in current_code:
                if time.time() > deadline:
                    break
                # Try replacing c with a random word
                for attempt in range(20):
                    if time.time() > deadline:
                        break
                    new_c = random.randint(0, total - 1)
                    if new_c in code_set:
                        continue
                    
                    remove_word(c)
                    add_word(new_c)
                    
                    if min(coverage) > 0:
                        code_set.remove(c)
                        code_set.add(new_c)
                        current_code.remove(c)
                        current_code.append(new_c)
                        swap_done = True
                        break
                    else:
                        remove_word(new_c)
                        add_word(c)
                if swap_done:
                    break
            
            if not swap_done:
                # Try 2-exchange: remove two codewords, add one
                # This can reduce size by 1
                if len(current_code) >= 2:
                    # Try random pairs
                    for attempt in range(10):
                        if time.time() > deadline:
                            break
                        idx1 = random.randint(0, len(current_code) - 1)
                        idx2 = random.randint(0, len(current_code) - 1)
                        if idx1 == idx2:
                            continue
                        c1 = current_code[idx1]
                        c2 = current_code[idx2]
                        
                        new_c = random.randint(0, total - 1)
                        if new_c in code_set:
                            continue
                        
                        remove_word(c1)
                        remove_word(c2)
                        add_word(new_c)
                        
                        if min(coverage) > 0:
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(new_c)
                            current_code.remove(c1)
                            current_code.remove(c2)
                            current_code.append(new_c)
                            if len(code_set) < best_size:
                                best_size = len(code_set)
                                best_code = set(code_set)
                            break
                        else:
                            remove_word(new_c)
                            add_word(c1)
                            add_word(c2)
                
                # If still stuck, try perturbation: random swap even if it doesn't improve
                if current_code:
                    c_to_swap = random.choice(current_code)
                    new_c = random.randint(0, total - 1)
                    if new_c in code_set:
                        new_c = (new_c + 1) % total
                        if new_c in code_set:
                            continue
                    
                    remove_word(c_to_swap)
                    add_word(new_c)
                    
                    code_set.remove(c_to_swap)
                    code_set.add(new_c)
                    current_code.remove(c_to_swap)
                    current_code.append(new_c)
    
    return [_to_word(c, q, n) for c in best_code]

def _construct_generic(q, n, radius):
    total = q ** n
    if total > 10000:
        raise ValueError("Generic fallback too slow for large spaces")
    
    words = [_to_word(v, q, n) for v in range(total)]
    
    # Build neighbors for radius
    neighbors = [None] * total
    for v in range(total):
        w = words[v]
        nbs = [v]
        if radius >= 1:
            for i in range(n):
                for nv in range(q):
                    if nv == w[i]:
                        continue
                    nw = w[:]
                    nw[i] = nv
                    nbs.append(_to_int(nw, q, n))
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    for vi in range(q):
                        if vi == w[i]:
                            continue
                        for vj in range(q):
                            if vj == w[j]:
                                continue
                            if vi == w[i] or vj == w[j]:
                                continue
                            nw = w[:]
                            nw[i] = vi
                            nw[j] = vj
                            nbs.append(_to_int(nw, q, n))
        neighbors[v] = nbs
    
    # Greedy set cover
    covered = [False] * total
    code = []
    remaining = list(range(total))
    
    while remaining:
        best_v = -1
        best_count = -1
        for v in remaining:
            cnt = sum(1 for u in neighbors[v] if not covered[u])
            if cnt > best_count:
                best_count = cnt
                best_v = v
        
        if best_v < 0:
            break
        
        code.append(words[best_v])
        for u in neighbors[best_v]:
            covered[u] = True
        remaining = [v for v in remaining if not covered[v]]
    
    return code

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    return _construct_generic(q, n, radius)

if __name__ == "__main__":
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
    
    # Verify
    q, n = 3, 6
    total = q ** n
    code_set = set(_to_int(w, q, n) for w in code)
    all_covered = True
    for w in range(total):
        word = _to_word(w, q, n)
        found = False
        for c in code_set:
            cw = _to_word(c, q, n)
            d = sum(1 for a, b in zip(word, cw) if a != b)
            if d <= 1:
                found = True
                break
        if not found:
            all_covered = False
            print(f"Uncovered: {word}")
            break
    print(f"Valid: {all_covered}")
