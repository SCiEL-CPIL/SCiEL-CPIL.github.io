import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit

    # Precompute all forbidden triples: distinct a<b<c in 1..128
    # such that a+b, a+c, b+c are all in 1..128
    forbidden = set()
    for a in range(1, 126):
        for b in range(a + 1, 127):
            for c in range(b + 1, 129):
                if a + b <= 128 and a + c <= 128 and b + c <= 128:
                    forbidden.add((a, b, c))

    # For fast lookup: given a set S and candidate x, check if adding x creates a forbidden triple
    # We need to check if there exist distinct a,b,c in S∪{x} forming a forbidden triple
    
    # Precompute for each value v, the list of (a,b,c) triples containing v
    triples_containing = {v: [] for v in range(1, 129)}
    for (a, b, c) in forbidden:
        triples_containing[a].append((a, b, c))
        triples_containing[b].append((a, b, c))
        triples_containing[c].append((a, b, c))

    def can_add(S, x):
        """Check if S U {x} is valid using precomputed triples."""
        # Check all triples containing x
        for (a, b, c) in triples_containing[x]:
            # x must be one of a,b,c
            # The other two must be in S
            if x == a:
                if b in S and c in S:
                    return False
            elif x == b:
                if a in S and c in S:
                    return False
            else:  # x == c
                if a in S and b in S:
                    return False
        
        # Also check triples where x is a sum: x = a+b, a+c, or b+c
        # For each pair (a,b) in S with a<b, check if x = a+b and a+c, b+c in S for some c
        # More efficiently: for each c in S, check if x-c in S and (c + (x-c)) = x... 
        # Actually we need: exist distinct a,b,c in S such that {a+b, a+c, b+c} contains x
        # WLOG x = a+b. Then need a+c in S and b+c in S.
        # So: for each c in S, let a = x - c... no, that's not right.
        # For x = a+b: for each c in S, c != a, c != b, check a+c in S and b+c in S.
        # But we don't know a,b yet. Let's iterate over c in S:
        #   For x = a+b: a+b = x, so for each c in S (c distinct from a,b), 
        #   we need a+c in S and b+c in S where a+b=x.
        #   Equivalently: for each c in S, for each a in S with a < x-a (so a < x/2),
        #   let b = x-a. If b in S, b != a, b != c, a != c, and a+c in S and b+c in S, return False.
        
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            b = x - a
            if b <= a or b not in S:
                continue
            for j in range(n):
                c = S_list[j]
                if c == a or c == b:
                    continue
                if (a + c) in S and (b + c) in S:
                    return False
        
        return True

    best_set = set()
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    def greedy_construct(order):
        """Greedy construction given a fixed order of candidates."""
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_passes=20):
        """Improve S via additions and systematic swaps."""
        S = set(S)
        not_in_S = [x for x in range(1, 129) if x not in S]
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            
            improved = False
            random.shuffle(not_in_S)
            
            for x in not_in_S[:]:
                if time.time() > deadline:
                    break
                
                if can_add(S, x):
                    S.add(x)
                    not_in_S.remove(x)
                    improved = True
                    update_best(S)
                    continue
                
                S_list = list(S)
                random.shuffle(S_list)
                
                found_swap = False
                for y in S_list[:15]:
                    if time.time() > deadline:
                        break
                    S_temp = S - {y}
                    if can_add(S_temp, x):
                        S = S_temp
                        S.add(x)
                        not_in_S.append(y)
                        if x in not_in_S:
                            not_in_S.remove(x)
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break
            
            if not improved:
                break
        
        return S

    # Phase 1: Multiple greedy constructions with diverse orderings
    random.seed(42)
    
    # Compute degree for each vertex in the "conflict" sense:
    # degree(v) = number of triples containing v
    degree = {v: len(triples_containing[v]) for v in range(1, 129)}
    
    # Lowest degree first (least restrictive)
    by_degree = sorted(range(1, 129), key=lambda v: degree[v])
    S = greedy_construct(by_degree)
    update_best(S)
    
    # Ascending
    S = greedy_construct(list(range(1, 129)))
    update_best(S)
    
    # Descending
    S = greedy_construct(list(range(128, 0, -1)))
    update_best(S)
    
    # Highest degree first
    by_degree_desc = sorted(range(1, 129), key=lambda v: -degree[v])
    S = greedy_construct(by_degree_desc)
    update_best(S)
    
    # Even/odd interleaved
    evens = list(range(2, 129, 2))
    odds = list(range(1, 129, 2))
    interleaved = []
    for i in range(64):
        if i < len(odds):
            interleaved.append(odds[i])
        if i < len(evens):
            interleaved.append(evens[i])
    S = greedy_construct(interleaved)
    update_best(S)
    
    # Several random orders
    for i in range(30):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        update_best(S)
    
    # Phase 2: Local search from best set
    if time.time() < deadline:
        S = local_search(best_set, max_passes=30)
        update_best(S)
    
    # Phase 3: More random restarts with local search
    for i in range(40):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        S = local_search(S, max_passes=15)
        update_best(S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add(best_set, x):
                best_set.add(x)
    
    # One more local search pass on the final best set
    if time.time() < deadline and best_size < 128:
        S = local_search(best_set, max_passes=10)
        update_best(S)
    
    result = [int(x) for x in sorted(best_set)]
    return result
