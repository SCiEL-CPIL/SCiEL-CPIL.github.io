import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 100
    target = 200
    
    def compute_reachable(S):
        mask = (1 << (target + 1)) - 1
        r = 1
        for x in S:
            r = (r | (r << x)) & mask
        return r

    def is_valid(S):
        r = compute_reachable(S)
        return ((r >> target) & 1) == 0

    best_set = set()
    best_size = 0
    
    # 1. Try greedy removal from large ranges [L, 100]
    # For L >= 50, only triples sum to 200.
    for L in range(50, 67):
        if time.time() > deadline - 10: break
        
        candidates = list(range(L, 101))
        
        # Generate all triples summing to 200
        # a < b < c
        triples = []
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i+1, len(candidates)):
                b = candidates[j]
                c = target - a - b
                # c must be > b and in candidates
                if c <= b: continue
                if c > 100: continue
                if c in candidates:
                    triples.append((a, b, c))
        
        if not triples:
            # No conflicts, take all
            S_try = set(candidates)
            if len(S_try) > best_size:
                best_size = len(S_try)
                best_set = S_try
            continue
            
        # Greedy Vertex Cover
        # We want to remove min vertices to hit all triples.
        # Heuristic: pick vertex in most triples.
        
        # Build adjacency: vertex -> list of triple indices
        vertex_triples = {v: [] for v in candidates}
        for idx, (a, b, c) in enumerate(triples):
            vertex_triples[a].append(idx)
            vertex_triples[b].append(idx)
            vertex_triples[c].append(idx)
            
        active_triples = set(range(len(triples)))
        removed = set()
        
        while active_triples:
            # Find vertex with max degree in active_triples
            best_v = None
            max_deg = -1
            for v in candidates:
                if v in removed: continue
                deg = sum(1 for t in vertex_triples[v] if t in active_triples)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    
            if best_v is None or max_deg == 0:
                break
                
            removed.add(best_v)
            
            # Remove all triples containing best_v from active_triples
            to_remove = set(vertex_triples[best_v])
            active_triples -= to_remove
            
        S_try = set(candidates) - removed
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # 2. Try greedy addition on [66, 100] (original base)
    # This is kept as a fallback or to see if it finds something better
    base_S = set(range(66, 101))
    if is_valid(base_S):
        if len(base_S) > best_size:
            best_size = len(base_S)
            best_set = base_S

    # 3. Local search on best_set
    current_set = set(best_set)
    current_size = len(current_set)
    
    candidates = list(range(1, 101))
    random.seed(42)
    
    no_improvement = 0
    while time.time() < deadline - 2:
        improved = False
        not_in_set = [x for x in candidates if x not in current_set]
        
        # Try add
        if not_in_set:
            # Try to add a random element
            x = random.choice(not_in_set)
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
        
        # Try swap
        if current_set and not_in_set:
            u = random.choice(list(current_set))
            v = random.choice(not_in_set)
            if u != v:
                new_set = (current_set - {u}) | {v}
                if is_valid(new_set):
                    # Accept swap if it doesn't reduce size (it doesn't) 
                    # and potentially helps future adds.
                    # We accept all valid swaps to explore.
                    current_set = new_set
                    improved = True
        
        if not improved:
            no_improvement += 1
            if no_improvement > 100:
                break
        else:
            no_improvement = 0

    return list(best_set)
