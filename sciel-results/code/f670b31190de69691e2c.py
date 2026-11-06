import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) distinct in [1, N] with 1/a = 1/b + 1/c
    triples = []
    seen = set()
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = tuple(sorted([a, b, c]))
                    if t not in seen:
                        seen.add(t)
                        triples.append(t)
    
    # Build adjacency: for each element, list of triple indices containing it
    element_triples = defaultdict(list)
    for i, t in enumerate(triples):
        for x in t:
            element_triples[x].append(i)
    
    # Build pairs for fast conflict checking
    # pairs[x] = list of (y, z) such that {x, y, z} is a triple
    pairs = defaultdict(list)
    for t in triples:
        a, b, c = t
        pairs[a].append((b, c))
        pairs[b].append((a, c))
        pairs[c].append((a, b))
    
    # Count how many triples each element participates in
    triple_count = [0] * (N + 1)
    for t in triples:
        for x in t:
            triple_count[x] += 1
            
    def can_add(x, S_bool):
        for y, z in pairs[x]:
            if S_bool[y] and S_bool[z]:
                return False
        return True
        
    # Strategy 1: Greedy removal from full set
    S = set(range(1, N + 1))
    S_bool = [False] * (N + 1)
    for x in S:
        S_bool[x] = True
        
    # Find all violated triples
    violated = set()
    for i, t in enumerate(triples):
        if all(x in S for x in t):
            violated.add(i)
    
    while violated and time.time() < deadline - 10:
        elem_viol = [0] * (N + 1)
        for i in violated:
            t = triples[i]
            for x in t:
                if x in S:
                    elem_viol[x] += 1
        
        max_count = 0
        best_elem = -1
        for x in range(1, N + 1):
            if x in S and elem_viol[x] > max_count:
                max_count = elem_viol[x]
                best_elem = x
        
        if best_elem == -1:
            break
        
        S.discard(best_elem)
        S_bool[best_elem] = False
        
        # Update violated set efficiently? 
        # Removing best_elem breaks all triples containing it.
        # We can just recompute violated set, it's O(T)
        violated = set()
        for i, t in enumerate(triples):
            if all(x in S for x in t):
                violated.add(i)
    
    best_set = set(S)
    best_size = len(S)
    
    # Try to add back elements that don't create violations
    removed = set(range(1, N + 1)) - S
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in list(removed):
            if time.time() > deadline - 10:
                break
            if can_add(x, S_bool):
                S.add(x)
                S_bool[x] = True
                removed.discard(x)
                improved = True
                best_size = len(S)
    
    best_set = set(S)
    best_size = len(S)
    
    # Strategy 2: Randomized greedy construction
    random.seed(12345)
    for restart in range(100):
        if time.time() > deadline - 15:
            break
        
        S2 = set()
        S2_bool = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 15:
                break
            if can_add(x, S2_bool):
                S2.add(x)
                S2_bool[x] = True
        
        # Try to add back
        remaining = set(range(1, N + 1)) - S2
        improved2 = True
        while improved2 and time.time() < deadline - 15:
            improved2 = False
            for x in list(remaining):
                if time.time() > deadline - 15:
                    break
                if can_add(x, S2_bool):
                    S2.add(x)
                    S2_bool[x] = True
                    remaining.discard(x)
                    improved2 = True
        
        if len(S2) > best_size:
            best_set = set(S2)
            best_size = len(S2)
            
    # Strategy 3: Weighted greedy
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        S3 = set()
        S3_bool = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        candidates.sort(key=lambda x: (triple_count[x], random.random()))
        
        for x in candidates:
            if time.time() > deadline - 15:
                break
            if can_add(x, S3_bool):
                S3.add(x)
                S3_bool[x] = True
        
        remaining = set(range(1, N + 1)) - S3
        improved3 = True
        while improved3 and time.time() < deadline - 15:
            improved3 = False
            for x in list(remaining):
                if time.time() > deadline - 15:
                    break
                if can_add(x, S3_bool):
                    S3.add(x)
                    S3_bool[x] = True
                    remaining.discard(x)
                    improved3 = True
        
        if len(S3) > best_size:
            best_set = set(S3)
            best_size = len(S3)
            
    # Strategy 4: Local Search with Kicks
    if time.time() < deadline - 5:
        current_set = set(best_set)
        current_bool = [False] * (N + 1)
        for x in current_set:
            current_bool[x] = True
            
        current_size = len(current_set)
        global_best_size = best_size
        global_best_set = set(best_set)
        
        no_improve_count = 0
        kick_size = 1
        
        while time.time() < deadline - 2:
            # Perform a move: Remove k random elements, then greedily add
            if no_improve_count > 50:
                # Increase kick size
                kick_size = min(kick_size + 1, 5)
                no_improve_count = 0
            
            # Remove kick_size random elements
            elems_to_remove = []
            if current_set:
                elems_to_remove = random.sample(list(current_set), min(kick_size, len(current_set)))
            
            for x in elems_to_remove:
                current_set.discard(x)
                current_bool[x] = False
                current_size -= 1
                
            # Greedily add all possible elements in random order
            candidates = [x for x in range(1, N + 1) if not current_bool[x]]
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, current_bool):
                    current_set.add(x)
                    current_bool[x] = True
                    current_size += 1
                    
            # Check improvement
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_set = set(current_set)
                no_improve_count = 0
                kick_size = 1
            else:
                no_improve_count += 1
                
            # If we removed elements but didn't add back enough, we might be stuck
            # But the greedy add should restore most of them.
            
    return list(global_best_set)
