import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    # Precompute triples
    triples = []
    elem_to_triples = defaultdict(list)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = frozenset((a, b, c))
                    triples.append(t)
                    elem_to_triples[a].append(t)
                    elem_to_triples[b].append(t)
                    elem_to_triples[c].append(t)
    
    # Remove duplicate triples if any (frozenset handles this if we use a set, but list is fine)
    # Actually, (a,b,c) and (a,c,b) are the same frozenset.
    # The generation loop ensures b > a. c is determined.
    # Is it possible to generate the same frozenset twice?
    # a=2, b=3 -> c=6. (2,3,6)
    # a=2, b=6 -> d=4, c=3. (2,6,3) -> frozenset(2,3,6).
    # So yes, duplicates are possible.
    unique_triples = list(set(triples))
    # Rebuild elem_to_triples with unique triples
    elem_to_triples = defaultdict(list)
    for t in unique_triples:
        for x in t:
            elem_to_triples[x].append(t)
    
    def is_valid(S):
        for t in unique_triples:
            if t.issubset(S):
                return False
        return True
        
    def can_add(S, u):
        # Check if adding u to S creates a triple
        for t in elem_to_triples[u]:
            # t contains u. Let other = t - {u}
            # If other is subset of S, then adding u creates triple
            if (t - {u}).issubset(S):
                return False
        return True
        
    def greedy_remove(S):
        S = set(S)
        # Count active triples for each element
        # To speed up, we can just scan triples
        changed = True
        while changed:
            changed = False
            # Find active triples
            active = [t for t in unique_triples if t.issubset(S)]
            if not active:
                break
            # Count conflicts
            conf_count = defaultdict(int)
            for t in active:
                for x in t:
                    conf_count[x] += 1
            # Find max
            max_conf = max(conf_count.values())
            candidates = [x for x in S if conf_count[x] == max_conf]
            # Random choice
            v = random.choice(candidates)
            S.remove(v)
            changed = True
        return S
        
    def local_search(S):
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1. Try to add any element
            for u in range(1, N + 1):
                if u not in S:
                    if can_add(S, u):
                        S.add(u)
                        improved = True
                        break
            
            if improved:
                continue
                
            # 2. Try 1-swap perturbation
            # Try removing each v in S and then greedily adding
            s_list = list(S)
            random.shuffle(s_list)
            for v in s_list:
                if time.time() > deadline:
                    return S
                
                S_temp = S - {v}
                # Greedily add elements to S_temp
                added = 0
                # We can try to add in a specific order or random
                candidates = [u for u in range(1, N + 1) if u not in S_temp]
                random.shuffle(candidates)
                for u in candidates:
                    if can_add(S_temp, u):
                        S_temp.add(u)
                        added += 1
                
                if len(S_temp) > len(S):
                    S = S_temp
                    improved = True
                    break
        
        return S
        
    best_S = set()
    best_size = 0
    
    # Multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        S = set(range(1, N + 1))
        S = greedy_remove(S)
        
        # Local search
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    return list(best_S)
