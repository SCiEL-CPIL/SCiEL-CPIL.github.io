import time
import random

def construct():
    deadline = time.time() + 85
    
    # Precompute forbidden pairs for each a in [1,180]
    # For a: forbidden (b,c) with b>a, c>a, b!=c, (b-a)(c-a)=a^2
    forbidden_pairs = {}
    for a in range(1, 181):
        a2 = a * a
        pairs = []
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if c > a and c != b:
                    pairs.append((b, c))
        forbidden_pairs[a] = pairs
    
    def has_triple_with_a(a, S_set):
        for (b, c) in forbidden_pairs[a]:
            if b in S_set and c in S_set:
                return True
        return False
    
    # Start with 91-180
    S = set(range(91, 181))
    
    # Greedy add
    for a in range(1, 91):
        if time.time() > deadline:
            break
        if a not in S and not has_triple_with_a(a, S):
            S.add(a)
    
    for a in range(1, 181):
        if time.time() > deadline:
            break
        if a not in S and not has_triple_with_a(a, S):
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    # Local search
    random.seed(42)
    excluded = [x for x in range(1, 181) if x not in S]
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        a = random.choice(excluded)
        if not has_triple_with_a(a, S):
            S.add(a)
            excluded.remove(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        else:
            if S:
                to_remove = random.choice(list(S))
                S.remove(to_remove)
                if not has_triple_with_a(a, S):
                    S.add(a)
                    excluded.remove(a)
                    excluded.append(to_remove)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    S.add(to_remove)
    
    return sorted(best_S)
