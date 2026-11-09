import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe
    
    N = 180
    
    # Enumerate all forbidden triples (a, b, c) with a < b < c and a = bc/(b+c)
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b:
                    triples.append((a, b, c))
    
    # Build adjacency: for each element x, list of (other1, other2) such that (x, other1, other2) is a triple
    adj = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in triples:
        adj[a].append((b, c))
        adj[b].append((a, c))
        adj[c].append((a, b))
    
    def greedy_remove(S_set, shuffled_triples=None):
        """Remove elements to eliminate all violations."""
        if shuffled_triples is None:
            shuffled_triples = triples
        changed = True
        while changed:
            changed = False
            violated = []
            for (a, b, c) in shuffled_triples:
                if a in S_set and b in S_set and c in S_set:
                    violated.append((a, b, c))
            
            if not violated:
                break
            
            freq = {}
            for (a, b, c) in violated:
                for x in (a, b, c):
                    freq[x] = freq.get(x, 0) + 1
            
            worst = max(freq.keys(), key=lambda x: freq[x])
            S_set.remove(worst)
            changed = True
        
        return S_set
    
    def try_add_back(S_set):
        """Try to add back elements that don't create violations."""
        all_elements = set(range(1, N + 1))
        removed = all_elements - S_set
        improved = True
        while improved:
            improved = False
            for x in sorted(removed):
                can_add = True
                for (other1, other2) in adj[x]:
                    if other1 in S_set and other2 in S_set:
                        can_add = False
                        break
                if can_add:
                    S_set.add(x)
                    improved = True
        return S_set
    
    # Initial solution
    best_S = set(range(1, N + 1))
    best_S = greedy_remove(best_S)
    best_S = try_add_back(best_S)
    best_size = len(best_S)
    
    # Random restarts
    random.seed(42)
    
    while time.time() < deadline:
        shuffled = triples[:]
        random.shuffle(shuffled)
        
        S_set = set(range(1, N + 1))
        changed = True
        while changed:
            changed = False
            violated = []
            for (a, b, c) in shuffled:
                if a in S_set and b in S_set and c in S_set:
                    violated.append((a, b, c))
                    if len(violated) > 50:
                        break
            
            if not violated:
                break
            
            freq = {}
            for (a, b, c) in violated:
                for x in (a, b, c):
                    freq[x] = freq.get(x, 0) + 1
            
            if freq:
                max_f = max(freq.values())
                candidates = [x for x, f in freq.items() if f == max_f]
                worst = random.choice(candidates)
                S_set.remove(worst)
                changed = True
        
        S_set = try_add_back(S_set)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S = S_set
        
        if time.time() > deadline - 3:
            break
    
    return list(best_S)
