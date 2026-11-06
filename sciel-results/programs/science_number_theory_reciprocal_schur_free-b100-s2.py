import time
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    N = 180
    
    # Precompute all forbidden triples: (a, b, c) distinct, 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => c = ab/(b-a) for b > a
    # We need a, b, c all in [1, N] and all distinct
    triples = []
    triple_count = [0] * (N + 1)  # count of triples each element is in
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Store as sorted tuple for uniqueness
                    t = tuple(sorted([a, b, c]))
                    triples.append(t)
                    triple_count[t[0]] += 1
                    triple_count[t[1]] += 1
                    triple_count[t[2]] += 1
    
    # Remove duplicates
    triples = list(set(triples))
    # Recompute counts
    triple_count = [0] * (N + 1)
    for t in triples:
        triple_count[t[0]] += 1
        triple_count[t[1]] += 1
        triple_count[t[2]] += 1
    
    # Build adjacency: for each element, which triples contain it
    element_triples = [[] for _ in range(N + 1)]
    for i, t in enumerate(triples):
        for x in t:
            element_triples[x].append(i)
    
    def is_valid(S):
        """Check if set S has no forbidden triple"""
        S_set = set(S)
        for t in triples:
            if all(x in S_set for x in t):
                return False
        return True
    
    # Greedy removal: start with all, remove elements in triples
    S = set(range(1, N + 1))
    
    # Count how many triples are currently violated
    def count_violations(S_set):
        count = 0
        for t in triples:
            if all(x in S_set for x in t):
                count += 1
        return count
    
    # Iteratively remove the element that is in the most violated triples
    violated = set()
    for i, t in enumerate(triples):
        if all(x in S for x in t):
            violated.add(i)
    
    while violated and time.time() < deadline - 5:
        # Find element in violated triples that appears most
        elem_viol_count = [0] * (N + 1)
        for i in violated:
            t = triples[i]
            for x in t:
                if x in S:
                    elem_viol_count[x] += 1
        
        # Pick element with max count
        max_count = 0
        best_elem = -1
        for x in range(1, N + 1):
            if x in S and elem_viol_count[x] > max_count:
                max_count = elem_viol_count[x]
                best_elem = x
        
        if best_elem == -1:
            break
        
        S.discard(best_elem)
        
        # Update violated set
        new_violated = set()
        for i, t in enumerate(triples):
            if all(x in S for x in t):
                new_violated.add(i)
        violated = new_violated
    
    best_set = set(S)
    best_size = len(S)
    
    # Try to add back elements (local search)
    removed = set(range(1, N + 1)) - S
    
    # Try adding back elements that don't create violations
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in list(removed):
            if time.time() > deadline - 5:
                break
            # Check if adding x creates any violation
            ok = True
            for i in element_triples[x]:
                t = triples[i]
                if all(y in S for y in t if y != x):
                    ok = False
                    break
            if ok:
                S.add(x)
                removed.discard(x)
                improved = True
                best_size = len(S)
    
    # Try multiple random restarts with different greedy orders
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random greedy: add elements in random order, skip if creates violation
        S2 = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 10:
                break
            # Check if adding x creates violation
            ok = True
            for i in element_triples[x]:
                t = triples[i]
                if all(y in S2 for y in t if y != x):
                    ok = False
                    break
            if ok:
                S2.add(x)
        
        if len(S2) > best_size:
            best_set = set(S2)
            best_size = len(S2)
    
    return list(best_set)
