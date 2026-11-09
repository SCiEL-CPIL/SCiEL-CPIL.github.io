import time
import random

def construct():
    deadline = time.time() + 55  # safety margin
    
    def is_valid_triple(a, b, c, S_set):
        """Check if triple (a,b,c) violates: all three pair sums in S_set."""
        return (a + b) in S_set and (a + c) in S_set and (b + c) in S_set
    
    def count_violations(S_set):
        """Count number of violating triples."""
        count = 0
        elems = list(S_set)
        n = len(elems)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = elems[i], elems[j], elems[k]
                    if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                        count += 1
        return count
    
    def can_add(x, S_set):
        """Check if adding x to S_set creates any violation."""
        elems = list(S_set)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                a, b = elems[i], elems[j]
                if (x+a) in S_set and (x+b) in S_set and (a+b) in S_set:
                    return False
        return True
    
    # Start with the safe core: 65..128
    base = list(range(65, 129))
    S = set(base)
    
    # Try adding elements from 1..64 in descending order (larger first, fewer conflicts)
    for x in range(64, 0, -1):
        if time.time() > deadline - 5:
            break
        if can_add(x, S):
            S.add(x)
    
    best_set = set(S)
    best_size = len(S)
    
    # Local search: try swapping elements to improve
    if time.time() < deadline - 10:
        random.seed(42)
        for _iter in range(200):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random element from 1..64 not in S
            candidates = [x for x in range(1, 65) if x not in S]
            if not candidates:
                break
            
            # Try each candidate in random order
            random.shuffle(candidates)
            improved = False
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                    break
            
            # Try removing one element and adding another
            if not improved and time.time() < deadline - 5:
                elems = list(S)
                random.shuffle(elems)
                for rem in elems[:5]:  # try removing up to 5 random elements
                    if rem >= 65:
                        continue  # don't remove from base
                    S.remove(rem)
                    # Now try to add multiple elements
                    for x in range(64, 0, -1):
                        if x not in S and can_add(x, S):
                            S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                    S.remove(rem) if rem in S else None
                    S.add(rem)  # put back
    
    return sorted(best_set)
