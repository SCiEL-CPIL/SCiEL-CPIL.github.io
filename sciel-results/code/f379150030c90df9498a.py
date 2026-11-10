import time
import random

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that (i*j) % (i+j) == 0
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def is_valid(s):
        for a in s:
            if conflicts[a] & s:
                return False
        return True

    def greedy_fill(s, candidates, order):
        """Add elements from candidates (filtered by order) to s if valid."""
        s = set(s)
        for e in order:
            if e in s:
                continue
            if not (conflicts[e] & s):
                s.add(e)
        return s

    def local_search(s, max_iters=1000):
        """Local search with 1-swap and add."""
        s = set(s)
        best_size = len(s)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False
            
            # 1. Try to add any non-member
            added = False
            for b in range(1, n + 1):
                if b in s:
                    continue
                if not (conflicts[b] & s):
                    s.add(b)
                    best_size = len(s)
                    improved = True
                    added = True
                    break
            if added:
                continue
                
            # 2. Try 1-swap: remove a, add b
            # Iterate over all b not in s
            for b in range(1, n + 1):
                if b in s:
                    continue
                conflicts_with_b = conflicts[b] & s
                if not conflicts_with_b:
                    # Should have been caught above
                    continue
                
                # Try replacing each conflicting element a
                for a in list(conflicts_with_b):
                    # Check if S - {a} + {b} is valid
                    # b must not conflict with S - {a}
                    # b conflicts with S only via conflicts_with_b
                    # So we just need to ensure b doesn't conflict with S - conflicts_with_b?
                    # No, b might conflict with other elements in S?
                    # conflicts[b] & s is exactly the set of elements in S that conflict with b.
                    # If we remove a, does b still conflict with S?
                    # Yes, if there are other elements in conflicts_with_b.
                    # So S - {a} + {b} is valid iff conflicts_with_b == {a}.
                    # Wait, if conflicts_with_b has size > 1, removing one 'a' leaves others.
                    # So 1-swap only works if b conflicts with EXACTLY one element in S.
                    if len(conflicts_with_b) == 1:
                        # Check if new set is valid
                        # new_s = s - {a} | {b}
                        # We know b doesn't conflict with s - conflicts_with_b.
                        # And a is the only conflict.
                        # So new_s is valid.
                        s = s - {a} | {b}
                        if len(s) > best_size:
                            best_size = len(s)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return s, best_size

    best = []
    best_size = 0
    
    # Phase 1: Initial Construction
    odds = set(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))
    
    # Sort evens by number of conflicts with odds
    def conflict_count(e):
        return len(conflicts[e] & odds)
    
    evens_sorted = sorted(evens, key=conflict_count)
    
    s = set(odds)
    for e in evens_sorted:
        if not (conflicts[e] & s):
            s.add(e)
    
    s, sz = local_search(s, max_iters=2000)
    if sz > best_size:
        best = list(s)
        best_size = sz
        
    # Phase 2: Restart with Randomized Kick
    all_elements = list(range(1, n + 1))
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
            
        # Start from best or random?
        # Let's start from best and kick
        s = set(best)
        
        # Kick: remove K random elements
        K = random.randint(5, 15)
        to_remove = random.sample(list(s), min(K, len(s)))
        for r in to_remove:
            s.discard(r)
            
        # Greedily re-add from all elements not in s, in random order
        remaining = [e for e in all_elements if e not in s]
        random.shuffle(remaining)
        
        for e in remaining:
            if not (conflicts[e] & s):
                s.add(e)
                
        # Local search
        s, sz = local_search(s, max_iters=1000)
        
        if sz > best_size:
            best = list(s)
            best_size = sz
            
    return best
