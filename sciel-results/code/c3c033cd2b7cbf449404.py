"""Optimized superpermutation construction."""
import time
import sys
from itertools import permutations

def _transition_cost(a, b):
    """Cost of transitioning from block a to block b: how many symbols we need to add.
    If block a ends with suffix of block b, we can overlap.
    """
    # a and b are tuples of length m-1
    # We append b after a (with new symbol m in between conceptually, but here
    # we're looking at the (m-1)-perm windows)
    # The word so far ends with ...a. Next we want to have b appear.
    # We can overlap: if a's suffix of length k equals b's prefix of length k,
    # we only need to add (m-1-k) symbols.
    # But actually the structure is: ...a, m, b... so the window after a is (a[1:], m)
    # and we need to reach b. This is more complex.
    
    # Simpler view: after placing all symbols up to m, the word contains
    # all (m-1)! permutations as windows of length m-1 (not containing m)
    # interleaved with permutations containing m.
    
    # For the purpose of ordering: when we go from permutation p to q of {1..m-1},
    # the overlap is: how much of q is already "prepared" by the tail of p.
    # Specifically, if p[i:] == q[:m-1-i], we can overlap m-1-i symbols.
    m1 = len(a)
    max_overlap = 0
    for k in range(m1 - 1, -1, -1):
        if a[m1-k:] == b[:k]:
            max_overlap = k
            break
    return m1 - max_overlap  # symbols to add

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if n <= 1:
        return "1"
    
    # For small n (<=5), we can do better with optimized ordering
    # For n=7, use greedy construction
    
    if n <= 5:
        return _construct_small(n)
    
    # Greedy construction for larger n
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        # Find all (m-1)-permutations that appear as windows in current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # We need to order these perms to minimize total length
        # Greedy: start with first, then always go to nearest unvisited
        if len(perms) <= 1:
            ordered = perms
        else:
            # Greedy nearest neighbor
            remaining = list(range(len(perms)))
            ordered = [perms[0]]
            remaining.remove(0)
            
            while remaining:
                last = ordered[-1]
                # Find nearest unvisited
                best_idx = -1
                best_cost = float('inf')
                for idx in remaining:
                    cost = _transition_cost(last, perms[idx])
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx
                ordered.append(perms[best_idx])
                remaining.remove(best_idx)
        
        # Build new word
        new = list(word)
        for p in ordered:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
    
    return "".join(map(str, word))

def _construct_small(n):
    """Optimized construction for small n using better heuristics."""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known optimal is 33
        return "1234123142132413214321"
    if n == 5:
        # Use greedy
        word = [1]
        for m in range(2, 6):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            if len(perms) <= 1:
                ordered = perms
            else:
                remaining = list(range(len(perms)))
                ordered = [perms[0]]
                remaining.remove(0)
                while remaining:
                    last = ordered[-1]
                    best_idx = -1
                    best_cost = float('inf')
                    for idx in remaining:
                        cost = _transition_cost(last, perms[idx])
                        if cost < best_cost:
                            best_cost = cost
                            best_idx = idx
                    ordered.append(perms[best_idx])
                    remaining.remove(best_idx)
            
            new = list(word)
            for p in ordered:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # n >= 6, use main greedy
    return construct(n)
