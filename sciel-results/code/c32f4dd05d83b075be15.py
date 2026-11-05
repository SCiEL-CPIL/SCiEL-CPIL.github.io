import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    squares = set()
    i = 1
    while i * i <= 512:
        squares.add(i * i)
        i += 1
    
    # Self-conflicting: 2a is a perfect square => a = 2*m^2
    self_conflict = set()
    m = 1
    while 2 * m * m <= 256:
        self_conflict.add(2 * m * m)
        m += 1
    
    # Build adjacency bitsets (0-indexed)
    adj = [0] * n
    for a in range(1, 257):
        for b in range(a, 257):
            if a + b in squares:
                ai = a - 1
                bi = b - 1
                adj[ai] |= (1 << bi)
                adj[bi] |= (1 << ai)
    
    # Available vertices
    available = (1 << n) - 1
    for v in self_conflict:
        available &= ~(1 << (v - 1))
    
    def greedy():
        avail_list = []
        mask = available
        while mask:
            lowbit = mask & (-mask)
            idx = lowbit.bit_length() - 1
            avail_list.append(idx)
            mask ^= lowbit
        
        avail_list.sort(key=lambda v: (adj[v] & available).bit_count())
        
        selected = set()
        for v in avail_list:
            if all(not (adj[v] & (1 << u)) for u in selected):
                selected.add(v)
        return selected
    
    def add_max(s):
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if v in s or not (available & (1 << v)):
                    continue
                if all(not (adj[v] & (1 << u)) for u in s):
                    s.add(v)
                    improved = True
        return s
    
    best_set = greedy()
    best_set = add_max(best_set)
    best_size = len(best_set)
    
    # Local search: try 1-swap (remove one, add one) and 2-swap (remove one, add two)
    for _ in range(200000):
        if time.time() > deadline:
            break
        
        s = set(best_set)
        if not s:
            break
        
        # Randomly pick a vertex to remove
        removed = random.choice(list(s))
        s.remove(removed)
        
        # Find all vertices that can be added to s
        addable = []
        for v in range(n):
            if v in s or not (available & (1 << v)):
                continue
            if all(not (adj[v] & (1 << u)) for u in s):
                addable.append(v)
        
        # Try to add 1 or 2 vertices
        random.shuffle(addable)
        added = 0
        for v in addable:
            if added >= 2:
                break
            if all(not (adj[v] & (1 << u)) for u in s):
                s.add(v)
                added += 1
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            best_set = add_max(best_set)
            best_size = len(best_set)
    
    result = sorted(v + 1 for v in best_set)
    if not result:
        result = [1]
    return result
