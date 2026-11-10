import time
import random
import numpy as np
from collections import deque

def _check_girth_circulant(n, S):
    # S is a list of 4 symmetric offsets
    # Build adjacency matrix
    # For speed, we can just do BFS on the circulant structure directly without building the full matrix
    # d(u, v) is min(|u-v|, n-|u-v|) mod n ? No, it's the shortest path in the Cayley graph.
    
    # BFS from 0 is enough for vertex-transitive graphs (Circulants are vertex-transitive).
    # If girth is < 10, it will show up from 0.
    
    # We need to verify that the ball of radius 4 around 0 has exactly 1 + 4 + 12 + 36 + 108 = 161 nodes.
    # And that there are no collisions (shortcuts).
    
    # Let's do BFS from 0.
    # dist[0] = 0
    # visited = {0}
    # queue = [(0, 0)]
    # counts = [0]*5
    # counts[0] = 1
    # while queue:
    #   u, d = queue.pop(0)
    #   if d == 4: continue
    #   for s in S:
    #      v1 = (u + s) % n
    #      v2 = (u - s) % n
    #      for v in (v1, v2):
    #          if v not in visited:
    #              visited.add(v)
    #              dist[v] = d + 1
    #              counts[d+1] += 1
    #              queue.append((v, d+1))
    #          elif v == 0 and d+1 < 10:
    #              # Found a cycle?
    #              # If we hit 0 at distance < 10, we have a cycle.
    #              # But BFS finds shortest path.
    #              # If 0 is reached at dist 4, then there is a cycle of length 4?
    #              # No, if 0 is reached at dist 4, it means there is a path of length 4 from 0 to 0.
    #              # That is a cycle of length 4.
    #              return False
    #   # Also need to check if we hit a previously visited node at a shorter distance?
    #   # BFS handles this: if v is in visited, it was reached at a shorter or equal distance.
    #   # If v is in visited and dist[v] < d+1, it's a shortcut (cycle).
    #   # Actually, if we find a neighbor v that is already visited, and dist[v] != d+1, 
    #   # then we have a cycle of length dist[v] + 1?
    #   # Wait, if dist[v] < d+1, then there is a path of length dist[v] and a path of length d+1.
    #   # The cycle length is dist[v] + d + 1? No.
    #   # The cycle length is dist[v] + 1 (if we close the loop to v's parent)?
    #   # Simpler: If the number of nodes at distance k is less than the tree count, girth < 2k+1?
    #   # For girth 10, we need tree structure up to depth 4.
    #   # Tree counts: 1, 4, 12, 36, 108.
    #   # If counts[1] != 4, fail.
    #   # If counts[2] != 12, fail.
    #   # If counts[3] != 36, fail.
    #   # If counts[4] != 108, fail.
    #   # This is sufficient for girth >= 9?
    #   # If there is a cycle of length 8, it will reduce the count at depth 4?
    #   # Yes.
    
    # So we just check the counts.
    
    counts = [0]*5
    counts[0] = 1
    visited = {0}
    # Use a queue for BFS
    # For speed, use a list as queue
    queue = [0]
    dists = {0: 0}
    # We need to process level by level to know the depth
    current_level = [0]
    next_level = []
    for depth in range(1, 5):
        for u in current_level:
            for s in S:
                v1 = (u + s) % n
                v2 = (u - s) % n
                for v in (v1, v2):
                    if v not in visited:
                        visited.add(v)
                        dists[v] = depth
                        next_level.append(v)
                    else:
                        # If v is already visited, it might be a cycle.
                        # But we just check the count at the end.
                        pass
        counts[depth] = len(next_level)
        if len(next_level) == 0:
            break
        current_level = next_level
        next_level = []
        
    # Expected counts for 4-regular tree:
    # depth 0: 1
    # depth 1: 4
    # depth 2: 12
    # depth 3: 36
    # depth 4: 108
    if counts[1] != 4: return False
    if counts[2] != 12: return False
    if counts[3] != 36: return False
    if counts[4] != 108: return False
    return True

def _luw_construct(k, g):
    # Same as seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 110
    best_n = None
    best_edges = None
    seed_result = _luw_construct(k, g)
    best_n, best_edges = seed_result
    
    # Try to find a smaller circulant graph
    # n must be even. Start from 244.
    for n in range(244, 1000, 2):
        if time.time() > deadline:
            break
        
        max_val = n // 2 - 1
        if max_val < 2:
            continue
        
        # Adaptive number of tries: more for smaller n (more likely to succeed)
        # Smaller n has fewer valid candidates, so we need more tries
        if n < 300:
            max_tries = 2000
        elif n < 400:
            max_tries = 1500
        elif n < 600:
            max_tries = 1000
        else:
            max_tries = 800
        
        for _ in range(max_tries):
            if time.time() > deadline:
                break
            
            a = random.randint(1, max_val)
            b = random.randint(1, max_val)
            if a == b:
                continue
            # Avoid trivial 4-cycles: a + a == b (mod n) means 2a = b or 2a = n-b
            # This creates a 4-cycle: 0 -> a -> 2a -> a -> 0 ... wait
            # Actually 4-cycle: 0, a, b, b-a, 0 requires b-a in S or -(b-a) in S
            # Simpler filter: avoid a+b == 0 mod n (always true since we use ±), 
            # and avoid 2a == b mod n (gives 4-cycle 0-a-b-(b-a)-0 if b-a is also a generator)
            # Let's just filter: if (2*a) % n == b or (2*a) % n == n - b: skip
            # Also if (a+b) % n == a or (a+b)%n == b: trivial
            # The main issue: 4-cycle exists if {a, b, b-a, a-b} or similar small combos
            # For now, skip if 2a ≡ b or 2a ≡ -b (mod n)
            if (2 * a) % n == b or (2 * a) % n == n - b:
                continue
            if (2 * b) % n == a or (2 * b) % n == n - a:
                continue
            # Also skip if a+b is a generator (creates 4-cycle)
            # a+b mod n should not be a, b, n-a, n-b
            ab = (a + b) % n
            if ab in (a, b, n - a, n - b):
                continue
            ab2 = (a - b) % n
            if ab2 in (a, b, n - a, n - b):
                continue
            
            S = [a, n - a, b, n - b]
            
            # Check girth
            if _check_girth_circulant(n, S):
                # Found a valid graph
                edges = []
                for u in range(n):
                    for s in S:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    return best_n, best_edges
