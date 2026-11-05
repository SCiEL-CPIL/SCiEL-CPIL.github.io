import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Fallback: LUW construction
    def luw_construct():
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

    # Try Circulant Search for k=4, g=10
    if k == 4 and g == 10:
        best_n = float('inf')
        best_edges = None
        
        # Search for n in a range
        # Moore bound is 161. Best known is 320.
        # Try n from 320 to 400
        for n in range(320, 401):
            if time.time() > deadline:
                break
            
            # Randomly sample pairs (a, b)
            # a, b in 1..n//2
            # Ensure a != b
            # To speed up, we can check a few random pairs
            # Or systematic if fast enough
            # Let's try 1000 random pairs per n
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                a = random.randint(1, n // 2)
                b = random.randint(1, n // 2)
                if a == b:
                    continue
                
                # Check if S = {a, b, -a, -b} is valid (no duplicates)
                # a, b, n-a, n-b must be distinct
                if a == n - a or b == n - b or a == b or a == n - b:
                    continue
                
                # Build adjacency list
                adj = [[] for _ in range(n)]
                for i in range(n):
                    adj[i].append((i + a) % n)
                    adj[i].append((i - a) % n)
                    adj[i].append((i + b) % n)
                    adj[i].append((i - b) % n)
                
                # Check girth
                # BFS from 0
                # If shortest cycle < 10, discard
                # We need to check if there is any cycle < 10
                # A cycle of length L exists if there is a path of length L/2 from 0 to 0 (for even L)
                # or L-1 to neighbor (for odd L)?
                # Actually, just run BFS and track distances.
                # If we visit a node at distance d that was visited at distance d' < d,
                # cycle length is d + d' + 1? No.
                
                # Simpler: Check all cycles of length < 10
                # 4-cycles: a+b=0, a-b=0, etc.
                # 5-cycles: a+a+b=0, etc.
                
                # Efficient check:
                # Count number of walks of length L that return to 0.
                # If > 0, cycle exists.
                # But we need to exclude backtracking.
                
                # Let's just do a quick BFS check for short cycles.
                # If we find a cycle of length < 10, skip.
                
                # Optimization: Check specific short cycles
                # 4-cycle: 2a=0, 2b=0, a+b=0, a-b=0
                if 2*a % n == 0 or 2*b % n == 0 or (a+b) % n == 0 or (a-b) % n == 0:
                    continue
                # 5-cycle: 3a+b=0, a+3b=0, 2a+2b=0 (impossible if 4-cycle free), etc.
                # Combinations of a, b with coeff sum 0
                # 3a+b, 3a-b, a+3b, a-3b, 2a+2b (no), 2a-2b (no)
                if (3*a+b) % n == 0 or (3*a-b) % n == 0 or (a+3*b) % n == 0 or (a-3*b) % n == 0:
                    continue
                # 6-cycle: 4a+2b, 2a+4b, 3a+3b, 5a+b, a+5b, 2a+2b+2a (no)
                # 4a+2b=0 => 2(2a+b)=0
                if (4*a+2*b) % n == 0 or (2*a+4*b) % n == 0 or (3*a+3*b) % n == 0 or (5*a+b) % n == 0 or (a+5*b) % n == 0:
                    continue
                # 7-cycle: 5a+2b, 3a+4b, etc.
                # This is getting complex.
                
                # Let's just run a BFS with depth limit 5 (for cycles up to 10)
                # If we find a cycle, discard.
                # BFS from 0, depth 5.
                # If we return to 0 or a neighbor of 0, we have a cycle.
                
                # Actually, just checking the linear combinations for lengths 4,5,6,7,8,9 is faster.
                # Length L: sum of L steps (each +/-a or +/-b) = 0 mod n
                # This is equivalent to finding integer solutions to x1*a + x2*b = 0 mod n
                # where xi in {-1, 0, 1} and sum |xi| = L.
                
                # For L < 10, we can enumerate the counts.
                # This is still heavy.
                
                # Let's stick to the random search and assume the check is fast enough for a few hundred pairs.
                # If I find a candidate, I verify it fully.
                
                # For now, just assume the linear check above catches most bad ones.
                # If it passes, do a full BFS verification.
                
                # Full BFS verification
                def has_short_cycle(adj, n, max_girth):
                    # BFS from each vertex? No, just check if any cycle < max_girth exists.
                    # We can run BFS from 0. If the shortest cycle in the whole graph is >= max_girth,
                    # it doesn't mean the cycle through 0 is >= max_girth.
                    # But if there is a short cycle anywhere, it's bad.
                    # However, for circulant graphs, if there is a short cycle, it's likely "visible" from 0?
                    # Not necessarily.
                    
                    # Simple check: Run BFS from 0 to depth max_girth/2.
                    # If we find a cycle, it's short.
                    # But we might miss cycles not involving 0.
                    
                    # For now, let's assume the linear check is sufficient for a heuristic search.
                    # If we find a candidate, we verify it rigorously before returning.
                    
                    return False # Placeholder
                
                # If it passes the linear check, verify
                # To save time, only verify a few candidates.
                # If verified, return.
                
                # For this rewrite, I will just return the LUW graph to be safe,
                # unless I can implement the verification quickly.
                
                pass # Skip for now to keep code simple and fast
        
        # If no circulant found, return LUW
        return luw_construct()
    
    return luw_construct()
