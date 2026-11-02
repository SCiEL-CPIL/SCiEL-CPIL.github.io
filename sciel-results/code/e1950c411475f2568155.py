import time
import random
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def _luw_construct(k, g):
        # Fallback to the original LUW construction from the seed
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
                    aa, bb = rule[c]
                    y[c] = add(x[c], mul(y[aa], x[bb])) if side == 0 else sub(x[c], mul(x[aa], y[bb]))
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

    def check_girth(N, a, b, target_girth):
        """
        Checks if circulant graph Z_N with generators a, b has girth >= target_girth.
        Uses BFS on non-backtracking walks.
        """
        a_mod = a % N
        b_mod = b % N
        
        # Steps: 0: a, 1: -a, 2: b, 3: -b
        steps = [a_mod, (N - a_mod) % N, b_mod, (N - b_mod) % N]
        inverse = [1, 0, 3, 2]
        
        # State: (sum, last_step_index)
        # We want to find if 0 is reachable with length < target_girth
        # Start states: length 1
        # visited[sum][last_step] = length
        # Since N is small (< 4096) and target_girth is 16, we can use a set of visited states per level.
        
        # Level 1
        current_states = set()
        for s in range(4):
            current_states.add((steps[s], s))
            
        # If N is small, we might hit 0 at length 1? No, simple graph.
        # Check length 2, 3, ... up to target_girth - 1
        
        for L in range(2, target_girth):
            next_states = set()
            for (sum_val, last_s) in current_states:
                for s in range(4):
                    if s == inverse[last_s]:
                        continue
                    new_sum = (sum_val + steps[s]) % N
                    # If we return to 0, we found a cycle of length L
                    if new_sum == 0:
                        return False
                    next_states.add((new_sum, s))
            current_states = next_states
            if not current_states:
                # No walks of this length? Should not happen for connected graph
                break
                
        return True

    def generate_circulant(N, a, b):
        a = a % N
        b = b % N
        
        if a == 0 or b == 0:
            return None
        if a == (N - a) % N:
            return None
        if b == (N - b) % N:
            return None
        if a == b or a == (N - b) % N or (N - a) % N == b or (N - a) % N == (N - b) % N:
            return None
        
        edges = set()
        for i in range(N):
            for j in [(i + a) % N, (i - a) % N, (i + b) % N]:
                if i == j:
                    return None
                u, v = min(i, j), max(i, j)
                edges.add((u, v))
                
        # Verify degree
        deg = [0] * N
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != 3:
                return None
                
        return N, list(edges)

    # Search for smallest N
    # Start from Moore bound 510, round up to even
    start_N = 512
    # We can search up to 4096 (the LUW size)
    # If we find nothing, we return LUW.
    
    # To speed up, we can skip N that are not "good" candidates?
    # For now, iterate N.
    # For each N, try a few random (a,b) pairs.
    
    random.seed(42)
    
    for N in range(start_N, 4097, 2):
        if time.time() > deadline - 10:
            break
            
        # Try a limited number of candidates for this N
        # 1. Structured: a=1, b=k? No.
        # 2. Random
        found = False
        # Try 50 random pairs
        for _ in range(50):
            a = random.randint(1, N - 1)
            b = random.randint(1, N - 1)
            
            # Quick filters
            if a == b or a == (N-b)%N or b == (N-a)%N:
                continue
            if a == (N-a)%N or b == (N-b)%N:
                continue
            if a == 0 or b == 0:
                continue
                
            if check_girth(N, a, b, g):
                res = generate_circulant(N, a, b)
                if res is not None:
                    return res
                    
    # Fallback to LUW
    return _luw_construct(k, g)
