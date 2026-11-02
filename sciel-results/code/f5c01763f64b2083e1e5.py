import time
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses cyclic Cayley graphs Z_N with generators {a, -a, b}.
    Searches for the smallest N and generators that satisfy the girth constraint.
    """
    if k != 3:
        return _luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def _luw_construct(k, g):
        """LUW construction fallback."""
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

    def check_girth_walk(N, a, b, target_girth):
        """
        Checks if the circulant graph Z_N with generators {a, -a, b} has girth >= target_girth.
        Uses the property that girth is the shortest non-backtracking closed walk.
        Tracks sums of walks ending with each step type.
        """
        a_mod = a % N
        b_mod = b % N
        neg_a = (N - a_mod) % N
        neg_b = (N - b_mod) % N
        
        # Steps: 0: a, 1: -a, 2: b, 3: -b
        # Inverse map: 0<->1, 2<->3
        inverse = [1, 0, 3, 2]
        steps = [a_mod, neg_a, b_mod, neg_b]
        
        # D[s] is a set of sums of walks of current length L ending with step s
        # Initialize for L=1
        D = [set() for _ in range(4)]
        for s in range(4):
            D[s].add(steps[s])
        
        # Check L=1: cycle of length 1? No, simple graph.
        # Check L=2: cycle of length 2? No, simple graph.
        # We need to check L from 3 to target_girth - 1.
        
        for L in range(2, target_girth):
            new_D = [set() for _ in range(4)]
            for s in range(4):
                # To end with step s, the previous step t must not be inverse(s)
                # Sum = prev_sum + steps[s]
                for t in range(4):
                    if t == inverse[s]:
                        continue
                    # Add all sums from D[t] + steps[s] to new_D[s]
                    # Optimization: if sets are large, this is expensive.
                    # But for L < 16, size is small.
                    for val in D[t]:
                        new_D[s].add((val + steps[s]) % N)
            D = new_D
            
            # Check if 0 is in any D[s]
            for s in range(4):
                if 0 in D[s]:
                    return False
            
            # Early termination if sets stop growing? No, they wrap around.
            # But if we have seen 0, we return.
            
        return True

    def generate_circulant(N, a, b):
        """
        Generates edges for Z_N with generators a, -a, b.
        Returns (n, edges) or None if invalid.
        """
        a = a % N
        b = b % N
        
        # Check for simple graph conditions
        if a == 0 or b == 0:
            return None
        if a == (N - a) % N:  # 2a = 0
            return None
        if b == (N - b) % N:  # 2b = 0
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

    # Search strategy: try N from Moore bound upward
    # Use structured generator selection for better success rate
    
    random.seed(42)
    
    # Golden ratio conjugate for good distribution
    phi = (1 + 5**0.5) / 2
    
    # Determine start N
    # If g is large, Moore bound is large.
    # For k=3, g=16, M=3280.
    # But prompt says lower bound 512.
    # We will search from 512 up to a limit.
    # If g is small, start lower.
    start_N = 512
    if g <= 10:
        start_N = 30
    elif g <= 12:
        start_N = 100
    elif g <= 14:
        start_N = 300

    for N in range(start_N, 1500):
        if time.time() > deadline - 15:
            break
            
        # Try structured choices first, then random
        candidates = set()
        
        # Golden ratio based
        a_golden = int(N / phi) % N
        b_golden = int(N / (phi * phi)) % N
        if a_golden != 0 and b_golden != 0:
            candidates.add((a_golden, b_golden))
            candidates.add((b_golden, a_golden))
        
        # Try a few offsets
        for offset in range(-10, 11):
            a_try = (a_golden + offset) % N
            b_try = (b_golden + offset) % N
            if a_try != 0 and b_try != 0:
                candidates.add((a_try, b_try))
        
        # Add random candidates
        for _ in range(100):
            a_r = random.randint(1, N - 1)
            b_r = random.randint(1, N - 1)
            candidates.add((a_r, b_r))
        
        for a, b in candidates:
            if time.time() > deadline - 15:
                break
            
            # Quick filter: a, b must be distinct and not inverses
            if a == b or a == (N-b)%N or b == (N-a)%N:
                continue
            if a == 0 or b == 0:
                continue
            if a == (N-a)%N or b == (N-b)%N:
                continue

            # Fast girth check
            if check_girth_walk(N, a, b, g):
                res = generate_circulant(N, a, b)
                if res is not None:
                    return res

    # Fallback to LUW
    return _luw_construct(k, g)
