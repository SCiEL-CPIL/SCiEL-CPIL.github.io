"""Construct k-regular graph of girth g using random Cayley graphs on (Z/2Z)^d."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try random Cayley graphs on (Z/2Z)^d with k involutory generators.
    Fall back to L-U-W construction if no valid graph is found in time.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_budget = 100.0  # Leave 20s margin for verification
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as fallback
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    # Verify girth using non-backtracking BFS from identity
    def verify_girth_cayley(gens, g, d):
        """Check if Cayley graph on (Z/2Z)^d with generators gens has girth >= g."""
        # Non-backtracking BFS: state is (vertex, prev_gen_idx)
        # We check if identity (0) is reachable via non-backtracking walk of length < g
        # that doesn't immediately backtrack
        
        # For a Cayley graph on (Z/2Z)^d, vertex v has neighbors v^gens[i] for i in range(k)
        # A non-backtracking walk starting from 0: 0 -> g0 -> g0^g1 -> ... 
        # where g_{i} != g_{i-1} (can't use same generator twice in a row)
        
        # We want to check if there's a cycle of length < g
        # A cycle of length L means a non-backtracking walk of length L from 0 back to 0
        
        # BFS over states (vertex, last_gen_used)
        # Start: for each gen i, state (gens[i], i) at depth 1
        # Goal: reach state (0, i) for any i with depth < g
        
        min_depth = {}
        queue = deque()
        
        for i in range(len(gens)):
            v = gens[i]
            state = (v, i)
            min_depth[state] = 1
            queue.append((v, i, 1))
        
        while queue:
            v, last_gen, depth = queue.popleft()
            
            if depth >= g - 1:
                continue
            
            for gi in range(len(gens)):
                if gi == last_gen:
                    continue
                new_v = v ^ gens[gi]
                new_state = (new_v, gi)
                new_depth = depth + 1
                
                if new_state not in min_depth:
                    min_depth[new_state] = new_depth
                    if new_v == 0 and new_depth < g:
                        return False  # Found a cycle shorter than g
                    queue.append((new_v, gi, new_depth))
        
        return True
    
    def generate_random_generators(d, k):
        """Generate k random distinct non-zero elements of (Z/2Z)^d."""
        gens = set()
        while len(gens) < k:
            x = random.randint(1, (1 << d) - 1)
            gens.add(x)
        return list(gens)
    
    def build_edges_cayley(gens, d):
        """Build edge list for Cayley graph on (Z/2Z)^d with generators gens."""
        n = 1 << d
        edges = []
        for v in range(n):
            for gen in gens:
                u = v ^ gen
                if v < u:
                    edges.append((v, u))
        return n, edges
    
    # Try random Cayley graph search
    best_result = None
    best_n = float('inf')
    
    # Try various dimensions
    # For girth g, we need the diameter of the non-backtracking graph to be at least (g-1)/2
    # The Cayley graph on (Z/2Z)^d with k generators has n = 2^d vertices
    # We want n as small as possible but >= Moore bound
    
    # Start from dimensions that give n close to Moore bound
    min_d = max(1, (mb).bit_length() - 1)  # 2^min_d >= mb roughly
    max_d = min(20, 15)  # Limit dimension for performance
    
    random.seed(42)
    
    for d in range(min_d, max_d + 1):
        if time.time() - start_time > time_budget:
            break
        
        n = 1 << d
        if n < mb:
            continue
        
        # Try multiple random generator sets
        trials = 2000 if d <= 10 else 500
        
        for trial in range(trials):
            if time.time() - start_time > time_budget:
                break
            
            gens = generate_random_generators(d, k)
            
            # Quick check: verify girth
            if verify_girth_cayley(gens, g, d):
                if n < best_n:
                    best_n = n
                    best_result = build_edges_cayley(gens, d)
        
        # If we found a valid graph with n <= L-U-W size, we can stop early
        # But let's continue to try smaller d values first (we're going in increasing d)
        # Actually we should try to find the smallest n, so increasing d is correct
    
    # Fall back to L-U-W if no better Cayley graph found
    luw_n, luw_edges = luw_construct(k, g)
    
    if best_result is not None and best_n <= luw_n:
        return best_result
    else:
        return luw_n, luw_edges
# EVOLVE-BLOCK-END
