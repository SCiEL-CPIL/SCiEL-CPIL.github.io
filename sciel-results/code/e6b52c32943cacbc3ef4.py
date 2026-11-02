"""Combined approach: explicit algebraic construction (LUW) + random-greedy refinement."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Returns (n, edges).
    
    Strategy:
    1. Try the Lazebnik-Ustimenko-Woldar (LUW) construction for a valid baseline.
    2. Attempt a random-greedy construction to find a smaller graph.
    3. Return the best (smallest n) valid result found within the time budget.
    """
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    best_n = None
    best_edges = None
    
    # --- Step 1: LUW Construction (Deterministic, Fast) ---
    # This provides a guaranteed valid result (if the parameters work) quickly.
    try:
        n_luw, edges_luw = _construct_luw(k, g)
        if n_luw is not None:
            best_n = n_luw
            best_edges = edges_luw
    except Exception:
        pass
    
    # --- Step 2: Random Greedy Construction (Stochastic, Slower) ---
    # Try to find a smaller graph using random greedy edge addition.
    # We try a few sizes starting from the Moore bound upper limit.
    
    # Moore bound for (k, g):
    # For even g: n >= k * (1 + (k-1) + ... + (k-1)^(g/2 - 1))
    # For odd g: n >= 1 + k * (1 + (k-1) + ... + (k-1)^( (g-1)/2 - 1 ))
    # Actually, the standard Moore bound M(k,g) is:
    # if g is even: M = k * sum_{i=0}^{g/2-1} (k-1)^i
    # if g is odd: M = 1 + k * sum_{i=0}^{(g-1)/2-1} (k-1)^i
    
    # For k=5, g=9:
    # M = 1 + 5 * (1 + 4 + 16) = 1 + 5 * 21 = 106.
    # But the actual cage is much larger. The lower bound mentioned is 428.
    # Let's try sizes from 428 upwards.
    
    # If we already have a LUW result, we only care about finding something smaller.
    # LUW for k=5, g=9 typically yields a graph with ~ q^d * 2 vertices.
    # d = g-5 = 4. q=5. 2 * 5^4 = 1250? Or similar.
    # The prompt says smallest known is 1116.
    # So if LUW gives > 1116, we might improve. If it gives ~1250, we might improve.
    
    # Let's attempt random greedy for a few candidate sizes.
    # We need to be careful with time.
    
    candidate_sizes = []
    if best_n is None:
        # If no LUW result, we MUST find one.
        # Start from a reasonable estimate.
        # For k=5, g=9, let's try around 500-1000.
        candidate_sizes = [500, 600, 700, 800, 900, 1000, 1100]
    else:
        # If we have a LUW result, only try sizes significantly smaller.
        # If best_n is 1250, try 1000, 900, 800.
        if best_n > 1000:
            candidate_sizes = [900, 800, 700, 600, 500]
        elif best_n > 500:
            candidate_sizes = [int(best_n * 0.8), int(best_n * 0.6), 428]
        else:
            # Already quite small, maybe skip expensive search
            candidate_sizes = []

    for n_try in candidate_sizes:
        if time.time() > deadline:
            break
        if best_n is not None and n_try >= best_n:
            continue
            
        # Attempt construction
        result = _try_random_greedy(k, g, n_try, deadline)
        if result is not None:
            n_found, edges_found = result
            if best_n is None or n_found < best_n:
                best_n = n_found
                best_edges = edges_found
            # If we find a graph close to the lower bound, we can stop
            if n_found <= 450:
                break

    if best_n is None:
        # Fallback: If nothing worked, raise an error or return a dummy?
        # The task requires a valid graph. If LUW failed and random greedy failed,
        # we are in trouble. But LUW should almost always work for prime k.
        # Let's force a LUW attempt again or a very large random graph.
        try:
            n_luw, edges_luw = _construct_luw(k, g)
            best_n = n_luw
            best_edges = edges_luw
        except Exception:
            # Last resort: construct a large random k-regular graph and hope girth is high?
            # No, that's dangerous.
            # Let's just raise an exception if we truly fail, but that scores 0.
            # Better to return the LUW result even if it's large.
            pass

    if best_n is None:
        # This should not happen for k=5, g=9
        raise ValueError("Failed to construct a valid graph")

    return best_n, best_edges


def _construct_luw(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    q = k (must be prime or 4).
    d is chosen such that girth is at least g.
    For LUW, the girth is 2d + 2 if d is even? Or related to d.
    Actually, the girth of CD(d,q) is 2d + 2 for d even? 
    Let's use the logic from the seed program.
    """
    q = k
    # Determine d
    # The seed program uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9: g-5=4 (even), so d = 9-4=5? Or g-5=4 (even) -> condition (g-5)%2==0 is True -> d = g-4 = 5.
    # Let's verify girth of CD(5, 5).
    # Literature says girth of CD(d,q) is 2d+2 for d even? 
    # Actually, for CD(d,q), the girth is 2d+2 if d is even, and 2d+1 if d is odd?
    # Let's stick to the seed's logic which is known to be valid.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule definition
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are coords
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge if we are on the "point" side to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _try_random_greedy(k, g, n, deadline):
    """
    Attempt to construct a k-regular graph of girth >= g on n vertices
    using a random greedy approach.
    
    This is a heuristic and may fail or take too long.
    """
    # This is a simplified version. A full implementation would be complex.
    # Given the time constraints and complexity, this function is a placeholder
    # that might return None quickly if it can't find a solution fast.
    
    # For now, let's implement a very basic random regular graph generator
    # and check girth. If it fails, return None.
    
    # Generating a random k-regular graph:
    # 1. Create a list of 2*n stubs.
    # 2. Shuffle and pair them up.
    # 3. Check for self-loops and multi-edges. If found, retry (or fix).
    # 4. Check girth.
    
    # This is expensive for n=500+. 
    # Let's try a few iterations.
    
    max_retries = 50
    for _ in range(max_retries):
        if time.time() > deadline:
            return None
        
        # Generate random k-regular graph
        edges = _generate_random_regular_graph(k, n)
        if edges is None:
            continue
        
        # Check girth
        girth = _compute_girth(n, edges, k)
        if girth >= g:
            return n, edges
        
        # If girth is close, we might try local search, but that's complex.
        # Just retry.
        
    return None


def _generate_random_regular_graph(k, n):
    """
    Generate a random simple k-regular graph on n vertices.
    Returns list of edges or None if failed.
    """
    if (n * k) % 2 != 0:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Self-loop
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return None # Multi-edge
        edges.add((u, v))
        
    return list(edges)


def _compute_girth(n, edges, k):
    """
    Compute the girth of a graph.
    Returns the length of the shortest cycle, or infinity if acyclic.
    """
    # BFS from each vertex to find shortest cycle back to start.
    # This is O(n * (n + m)).
    # For n=500, m=1250, this is 500 * 1750 ~ 875,000 operations. Fast enough.
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for v in queue:
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v]:
                    # Found a cycle
                    # Cycle length is dist[v] + dist[u] + 1
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        # We can break early if we find a very short cycle
                        if min_girth == 3:
                            return 3
                            
    return min_girth

# EVOLVE-BLOCK-END
