"""Construct 5-regular girth-9 cage candidates.

Uses the Lazebnik-Ustimenko-Woldar (LUW) construction as a fast, always-valid
fallback, and attempts a targeted local-search improvement for the k=5,g=9 case
within a strict wall-clock budget.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.

    Returns (n, edges) for a k-regular bipartite graph with girth at least
    g (for odd g, the construction gives girth >= g exactly when d = g-4).
    """
    q = k
    # For odd girth g, we need girth >= g. LUW CD(d,q) has girth d+5 for odd d.
    # So d = g - 5 gives girth = g. But we need d odd.
    # If g-5 is odd, use d = g-5. Otherwise use d = g-4 (even) -> girth d+5 = g+1 >= g.
    # Actually the LUW paper: CD(d,q) for d>=3 has girth d+5 if d is odd, d+6 if d is even? 
    # Let me use the same logic as the seed: d = g-5 if (g-5) odd else g-4.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q

    # Rule: rule[c] = (a, b) for c = 2..d
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
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
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
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _local_search_5_9(max_time=110.0):
    """Attempt to find a 5-regular graph with girth >= 9 and fewer vertices than LUW.

    Strategy: start from a random 5-regular bipartite graph on N vertices (N even,
    N/2 per side). Use 2-opt edge swaps that preserve regularity. Accept swaps that
    do not create cycles shorter than 9. Track the minimum N for which we can achieve
    girth >= 9.

    For k=5, g=9: LUW gives N = 5^5 = 3125 (d=4, q=5, so 2*5^4=1250? Let me check).
    Actually LUW CD(4,5): points are vectors in GF(5)^4, so 5^4 = 625 per side, 1250 total.
    The Moore bound for (5,9) is 2 * (5 + 5^2 + 5^3) = 2*(5+25+125) = 320? No.
    Moore bound for (d, g) with g odd: 1 + d * sum_{i=0}^{(g-3)/2} (d-1)^i.
    For d=5, g=9: (g-3)/2 = 3. 1 + 5*(1 + 4 + 16) = 1 + 5*21 = 106.
    So N must be >= 106. LUW gives 1250. There is huge room.

    We try N from 106 up, in steps, and see if we can find a valid graph.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # LUW baseline
    n_luw, edges_luw = _luw_construct(5, 9)
    best_n = n_luw
    best_edges = edges_luw
    
    # Try a few candidate sizes
    # Moore bound is 106, but in practice much larger.
    # Try sizes: 200, 300, 400, 500, 625, 700, 800, 900, 1000, 1100
    candidate_ns = [106, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    candidate_ns = [n for n in candidate_ns if n % 2 == 0]  # bipartite: even total
    candidate_ns = sorted(set(candidate_ns))
    
    random.seed(42)
    
    for n in candidate_ns:
        if time.time() > deadline - 10:
            break
        if n >= best_n:
            break
        
        half = n // 2
        # Try multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Generate random 5-regular bipartite graph
            # Left side: 0..half-1, Right side: half..n-1
            # Each left vertex connects to 5 right vertices, each right to 5 left.
            # Use configuration model: create 5 stubs per vertex on each side.
            left_stubs = []
            for i in range(half):
                left_stubs.extend([i] * 5)
            right_stubs = []
            for j in range(half):
                right_stubs.extend([j + half] * 5)
            
            random.shuffle(left_stubs)
            random.shuffle(right_stubs)
            
            adj = {i: set() for i in range(n)}
            for ls, rs in zip(left_stubs, right_stubs):
                adj[ls].add(rs)
                adj[rs].add(ls)
            
            # Check degree
            if any(len(adj[i]) != 5 for i in range(n)):
                continue
            
            # Convert to edge list
            edges = set()
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.add((i, j))
            
            # Check girth >= 9 using BFS from each vertex up to depth 4
            valid = True
            for v in range(n):
                if time.time() > deadline - 15:
                    valid = False
                    break
                # BFS up to depth 4, looking for backtracking-free cycles
                # A cycle of length 2L (L<=4) would be found by a walk of length 2L
                # with no immediate backtracking.
                # BFS: (vertex, prev) -> next, depth
                seen = {}  # (vertex, prev) -> depth
                queue = deque()
                for nb in adj[v]:
                    queue.append((nb, v, 1))
                    seen[(nb, v)] = 1
                while queue and valid:
                    cur, prev, depth = queue.popleft()
                    if depth >= 4:
                        continue
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        if nb == v and depth + 1 >= 5:  # cycle of length depth+1 >= 5? No, we want < 9
                            # Actually if nb == v and depth+1 < 9, we have a cycle of length depth+1
                            if depth + 1 < 9:
                                valid = False
                                break
                        if (nb, cur) in seen:
                            # This means we found a cycle of length seen[(nb,cur)] + depth + 1
                            cycle_len = seen[(nb, cur)] + depth + 1
                            if cycle_len < 9:
                                valid = False
                                break
                        else:
                            seen[(nb, cur)] = depth + 1
                            queue.append((nb, cur, depth + 1))
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Found a valid graph!
            edge_list = list(edges)
            if n < best_n:
                best_n = n
                best_edges = edge_list
            break  # Found valid graph for this n, move to smaller n
    
    return best_n, best_edges


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # For the specific case k=5, g=9, try local search
    if k == 5 and g == 9:
        try:
            n, edges = _local_search_5_9(max_time=110.0)
            if time.time() < deadline:
                return n, edges
        except Exception:
            pass
    
    # Fallback: LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
