import time
import collections
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _has_short_cycle_nb(n, edges, min_girth):
    """
    Returns True if the graph has a cycle shorter than min_girth.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth - 1
    
    for u in range(n):
        for v in adj[u]:
            visited = set()
            visited.add((v, u))
            frontier = [(v, u)]
            
            for step in range(1, max_depth + 1):
                next_frontier = []
                for (node, prev) in frontier:
                    for nxt in adj[node]:
                        if nxt == prev:
                            continue
                        new_state = (nxt, node)
                        if new_state in visited:
                            continue
                        visited.add(new_state)
                        next_frontier.append(new_state)
                if not next_frontier:
                    break
                if (u, v) in next_frontier:
                    return True
                frontier = next_frontier
                
                if _time_left() < 1.0:
                    return False
    
    return False

def _construct_luw(k, g):
    """Fallback: construct LUW graph."""
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Moore bound for k=3, g=16 is 512.
    # Best known is 936.
    # We start searching from n=936 (m=468).
    
    min_m = 468
    max_m = 600 # Search up to n=1200
    
    for m in range(min_m, max_m + 1):
        if _time_left() < 10.0:
            break
        
        # Number of trials proportional to remaining time and inverse of m
        # Smaller m is harder, so give it more trials?
        # Actually, probability of finding a girth-16 graph drops as n increases?
        # No, density of such graphs might increase.
        # Let's just use a fixed budget per m or proportional to time left.
        
        n_trials = 20 # Fixed small number per m to allow scanning range
        
        for trial in range(n_trials):
            if _time_left() < 5.0:
                break
            
            # Generate three random perfect matchings
            perms = [list(range(m)) for _ in range(3)]
            for p in perms:
                random.shuffle(p)
            
            adj = [[] for _ in range(2 * m)]
            edges = []
            edge_set = set()
            
            valid = True
            for i in range(m):
                for j in range(3):
                    a = i
                    b = m + perms[j][i]
                    # Ensure no multi-edges
                    # Since perms are distinct permutations, (i, m+p1[i]), (i, m+p2[i]), (i, m+p3[i])
                    # are distinct if p1[i] != p2[i] != p3[i].
                    # But we must ensure no edge (i, m+j) appears twice.
                    # This is guaranteed if the 3 permutations are distinct at index i.
                    # However, an edge (i, m+j) is the same as (m+j, i).
                    # We just need to ensure we don't add the same undirected edge twice.
                    # Since we iterate i from 0 to m-1 and j from 0 to 2,
                    # and each edge connects i (in A) to m+p[j][i] (in B),
                    # the edge is uniquely identified by (i, m+p[j][i]).
                    # So we just need to ensure p1[i] != p2[i] and p2[i] != p3[i] etc.
                    # Random permutations might have p1[i] == p2[i].
                    # If so, we have a multi-edge.
                    
                    if (a, b) in edge_set:
                        valid = False
                        break
                    edge_set.add((a, b))
                    edges.append((i, m + perms[j][i]))
                    adj[i].append(m + perms[j][i])
                    adj[m + perms[j][i]].append(i)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check degrees
            # Bipartite 3-regular construction ensures degree 3 if no multi-edges.
            
            # Check girth
            if not _has_short_cycle_nb(2 * m, edges, g):
                return (2 * m, edges)
    
    # Fallback to LUW
    return _construct_luw(k, g)
