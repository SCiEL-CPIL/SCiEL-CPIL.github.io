import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115  # Leave 5s margin
    
    # 1. Get baseline from LUW
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
        if _verify_graph(edges_luw, n_luw, k, g):
            best_n = n_luw
            best_edges = edges_luw
        else:
            best_n = float('inf')
            best_edges = None
    else:
        best_n = float('inf')
        best_edges = None

    # 2. Try to find a smaller graph using local search on bipartite graphs
    if k % 2 != 0:
        if best_edges is not None:
            return best_n, best_edges
        return _luw_construct(k, g)
        
    # Moore bound
    moore_bound = _moore_bound(k, g)
    
    # Search for n values
    candidates = []
    start_n = max(moore_bound, 320)
    if start_n < best_n:
        for n in range(start_n, min(best_n, start_n + 100), 2):
            candidates.append(n)
            
    for n in candidates:
        if time.time() > deadline - 5:
            break
        result = _local_search_bipartite(k, g, n, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
                if n_found == 320:
                    return best_n, best_edges

    if best_edges is not None:
        return best_n, best_edges
        
    return _luw_construct(k, g)

def _moore_bound(k, g):
    if g % 2 == 0:
        m = g // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 2 * s
    else:
        m = (g - 1) // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 1 + k * s

def _verify_graph(edges, n, k, g):
    if n <= 0 or len(edges) == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False
            
    # Girth check using BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        max_depth = (g - 2) // 2
        
        for d in range(1, max_depth + 1):
            new_queue = []
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d
                        parent[v] = u
                        new_queue.append(v)
                    else:
                        if dist[v] < d - 1:
                            cycle_len = d + dist[v] + 1
                            if cycle_len < g:
                                return False
                        elif dist[v] == d - 1 and parent[u] != v:
                            cycle_len = 2 * d
                            if cycle_len < g:
                                return False
            queue = new_queue
    return True

def _luw_construct(k, g):
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

def _local_search_bipartite(k, g, n, deadline):
    M = n // 2
    if M * 2 != n:
        return None
    
    # Use a structured initialization based on modular arithmetic
    # For 4-regular bipartite graph with girth 10, we can use a circulant-like structure
    
    left_adj = [set() for _ in range(M)]
    right_adj = [set() for _ in range(M)]
    
    def init_structured():
        nonlocal left_adj, right_adj
        left_adj = [set() for _ in range(M)]
        right_adj = [set() for _ in range(M)]
        
        # Try to use a difference set approach
        # For each left node i, connect to right nodes based on offsets
        # Use offsets that avoid creating short cycles
        # For girth 10 in bipartite, we need to avoid 4-cycles and 6-cycles
        
        # Simple approach: use multiple offset sets
        # offset_sets[0] = {0, 1, 2, 3}  # too small, creates 4-cycles
        # Need offsets spread out
        
        # Use a greedy approach: assign edges to avoid short cycles
        # For each left node, pick 4 right nodes that don't create short cycles with existing edges
        
        for l in range(M):
            candidates = list(range(M))
            random.shuffle(candidates)
            chosen = []
            for r in candidates:
                if len(chosen) == k:
                    break
                # Check if adding (l, r) creates a short cycle
                if not _creates_short_cycle(l, r, left_adj, right_adj, k, g):
                    chosen.append(r)
            if len(chosen) < k:
                # Fallback: fill with random
                for r in candidates:
                    if r not in chosen:
                        chosen.append(r)
                    if len(chosen) == k:
                        break
            for r in chosen:
                left_adj[l].add(r)
                right_adj[r].add(l)
        return True
    
    def _creates_short_cycle(l, r, left_adj, right_adj, k, g):
        # Check if adding edge (l, r) creates a cycle shorter than g
        # A 4-cycle: l1-r1-l2-r2-l1 where l1=l, r1=r
        # So check if there exists l2 such that l2 is connected to both r and some r2 connected to l
        # More precisely: 4-cycle exists if there's l2 != l with r in right_adj[l2] and some r2 in left_adj[l] with l2 in right_adj[r2]
        
        # For efficiency, just check for 4-cycles (most common issue)
        # 4-cycle: l - r - l2 - r2 - l where r2 != r
        # This exists if there's l2 connected to r, and l2 connected to some r2 that l is also connected to
        
        # Actually simpler: 4-cycle through (l,r) exists if there's another path of length 3 from l to r
        # i.e., l - r2 - l2 - r where r2 != r, l2 != l
        
        # Check all r2 in left_adj[l] (excluding r if present, but r isn't in left_adj[l] yet)
        for r2 in left_adj[l]:
            if r2 == r:
                continue
            # Check if there's l2 connected to both r2 and r
            common = right_adj[r2] & right_adj[r] if r in right_adj else set()
            # Wait, right_adj[r] may not have l yet since we're checking before adding
            # Actually we're checking if adding (l,r) creates a cycle
            # So current state: l is connected to r2, r is connected to some set
            # After adding (l,r), we need to check for cycles
            # 4-cycle: l-r-l2-r2-l where l2 is connected to both r and r2
            # So check if there exists l2 in (right_adj[r] ∩ right_adj[r2])
            if r in right_adj:
                common_l2 = right_adj[r] & right_adj[r2]
                if common_l2:
                    return True
        return False
    
    init_structured()
    
    # Check initial girth
    def check_girth():
        adj = [None] * n
        for i in range(M):
            adj[i] = list(left_adj[i])
        for i in range(M):
            adj[M+i] = list(right_adj[i])
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            max_depth = (g - 2) // 2
            for d in range(1, max_depth + 1):
                new_queue = []
                for u in queue:
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = d
                            parent[v] = u
                            new_queue.append(v)
                        else:
                            if dist[v] < d - 1:
                                cycle_len = d + dist[v] + 1
                                if cycle_len < g:
                                    return False, cycle_len
                            elif dist[v] == d - 1 and parent[u] != v:
                                cycle_len = 2 * d
                                if cycle_len < g:
                                    return False, cycle_len
                queue = new_queue
        return True, g
    
    is_valid, girth = check_girth()
    if is_valid:
        edges = []
        for l in range(M):
            for r in left_adj[l]:
                edges.append((l, M+r))
        return n, edges
    
    # Local search with more targeted moves
    for iteration in range(50000):
        if time.time() > deadline:
            return None
            
        is_valid, girth = check_girth()
        if is_valid:
            edges = []
            for l in range(M):
                for r in left_adj[l]:
                    edges.append((l, M+r))
            return n, edges
            
        # Try to find and fix a short cycle
        # Pick a random left node and try to rewire one of its edges
        l = random.randint(0, M-1)
        if not left_adj[l]:
            continue
        r_old = random.choice(list(left_adj[l]))
        
        # Find a new right node r_new that doesn't create short cycles
        candidates = [r for r in range(M) if r != r_old and r not in left_adj[l]]
        random.shuffle(candidates)
        
        found = False
        for r_new in candidates:
            # Check if swapping (l, r_old) with (l, r_new) reduces short cycles
            # Temporarily make the change
            left_adj[l].remove(r_old)
            right_adj[r_old].remove(l)
            left_adj[l].add(r_new)
            right_adj[r_new].add(l)
            
            # Quick check: does this create a 4-cycle?
            has_4cycle = False
            for r2 in left_adj[l]:
                if r2 != r_new:
                    common = right_adj[r2] & right_adj[r_new]
                    if common:
                        has_4cycle = True
                        break
            
            if not has_4cycle:
                found = True
                break
            else:
                # Revert
                left_adj[l].remove(r_new)
                right_adj[r_new].remove(l)
                left_adj[l].add(r_old)
                right_adj[r_old].add(l)
        
        if not found:
            # Try a 2-opt swap
            l2 = random.randint(0, M-1)
            if l2 == l or not left_adj[l2]:
                continue
            r2_old = random.choice(list(left_adj[l2]))
            
            # Try swapping: (l, r_old) <-> (l2, r2_old)
            if r_old == r2_old or l == l2:
                continue
            if r2_old in left_adj[l] or r_old in left_adj[l2]:
                continue
                
            # Make the swap
            left_adj[l].remove(r_old)
            right_adj[r_old].remove(l)
            left_adj[l2].remove(r2_old)
            right_adj[r2_old].remove(l2)
            
            left_adj[l].add(r2_old)
            right_adj[r2_old].add(l)
            left_adj[l2].add(r_old)
            right_adj[r_old].add(l2)
            
            # Check for 4-cycles
            has_4cycle = False
            for r2 in left_adj[l]:
                if r2 != r2_old:
                    common = right_adj[r2] & right_adj[r2_old]
                    if common:
                        has_4cycle = True
                        break
            if not has_4cycle:
                for r2 in left_adj[l2]:
                    if r2 != r_old:
                        common = right_adj[r2] & right_adj[r_old]
                        if common:
                            has_4cycle = True
                            break
            if has_4cycle:
                # Revert
                left_adj[l].remove(r2_old)
                right_adj[r2_old].remove(l)
                left_adj[l2].remove(r_old)
                right_adj[r_old].remove(l2)
                
                left_adj[l].add(r_old)
                right_adj[r_old].add(l)
                left_adj[l2].add(r2_old)
                right_adj[r2_old].add(l2)
    
    return None
