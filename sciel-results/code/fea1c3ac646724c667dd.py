import time
import random
import heapq
from collections import deque
import sys

def _gq44_construct():
    """Construct the GQ(4,4) point-line incidence graph (9-regular bipartite, 810 vertices).
    Then select a 5-regular spanning subgraph that preserves girth >= 9.
    
    GQ(4,4): Points are (x,y,z) in GF(4)^3 with z = xy + x^2 + y^2 (or infinity).
    Lines are parameterized similarly. The incidence graph is 9-regular.
    """
    # GF(4) = GF(2)[alpha] / (alpha^2 + alpha + 1)
    # Elements: 0, 1, alpha, alpha+1 (where alpha^2 = alpha + 1)
    # We represent as integers 0,1,2,3 where:
    # 0 -> 0, 1 -> 1, 2 -> alpha, 3 -> alpha+1
    
    # Addition table (XOR)
    gf_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    
    # Multiplication table
    # alpha^2 = alpha+1 = 3, alpha^3 = 1
    gf_mul = [
        [0,0,0,0],
        [0,1,2,3],
        [0,2,3,1],
        [0,3,1,2]
    ]
    
    # Inverse: 0->0, 1->1, 2->3, 3->2
    gf_inv = [0, 1, 3, 2]
    
    def gf_eq(a, b):
        return a == b
    
    # Points of GQ(4,4): 
    # Standard form: points are (x, y, z) where z = x^2 + y^2 + xy for x,y in GF(4),
    # plus points at infinity.
    # Actually, let me use the standard construction.
    # GQ(s,t) has (s+1)(st+s+1) points. For s=t=4: 5*(16+4+1)=5*21=105 points.
    # Lines: same number, 105.
    # Incidence: each point on (t+1)=5 lines, each line has (s+1)=5 points.
    # Wait, that gives 5-regular, not 9-regular.
    
    # Let me reconsider. The point-line incidence graph of GQ(4,4) is 5-regular on 210 vertices.
    # That's actually what we want! Let me verify the girth.
    # GQ(4,4) has no 4-cycles (it's a generalized quadrangle), so the incidence graph
    # has girth at least 8. We need girth >= 9, so we need to check for 8-cycles.
    # Actually for GQ(s,t), the incidence graph has girth 8 (there are 8-cycles).
    # So this won't directly give us girth 9.
    
    # Alternative: Use the known 5-regular graph of girth 9 from the literature.
    # The smallest known is 1116 vertices. Let me try a different algebraic approach.
    
    # Actually, let me try a different strategy: use the LUW construction which is known
    # to produce graphs with good girth, and then try to minimize.
    
    # For k=5, g=9, the LUW construction should work. Let me check what n it produces.
    return None  # This approach doesn't directly work; fall back to other methods

def _luw_construct(k, g):
    # Original LUW construction (kept as fallback)
    # Copy of the seed logic
    # Copy of the seed logic
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

def _compute_girth_fast(adjacency, n, early_stop=None):
    """Compute girth using non-backtracking walks (BFS on directed edge states).
    Optimized with early termination and array-based storage."""
    if early_stop is None:
        early_stop = 100
    
    min_girth = early_stop + 1
    
    for start in range(n):
        if min_girth <= early_stop:
            break
        
        # dist_map: (prev, curr) -> distance
        dist_map = {}
        queue = deque()
        
        for v in adjacency[start]:
            dist_map[(start, v)] = 1
            queue.append((start, v, 1))
        
        while queue:
            prev, curr, d = queue.popleft()
            
            if d + 1 >= min_girth or d + 1 > early_stop:
                continue
            
            for nxt in adjacency[curr]:
                if nxt == prev:
                    continue
                new_d = d + 1
                if nxt == start:
                    if new_d < min_girth:
                        min_girth = new_d
                        if min_girth <= early_stop:
                            return min_girth
                else:
                    key = (curr, nxt)
                    if key not in dist_map or dist_map[key] > new_d:
                        dist_map[key] = new_d
                        queue.append((curr, nxt, new_d))
    
    return min_girth if min_girth != early_stop + 1 else -1

def _find_short_cycle_fast(adjacency, n, max_girth):
    """Find a cycle of length <= max_girth using non-backtracking BFS.
    Returns (u, v, cycle_len) where (u,v) is an edge on the cycle, or None."""
    for start in range(n):
        dist_map = {}
        queue = deque()
        
        for v in adjacency[start]:
            dist_map[(start, v)] = (1, None)
            queue.append((start, v, 1))
        
        while queue:
            prev, curr, d = queue.popleft()
            if d >= max_girth:
                continue
            
            for nxt in adjacency[curr]:
                if nxt == prev:
                    continue
                new_d = d + 1
                if nxt == start:
                    return (curr, start, new_d)
                else:
                    key = (curr, nxt)
                    if key not in dist_map or dist_map[key][0] > new_d:
                        dist_map[key] = (new_d, prev)
                        queue.append((curr, nxt, new_d))
    return None

def _has_short_path_bounded(adjacency, src, dst, max_dist, excluded_edges=None):
    """Check if there's a path from src to dst of length <= max_dist avoiding excluded edges."""
    if src == dst:
        return True
    if excluded_edges is None:
        excluded_edges = set()
    visited = {src}
    queue = deque([(src, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= max_dist:
            continue
        for neighbor in adjacency[node]:
            if (node, neighbor) in excluded_edges or (neighbor, node) in excluded_edges:
                continue
            if neighbor == dst:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
    return False

def _is_2_switch_valid_fast(adjacency, u, v, w, x, g):
    """Fast check: does 2-switch (u,v)&(w,x) -> (u,w)&(v,x) create a cycle < g?"""
    excluded = {(u, v), (v, u), (w, x), (x, w)}
    
    if _has_short_path_bounded(adjacency, u, w, g - 2, excluded):
        return False
    
    if _has_short_path_bounded(adjacency, v, x, g - 2, excluded):
        return False
    
    return True

def _generate_simple_k_regular(k, n, max_attempts=100):
    """Generate a simple k-regular graph on n vertices using configuration model with rejection."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(max_attempts):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if valid:
            return [list(s) for s in adjacency]
    
    return None

def _switching_construct(k, g, n, time_limit=100.0):
    """Construct a k-regular graph of girth >= g using edge switching with adaptive strategy."""
    start_time = time.time()
    
    adjacency = _generate_simple_k_regular(k, n, max_attempts=100)
    if adjacency is None:
        return None
    
    edge_set = set()
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edge_set.add((i, j))
    
    current_girth = _compute_girth_fast(adjacency, n, early_stop=g)
    
    if current_girth >= g:
        edges = list(edge_set)
        return n, edges
    
    # Track number of short cycles for adaptive strategy
    max_iterations = 50000
    no_improve_count = 0
    last_girth = current_girth
    
    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Find a short cycle
        result = _find_short_cycle_fast(adjacency, n, g - 1)
        if result is None:
            final_girth = _compute_girth_fast(adjacency, n, early_stop=g)
            if final_girth >= g:
                edges = []
                for i in range(n):
                    for j in adjacency[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            continue
        
        u, v, cycle_len = result
        
        # Try to break this cycle by switching edge (u,v) with another edge
        success = False
        # Increase attempts as we get closer to time limit
        num_attempts = 20 if elapsed < time_limit * 0.5 else 30
        for attempt in range(num_attempts):
            w = random.randint(0, n - 1)
            if not adjacency[w]:
                continue
            x = random.choice(adjacency[w])
            if x <= w:
                continue
            
            if len({u, v, w, x}) < 4:
                continue
            if (u, w) in edge_set or (w, u) in edge_set:
                continue
            if (v, x) in edge_set or (x, v) in edge_set:
                continue
            if (u, x) in edge_set or (x, u) in edge_set:
                continue
            if (v, w) in edge_set or (w, v) in edge_set:
                continue
            
            if _is_2_switch_valid_fast(adjacency, u, v, w, x, g):
                adjacency[u].remove(v)
                adjacency[v].remove(u)
                adjacency[w].remove(x)
                adjacency[x].remove(w)
                adjacency[u].append(w)
                adjacency[w].append(u)
                adjacency[v].append(x)
                adjacency[x].append(v)
                
                edge_set.discard((min(u,v), max(u,v)))
                edge_set.discard((min(w,x), max(w,x)))
                edge_set.add((min(u,w), max(u,w)))
                edge_set.add((min(v,x), max(v,x)))
                
                success = True
                break
        
        if success:
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, try random valid switches to escape
            if no_improve_count > 20:
                for _ in range(10):
                    w = random.randint(0, n - 1)
                    if not adjacency[w]:
                        continue
                    x = random.choice(adjacency[w])
                    if x <= w:
                        continue
                    u_r = random.randint(0, n - 1)
                    if not adjacency[u_r]:
                        continue
                    v_r = random.choice(adjacency[u_r])
                    if v_r <= u_r:
                        continue
                    if len({u_r, v_r, w, x}) < 4:
                        continue
                    if (u_r, w) in edge_set or (v_r, x) in edge_set:
                        continue
                    if (u_r, x) in edge_set or (v_r, w) in edge_set:
                        continue
                    if _is_2_switch_valid_fast(adjacency, u_r, v_r, w, x, g):
                        adjacency[u_r].remove(v_r)
                        adjacency[v_r].remove(u_r)
                        adjacency[w].remove(x)
                        adjacency[x].remove(w)
                        adjacency[u_r].append(w)
                        adjacency[w].append(u_r)
                        adjacency[v_r].append(x)
                        adjacency[x].append(v_r)
                        edge_set.discard((min(u_r,v_r), max(u_r,v_r)))
                        edge_set.discard((min(w,x), max(w,x)))
                        edge_set.add((min(u_r,w), max(u_r,w)))
                        edge_set.add((min(v_r,x), max(v_r,x)))
                        break
                no_improve_count = 0
    
    final_girth = _compute_girth_fast(adjacency, n, early_stop=g)
    if final_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def _greedy_construct(k, g, n, time_limit=10.0):
    """Greedy construction: add edges one at a time, checking girth constraint."""
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    
    def has_short_cycle(u, v, max_depth):
        """Check if adding edge (u,v) creates a cycle of length < max_depth+1."""
        if u == v:
            return True
        # BFS from u to find path to v of length <= max_depth-1
        # (because the new edge adds 1 to the path length)
        if max_depth <= 1:
            return False
        visited = {u}
        queue = deque([(u, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= max_depth - 1:
                continue
            for neighbor in adjacency[node]:
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
        return False

    # Process vertices in order, trying to give each vertex degree k
    for u in range(n):
        if time.time() - start_time > time_limit:
            return None
        
        # Find vertices that need more degree
        candidates = [v for v in range(n) if v != u and v not in adjacency[u] and degrees[v] < k]
        random.shuffle(candidates)
        
        while degrees[u] < k:
            found = False
            for v in candidates:
                if degrees[v] >= k:
                    continue
                if has_short_cycle(u, v, g):
                    continue
                
                adjacency[u].add(v)
                adjacency[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                candidates.remove(v)
                found = True
                break
            
            if not found:
                return None
    
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
    
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
    return n, edges

def construct(k, g):
    # For the specific case k=5, g=9, try switching algorithm
    if k == 5 and g == 9:
        total_time = 145.0
        start_time = time.time()
        
        # Try switching algorithm with various n values, ordered by desirability
        # Smaller n gives better score, so try smaller first but with less time
        # since they're harder. Use larger n as guaranteed fallback.
        target_ns = [1000, 1100, 1116, 1200, 1400, 1600, 1800, 2000]
        
        best_result = None
        best_n = float('inf')
        
        for n in target_ns:
            elapsed = time.time() - start_time
            remaining = total_time - elapsed
            if remaining < 8:
                break
            
            # Allocate time based on n: smaller n gets more time (harder but better score)
            if n <= 1100:
                time_per_attempt = min(35.0, remaining * 0.35)
            elif n <= 1200:
                time_per_attempt = min(30.0, remaining * 0.30)
            elif n <= 1400:
                time_per_attempt = min(25.0, remaining * 0.25)
            elif n <= 1600:
                time_per_attempt = min(20.0, remaining * 0.20)
            else:
                time_per_attempt = min(15.0, remaining * 0.15)
            
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                best_result = res
                best_n = n
                # If we found a small one, return immediately (best possible score)
                if n <= 1200:
                    return res
                # Otherwise continue trying smaller
        
        # If switching failed for all, try greedy on a moderate n
        elapsed = time.time() - start_time
        remaining = total_time - elapsed
        if remaining > 12:
            for n in [1200, 1400, 1600]:
                if time.time() - start_time > total_time - 6:
                    break
                res = _greedy_construct(k, g, n, time_limit=min(10.0, remaining * 0.4))
                if res is not None:
                    return res
        
        if best_result is not None:
            return best_result
    
    # Fallback to LUW
    return _luw_construct(k, g)
