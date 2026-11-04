import random
import time
from collections import deque

def construct(k, g):
    if k != 3 or g != 16:
        return _construct_luw(k, g)
    
    start_time = time.time()
    time_limit = 110
    
    # Moore bound for (3, 16) is 512.
    # We are looking for covers of K4, so n_vertices = 4 * n_group.
    # n_group must be at least 512 / 4 = 128.
    # Known best is 936 vertices -> n_group = 234.
    # We search n_group from 128 to 300 (to be safe).
    
    for n_group in range(128, 301):
        if time.time() - start_time > time_limit:
            break
        
        # Try random voltage assignments for K4 cover
        # Base graph K4: 0, 1, 2, 3
        # Fix voltages of tree edges (0,1), (0,2), (0,3) to 0.
        # Variables: v12, v13, v23
        # Edges:
        # (0,1): 0
        # (0,2): 0
        # (0,3): 0
        # (1,2): v12
        # (1,3): v13
        # (2,3): v23
        
        # We need to ensure no short cycles.
        # Check girth via BFS on (vertex, voltage) space.
        
        max_trials = 50
        for _ in range(max_trials):
            if time.time() - start_time > time_limit:
                break
            
            v12 = random.randint(1, n_group - 1)
            v13 = random.randint(1, n_group - 1)
            v23 = random.randint(1, n_group - 1)
            
            # Quick checks for triangles (length 3)
            # Triangles in K4:
            # 1-0-2-1: volt 0+0-v12 = -v12. Need -v12 != 0. (True since v12 in 1..n-1)
            # 1-0-3-1: volt -v13. (True)
            # 2-0-3-2: volt -v23. (True)
            # 1-2-3-1: v12 + v23 - v13. Need != 0.
            # 1-3-2-1: v13 - v23 - v12. Need != 0.
            # 2-1-3-2: -v12 + v13 - v23. Need != 0.
            
            if (v12 + v23 - v13) % n_group == 0: continue
            if (v13 - v23 - v12) % n_group == 0: continue
            if (-v12 + v13 - v23) % n_group == 0: continue
            
            # Check 4-cycles
            # 1-0-2-3-1: 0+0+v23-v13 = v23-v13. Need != 0.
            # 1-0-3-2-1: 0+0-v23-v12 = -(v23+v12). Need != 0.
            # 1-2-0-3-1: v12+0+0-v13 = v12-v13. Need != 0.
            # 1-3-0-2-1: v13+0+0-v12 = v13-v12. Need != 0.
            # 2-0-1-3-2: 0+0+v13-v23 = v13-v23. Need != 0.
            # 2-3-0-1-2: v23+0+0-v12 = v23-v12. Need != 0.
            
            if (v23 - v13) % n_group == 0: continue
            if (v23 + v12) % n_group == 0: continue
            if (v12 - v13) % n_group == 0: continue
            if (v13 - v12) % n_group == 0: continue
            if (v13 - v23) % n_group == 0: continue
            if (v23 - v12) % n_group == 0: continue
            
            # Full BFS check for girth
            if _check_girth_k4_cover(n_group, v12, v13, v23, g):
                # Construct edges
                n_vertices = 4 * n_group
                edges = []
                # Map (vertex, offset) to index
                # idx = vertex * n_group + offset
                
                # Edge (0,1) volt 0
                for x in range(n_group):
                    u = 0 * n_group + x
                    v = 1 * n_group + x
                    edges.append((min(u,v), max(u,v)))
                
                # Edge (0,2) volt 0
                for x in range(n_group):
                    u = 0 * n_group + x
                    v = 2 * n_group + x
                    edges.append((min(u,v), max(u,v)))
                
                # Edge (0,3) volt 0
                for x in range(n_group):
                    u = 0 * n_group + x
                    v = 3 * n_group + x
                    edges.append((min(u,v), max(u,v)))
                
                # Edge (1,2) volt v12
                for x in range(n_group):
                    u = 1 * n_group + x
                    v = 2 * n_group + (x + v12) % n_group
                    edges.append((min(u,v), max(u,v)))
                
                # Edge (1,3) volt v13
                for x in range(n_group):
                    u = 1 * n_group + x
                    v = 3 * n_group + (x + v13) % n_group
                    edges.append((min(u,v), max(u,v)))
                
                # Edge (2,3) volt v23
                for x in range(n_group):
                    u = 2 * n_group + x
                    v = 3 * n_group + (x + v23) % n_group
                    edges.append((min(u,v), max(u,v)))
                
                return n_vertices, edges

    return _construct_luw(k, g)

def _check_girth_k4_cover(n_group, v12, v13, v23, g):
    # BFS on (vertex, voltage)
    # Vertices 0, 1, 2, 3
    # Neighbors in base graph with voltages:
    # 0: (1, 0), (2, 0), (3, 0)
    # 1: (0, 0), (2, v12), (3, v13)
    # 2: (0, 0), (1, -v12), (3, v23)
    # 3: (0, 0), (1, -v13), (2, -v23)
    
    # We want to find the shortest closed walk starting at 0 with voltage 0.
    # Actually, any closed walk at any vertex with voltage 0.
    # Since the graph is vertex-transitive in the cover? No.
    # But we can just BFS from (0,0) and check if we return to (0,0) or any (v,0)?
    # No, a cycle in the cover is a closed walk in the base graph with net voltage 0.
    # It must start and end at the same base vertex.
    
    # We run BFS from (0,0).
    # dist[(v, volt)] = length
    # If we reach (0, 0) with length L > 0, we have a cycle of length L.
    # But we must ensure it's a *simple* cycle?
    # BFS finds shortest paths. The shortest path from (0,0) to (0,0) is a cycle.
    # Is it simple? Yes, in a BFS tree, the first time we reach a state, it's via a shortest path.
    # If the shortest path to (0,0) is length L, it corresponds to a cycle of length L.
    # However, we must be careful: the "path" in the state space might backtrack in the base graph?
    # No, BFS on the state space (v, volt) is equivalent to BFS on the cover graph.
    # The cover graph is simple.
    # So the shortest path from (0,0) to (0,0) in the cover graph is the girth (if we start at 0).
    # But the girth is the minimum over all vertices.
    # Due to symmetry of the construction (translations in voltage), the girth is the same for all vertices in the same base vertex class.
    # So checking from (0,0) is sufficient?
    # Yes, if there is a cycle of length < g anywhere, there is one at 0?
    # The graph is not necessarily vertex transitive, but the "fiber" over 0 is isomorphic to the fiber over 1, etc.
    # So checking one representative is enough.
    
    dist = {}
    dist[(0, 0)] = 0
    queue = deque([(0, 0)])
    
    while queue:
        v, volt = queue.popleft()
        d = dist[(v, volt)]
        if d >= g - 1: # If we are at depth g-1, neighbors are at depth g.
            # If a neighbor is (0,0), cycle length is g. This is allowed.
            # We only fail if cycle length < g.
            # So we stop expanding if d == g-1?
            # If we find (0,0) at d+1 < g, return False.
            pass
            
        # Get neighbors
        neighbors = []
        if v == 0:
            neighbors = [(1, 0), (2, 0), (3, 0)]
        elif v == 1:
            neighbors = [(0, 0), (2, v12), (3, v13)]
        elif v == 2:
            neighbors = [(0, 0), (1, -v12), (3, v23)]
        elif v == 3:
            neighbors = [(0, 0), (1, -v13), (2, -v23)]
            
        for nv, nvolt in neighbors:
            nvolt %= n_group
            nstate = (nv, nvolt)
            if nstate not in dist:
                dist[nstate] = d + 1
                if nv == 0 and nvolt == 0:
                    # Found a cycle of length d+1
                    if d + 1 < g:
                        return False
                    else:
                        # We found a valid cycle of length >= g.
                        # But we need to ensure NO cycle is shorter.
                        # BFS guarantees shortest path.
                        # So if the shortest path to (0,0) is >= g, we are good?
                        # Yes.
                        return True
                queue.append(nstate)
            else:
                # If we already visited this state, the path is longer or equal.
                # We don't care about non-shortest paths for girth.
                pass
                
    # If queue empties without finding (0,0) again?
    # Impossible, (0,0) is in dist.
    # But we only return True when we *first* hit (0,0) again.
    # What if (0,0) is the start?
    # We start at (0,0). We don't check neighbors for (0,0) immediately.
    # We expand.
    # If we never return to (0,0) with a valid path?
    # That means the graph is a tree? Impossible for finite regular graph.
    # It must return.
    
    return True

def _construct_luw(k, g):
    # Original LUW construction
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
