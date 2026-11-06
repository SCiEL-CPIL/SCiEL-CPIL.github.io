import itertools
import random
import time

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precompute squared distances between all pairs
dist = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        dist[i][j] = d
        dist[j][i] = d

# Build conflict graph: two points are adjacent if they participate in some isosceles triangle
# i.e., there exists a third point k such that the triangle (i, j, k) is isosceles
adj = [0] * NUM_POINTS  # bitset of neighbors
for i in range(NUM_POINTS):
    for j in range(i+1, NUM_POINTS):
        # Check if there exists k != i, j such that (i,j,k) is isosceles
        di = dist[i][j]
        # For each k, check if triangle is isosceles
        # isosceles if any two sides equal: di == dist[i][k] or di == dist[j][k] or dist[i][k] == dist[j][k]
        # We need to check if such k exists
        # Efficient: for each k, check
        found = False
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            dik = dist[i][k]
            djk = dist[j][k]
            if di == dik or di == djk or dik == djk:
                found = True
                break
        if found:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

# Precompute degree for ordering
degree = [bin(adj[i]).count('1') for i in range(NUM_POINTS)]

# Count total edges
total_edges = sum(degree) // 2

def popcount(x):
    return bin(x).count('1')

def min_vertex_cover(deadline):
    """Branch and bound for minimum vertex cover."""
    best = [total_edges + 1]  # upper bound
    best_set = [set()]
    
    # Start with a greedy upper bound
    def greedy_vc():
        remaining = set(range(NUM_POINTS))
        cover = set()
        while remaining:
            # Pick node with max degree in remaining
            best_node = -1
            best_deg = -1
            for v in remaining:
                deg_in_rem = popcount(adj[v] & ((1 << NUM_POINTS) - 1) & 0)
                # count neighbors in remaining
                cnt = 0
                mask = adj[v]
                for u in remaining:
                    if (mask >> u) & 1:
                        cnt += 1
                if cnt > best_deg:
                    best_deg = cnt
                    best_node = v
            if best_node < 0:
                break
            cover.add(best_node)
            remaining.discard(best_node)
            # Remove all neighbors from remaining (they're covered)
            for u in list(remaining):
                if (adj[best_node] >> u) & 1:
                    remaining.discard(u)
        return cover
    
    gc = greedy_vc()
    best[0] = len(gc)
    best_set[0] = gc
    
    # Branch and bound
    # State: uncovered edges represented by set of nodes that still need covering
    # We'll use a recursive search with memoization on the "active" edge set
    
    # Represent state as the set of vertices that have uncovered edges
    # A vertex is "active" if it has at least one neighbor that is also active
    # Actually, let's track which vertices are still "uncovered" in the sense that
    # they have an edge to another uncovered vertex
    
    # Simpler: track the set of vertices that are endpoints of at least one uncovered edge
    # When we pick a vertex v, we cover all edges incident to v, so we remove v and all
    # its neighbors from the active set (since all their edges to v are covered, but they
    # may still have edges to other active vertices)
    
    # Let's use a different representation: the set of active vertices (those with uncovered edges)
    # Initially all vertices with degree > 0
    # When we select v: all edges incident to v are covered. So v and its neighbors lose those edges.
    # A vertex u remains active if it has at least one neighbor w (w != v) that is also still active.
    
    # This is getting complex. Let me use a cleaner formulation.
    
    # State: frozenset of vertices that are endpoints of uncovered edges
    # Transition: pick vertex v from active set. New active set = {u in active \ {v} : u has neighbor in active \ {v} \ N(v)}
    # Wait, that's not quite right either.
    
    # Let me think again. The uncovered edges are those where both endpoints are in the "uncovered" set.
    # When we pick v, we cover all edges incident to v. So the new uncovered edges are those
    # where both endpoints are in (uncovered \ {v}) AND both are not v.
    # So new uncovered set = {u in uncovered \ {v} : u has a neighbor in uncovered \ {v}}
    
    # Let's implement this with bitsets for speed.
    
    all_mask = (1 << NUM_POINTS) - 1
    
    def get_active(mask):
        """Given a mask of vertices, return the mask of vertices that have at least one neighbor in the mask."""
        # For each vertex in mask, check if it has a neighbor in mask
        active = 0
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            if (adj[v] & mask) & ~lb:  # has a neighbor in mask other than itself
                active |= lb
            m ^= lb
        return active
    
    # Initial active set: all vertices with degree > 0
    initial_active = 0
    for i in range(NUM_POINTS):
        if degree[i] > 0:
            initial_active |= (1 << i)
    initial_active = get_active(initial_active)
    
    # Branch and bound
    nodes_explored = [0]
    
    def search(active_mask, current_cover_size, current_cover):
        if time.time() > deadline:
            return
        
        nodes_explored[0] += 1
        
        # Lower bound: we need at least ceil(edges_in_active / max_degree) more vertices
        # Simpler lower bound: active vertices / 2 (each vertex covers at most... no)
        # Each vertex we pick covers all edges incident to it.
        # Lower bound: number of active vertices / 2 (rough)
        # Better: matching lower bound is expensive. Use: ceil(active_count / max_degree_in_active)
        
        active_count = popcount(active_mask)
        if active_count == 0:
            if current_cover_size < best[0]:
                best[0] = current_cover_size
                best_set[0] = set(current_cover)
            return
        
        # Lower bound
        # Find max degree in active subgraph
        max_deg = 0
        m = active_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            d = popcount(adj[v] & active_mask)
            if d > max_deg:
                max_deg = d
            m ^= lb
        
        if max_deg == 0:
            if current_cover_size < best[0]:
                best[0] = current_cover_size
                best_set[0] = set(current_cover)
            return
        
        lb = (active_count + max_deg - 1) // max_deg
        if current_cover_size + lb >= best[0]:
            return
        
        # Choose vertex with max degree in active subgraph for branching
        best_v = -1
        best_d = -1
        m = active_mask
        while m:
            lb2 = m & (-m)
            v = lb2.bit_length() - 1
            d = popcount(adj[v] & active_mask)
            if d > best_d:
                best_d = d
                best_v = v
            m ^= lb2
        
        if best_v < 0:
            if current_cover_size < best[0]:
                best[0] = current_cover_size
                best_set[0] = set(current_cover)
            return
        
        # Branch 1: include best_v
        new_active = get_active(active_mask & ~(1 << best_v))
        search(new_active, current_cover_size + 1, current_cover + [best_v])
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude best_v, so we must include all its neighbors
        # This is a valid branch: if we don't pick v, we must pick all neighbors of v
        # to cover the edges (v, u) for each neighbor u.
        # But this might be expensive. Let's only do this if it's promising.
        
        # Actually, the standard B&B for vertex cover branches on: include v OR include all neighbors of v.
        # The second branch forces all neighbors into the cover.
        
        neighbors = adj[best_v] & active_mask
        forced_size = popcount(neighbors)
        if current_cover_size + forced_size < best[0]:
            # New active: remove v and all its neighbors
            new_active = get_active(active_mask & ~(1 << best_v) & ~neighbors)
            new_cover = list(current_cover)
            m2 = neighbors
            while m2:
                lb3 = m2 & (-m2)
                u = lb3.bit_length() - 1
                new_cover.append(u)
                m2 ^= lb3
            search(new_active, current_cover_size + forced_size, new_cover)
    
    search(initial_active, 0, [])
    
    return best_set[0]

def verify_maximal_no_isosceles(S):
    """Verify that S is a maximal no-isosceles set."""
    S_set = set(S)
    
    # Check no isosceles triangle in S
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                a, b, c = S[i], S[j], S[k]
                da = dist[a][b]
                db = dist[b][c]
                dc = dist[c][a]
                if da == db or da == dc or db == dc:
                    return False
    
    # Check maximality: no point can be added
    for p in range(NUM_POINTS):
        if p in S_set:
            continue
        # Check if adding p creates an isosceles triangle
        can_add = True
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a, b = S[i], S[j]
                da = dist[p][a]
                db = dist[p][b]
                dc = dist[a][b]
                if da == db or da == dc or db == dc:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Try minimum vertex cover via branch and bound
    try:
        best_set = min_vertex_cover(deadline)
    except Exception:
        best_set = set()
    
    # Verify the result
    if best_set and verify_maximal_no_isosceles(best_set):
        return [POINTS[i] for i in best_set]
    
    # Fallback: greedy construction
    chosen = []
    chosen_set = set()
    for p in range(NUM_POINTS):
        # Check if p can be added
        can_add = True
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                a, b = chosen[i], chosen[j]
                da = dist[p][a]
                db = dist[p][b]
                dc = dist[a][b]
                if da == db or da == dc or db == dc:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            chosen.append(p)
            chosen_set.add(p)
    
    return [POINTS[i] for i in chosen]
