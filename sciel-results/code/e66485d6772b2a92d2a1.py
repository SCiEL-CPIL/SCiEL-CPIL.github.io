"""Strong edge colourings of bounded-degree graphs via local search."""
import time
import random
import sys

def _generate_instance(n, d, seed):
    """Replicate the verifier's bounded_degree_graph generation."""
    rng = random.Random(seed)
    degree = [0] * n
    edges = set()
    # The seed program uses 4*n*d attempts
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            continue
        if degree[u] < d and degree[v] < d:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _build_strong_adjacency(edges, n):
    """
    Build the 'strong adjacency' relation between edges.
    Two edges are strongly adjacent if they share a vertex or are joined by an edge.
    We return a list of lists: for each edge index, the list of edge indices it conflicts with.
    Also return the vertex adjacency for quick lookup.
    """
    m = len(edges)
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Vertex adjacency
    vert_adj = [set() for _ in range(n)]
    for u, v in edges:
        vert_adj[u].add(v)
        vert_adj[v].add(u)
    
    # Strong adjacency (conflict graph on edges)
    # conflict[i] is a set of indices j such that edge i and edge j cannot have same color
    conflict = [set() for _ in range(m)]
    
    for i, (u, v) in enumerate(edges):
        # 1. Edges sharing a vertex
        for w in vert_adj[u]:
            e2 = (min(u, w), max(u, w))
            if e2 in edge_to_idx:
                j = edge_to_idx[e2]
                if j != i:
                    conflict[i].add(j)
                    conflict[j].add(i)
        for w in vert_adj[v]:
            e2 = (min(v, w), max(v, w))
            if e2 in edge_to_idx:
                j = edge_to_idx[e2]
                if j != i:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # 2. Edges joined by an edge (induced matching constraint)
        # For edge (u,v), check if there is an edge (x,y) such that x in {u,v} and y in {u,v} is false,
        # but rather: is there an edge connecting a vertex of (u,v) to a vertex of (x,y)?
        # Specifically, if there is an edge (a,b) such that a is u or v, and b is x or y, and {a,b} != {u,v}.
        # Actually, the condition "joined by an edge" means there exists a vertex a in {u,v} and b in {x,y} such that (a,b) is an edge.
        # Since we already handled sharing a vertex, we just need to check if there is an edge connecting u to y or v to x (or u to x, v to y) where {x,y} != {u,v}.
        
        # Let's iterate over neighbors of u and v again, but this time we look for edges incident to those neighbors that are NOT incident to u or v.
        # Wait, the standard definition: Two edges e1=(u,v) and e2=(x,y) are adjacent in the strong product if:
        # 1. They share a vertex.
        # 2. There is an edge connecting a vertex of e1 to a vertex of e2.
        
        # My loop above for "sharing a vertex" covers case 1.
        # For case 2:
        # If there is an edge (u, z) where z is not v, and (z, w) is an edge... no, that's path length 2.
        # "Joined by an edge" means there is an edge (a,b) such that a is in {u,v} and b is in {x,y} and (a,b) is not (u,v) and (a,b) is not (x,y).
        # Actually, if (a,b) is an edge, and a is u, b is x, then e1 and e2 are joined.
        
        # Let's refine:
        # For each neighbor z of u (z != v):
        #   If there is an edge (z, w), does it matter?
        #   The condition is: e1 and e2 are joined if there is an edge in G connecting a vertex of e1 to a vertex of e2.
        #   So, if (u, z) is an edge, and z is a vertex of e2, then e1 and e2 are joined.
        #   This means e2 must be incident to z.
        #   So, if (u, z) is an edge, and e2 is incident to z, then e1 and e2 are joined.
        #   This is already covered by "sharing a vertex" if e2 is (z, w) and e1 is (u, v)? No.
        #   If e1=(u,v) and e2=(z,w). They are joined if there is an edge between {u,v} and {z,w}.
        #   Edges between {u,v} and {z,w} could be: (u,z), (u,w), (v,z), (v,w).
        #   If (u,z) is an edge, they are joined.
        
        # So, for e1=(u,v):
        # For each z in neighbors(u) where z != v:
        #   For each e2 incident to z:
        #     If e2 != e1 and e2 does not share a vertex with e1 (i.e., z is not u or v, and the other end of e2 is not u or v):
        #        Then e1 and e2 are joined by edge (u,z).
        #     Wait, if e2 is (z, w), and w is u or v, then they share a vertex. That's case 1.
        #     So if we only consider e2 incident to z where w is NOT u or v, then they don't share a vertex.
        #     And since (u,z) is an edge, they are joined.
        
        # So, for each z in neighbors(u) \ {v}:
        #   For each e2=(z,w) in edges:
        #     if w not in {u,v}:
        #        conflict(e1, e2)
        # Similarly for neighbors(v) \ {u}.
        
    # The above logic is a bit complex to implement efficiently in a single pass without double counting.
    # Let's use a simpler, albeit slightly slower, O(m * d^2) approach for small graphs (m < 200).
    
    conflict = [set() for _ in range(m)]
    
    for i in range(m):
        u, v = edges[i]
        # Case 1: Share a vertex
        for j in range(m):
            if i == j:
                continue
            x, y = edges[j]
            # Check if they share a vertex
            if x == u or x == v or y == u or y == v:
                conflict[i].add(j)
                conflict[j].add(i)
            else:
                # Case 2: Joined by an edge
                # Check if there is an edge between {u,v} and {x,y}
                # Possible edges: (u,x), (u,y), (v,x), (v,y)
                if (u, x) in edge_to_idx or (u, y) in edge_to_idx or (v, x) in edge_to_idx or (v, y) in edge_to_idx:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    
    return conflict


def _strong_edge_coloring(n, d, seed, deadline):
    """
    Compute a strong edge coloring for a specific instance.
    Returns a list of colors (integers 0..k-1) for each edge in sorted order.
    """
    edges = _generate_instance(n, d, seed)
    m = len(edges)
    if m == 0:
        return []
    
    conflict = _build_strong_adjacency(edges, n)
    
    # Initial greedy coloring
    colors = [-1] * m
    current_max_color = 0
    
    for i in range(m):
        used_colors = set()
        for j in conflict[i]:
            if colors[j] != -1:
                used_colors.add(colors[j])
        
        # Find smallest available color
        c = 0
        while c in used_colors:
            c += 1
        colors[i] = c
        if c > current_max_color:
            current_max_color = c
            
    best_colors = colors[:]
    best_num_colors = current_max_color + 1
    
    # Local Search (WalkSAT style)
    # We want to minimize the number of distinct colors.
    # Objective: number of distinct colors used.
    # We can try to recolor an edge to a lower color if possible, or to a color already used by another edge.
    
    rng = random.Random(seed)
    
    # Precompute conflict counts for efficiency
    # conflict[i] is a set of ints.
    
    def count_colors(col):
        c = set(col)
        return len(c)
    
    # To speed up, maintain a count of how many times each color is used?
    # Or just recalculate? m is small (~50-100). Recalculating set size is O(m).
    
    num_iterations = 0
    max_iterations = 5000 # Safety cap
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and num_iterations < max_iterations and no_improve_count < max_no_improve:
        num_iterations += 1
        
        # Pick a random edge
        i = rng.randrange(m)
        
        # Try to change its color to a better one
        # "Better" means:
        # 1. A color already in use by a neighbor? No, that's invalid.
        # 2. A color < colors[i] that is not used by any neighbor.
        # 3. A color > colors[i] that is already in use (to potentially reduce max color later? No, that increases usage of that color but doesn't change distinct count unless it's the last user of colors[i]).
        
        # Strategy: 
        # 1. Try to assign the smallest possible valid color.
        # 2. If that's the same, try to assign a random valid color < current.
        # 3. If that fails, try to assign a random valid color > current (might help if it allows others to drop).
        
        # Let's just try to minimize the color of this specific edge.
        # Find all valid colors for i.
        neighbor_colors = set()
        for j in conflict[i]:
            neighbor_colors.add(colors[j])
            
        # Remove current color from consideration if we want to change?
        # No, we can keep it.
        
        # Valid colors are those not in neighbor_colors.
        # We want the smallest valid color.
        min_valid = 0
        while min_valid in neighbor_colors:
            min_valid += 1
            
        if min_valid < colors[i]:
            # Improvement: lower color
            colors[i] = min_valid
            no_improve_count = 0
            # Check if this reduced the total distinct colors?
            # If colors[i] was the only edge with that color, and min_valid is already used, then distinct count drops.
            # If min_valid is not used, distinct count stays same (swapped).
            # We track best_num_colors at the end or periodically.
            
        elif min_valid == colors[i]:
            # No improvement possible by lowering.
            # Try a random valid color to escape local minima?
            # Or just skip.
            # Let's try a random valid color different from current to shuffle.
            # But only if it doesn't increase the max color?
            # Actually, WalkSAT usually flips a variable to satisfy clauses. Here, "satisfying" is having a valid coloring.
            # We always have a valid coloring. We want to minimize distinct colors.
            
            # Let's try to move to a color that is already used by another edge, to potentially merge color classes.
            # This is hard to do directly.
            
            # Alternative: Simulated Annealing / Random Walk on the space of colorings.
            # Energy = number of distinct colors.
            # Move: recolor edge i to a random valid color.
            # Accept if energy decreases.
            # Accept if energy increases with probability exp(-delta/T).
            
            # Let's implement a simple random walk.
            # Generate a random valid color.
            # There might be many.
            # Let's pick a random color from 0 to current_max_color + 1.
            
            # Actually, let's just try to recolor to a random valid color.
            # To avoid infinite loops, we limit this.
            
            # Let's try to recolor to a color that is in neighbor_colors? No, that's invalid.
            
            # Let's try to recolor to a random valid color.
            # How to pick?
            # The valid colors are infinite. But we only care about 0..current_max_color.
            # Let's pick a random c in 0..current_max_color.
            # If c is not in neighbor_colors, it's valid.
            
            # If we can't find one quickly, we skip.
            
            # For now, let's just skip if no lower color is available.
            no_improve_count += 1
            
        else:
            # min_valid > colors[i] is impossible since colors[i] is valid, so it's not in neighbor_colors.
            # So min_valid <= colors[i].
            pass
            
        # Update best
        current_distinct = len(set(colors))
        if current_distinct < best_num_colors:
            best_num_colors = current_distinct
            best_colors = colors[:]
            no_improve_count = 0 # Reset on improvement
        
    return best_colors


def construct():
    start_time = time.time()
    # Total budget 120s. Keep 10s safety margin.
    deadline = start_time + 110.0
    
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Allocate time roughly proportional to difficulty (m) or just split evenly?
    # m is roughly 2*n*d / d = 2n? No, m is number of edges.
    # For n=40, d=4, m ~ 80?
    # For n=60, d=5, m ~ 120?
    # For n=80, d=6, m ~ 160?
    
    # Let's give each instance 1/3 of the time.
    time_per_instance = (deadline - start_time) / 3.0
    
    results = []
    
    for i, (n, d, seed) in enumerate(specs):
        inst_start = time.time()
        inst_deadline = inst_start + time_per_instance - 1.0 # 1s buffer between instances
        
        # If we are already running out of time, use remaining time
        if time.time() > deadline - 1.0:
            inst_deadline = deadline - 0.1
        
        colors = _strong_edge_coloring(n, d, seed, inst_deadline)
        results.append(colors)
        
        # Update deadline for next instance if we finished early?
        # No, the total deadline is fixed.
        
    return results


# Test locally
if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} edges, max color {max(r) if r else 0}, distinct colors {len(set(r)) if r else 0}")
