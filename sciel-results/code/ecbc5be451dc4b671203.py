import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Best known solution is n=17, but we might not find it.
    # Current best in population is n=13 (score 1.44).
    # We will try to find n=13, 14, 15, 16, 17.
    # Start with a valid small solution to guarantee a score.
    # n=9 is easy.
    best_n = 9
    best_colors = [0] * (9*8//2)
    # Simple valid coloring for n=9:
    # Color (i,j) = (j-i) % 4
    idx = 0
    for i in range(9):
        for j in range(i+1, 9):
            best_colors[idx] = (j-i) % 4
            idx += 1
            
    # Try to find larger n
    # Order: 13, 14, 15, 16, 17
    # Allocate time based on difficulty
    # n=13: 30s, n=14: 20s, n=15: 10s, n=16: 5s, n=17: 5s
    targets = [
        (13, 30.0),
        (14, 20.0),
        (15, 10.0),
        (16, 5.0),
        (17, 5.0)
    ]
    
    for n, budget in targets:
        if time.time() > deadline - 2.0:
            break
            
        end_time = time.time() + budget
        res = _find_coloring_local_search(n, end_time)
        if res:
            return res
            
    return {'n': best_n, 'colors': best_colors}

def _find_coloring_local_search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute edge index
    edge_idx = {}
    for k, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = k
        edge_idx[(v, u)] = k
        
    # Precompute all C6 cycles
    # A C6 is a sequence of 6 distinct vertices v0..v5 such that edges (v0,v1)..(v5,v0) exist.
    # In K_n, all edges exist.
    # We need to identify unique cycles.
    # Two cycles are the same if they have the same set of edges?
    # "No colour class may contain a six-cycle"
    # Usually means the subgraph induced by one color does not contain a cycle of length 6.
    # A cycle is a set of edges.
    # So we need to enumerate all sets of 6 edges that form a C6.
    
    # Number of C6s in K_n:
    # Choose 6 vertices: C(n, 6)
    # Number of distinct C6s on 6 labeled vertices: (6-1)!/2 = 60
    # Total cycles = C(n, 6) * 60
    
    # For n=13: C(13,6) = 1716. 1716 * 60 = 102,960 cycles.
    # For n=17: C(17,6) = 12376. 12376 * 60 = 742,560 cycles.
    
    # We will generate all cycles.
    # To save time, we can generate them on the fly or precompute.
    # Precomputation for n=17 might take a second or two.
    
    import itertools
    
    cycles = []
    edge_to_cycles = [[] for _ in range(num_edges)]
    
    # Generate all combinations of 6 vertices
    # This might be slow for n=17.
    # Let's try to optimize.
    
    # Alternative: Just store the 6 edges for each cycle.
    # And for each edge, the list of cycle indices.
    
    # We can generate cycles by picking 6 vertices and then the 60 permutations.
    # To avoid duplicates, we can fix the smallest vertex to be v0, and the second smallest to be v1?
    # No, that's for undirected cycles.
    # A cycle is defined by the set of edges.
    # Two permutations that result in the same set of edges are the same cycle.
    # The 60 permutations correspond to the 60 distinct edge sets?
    # Yes, for K6, there are 60 distinct C6 subgraphs.
    
    # Let's precompute the 60 edge patterns for 6 vertices 0..5.
    # Then for each combination of 6 vertices, map the pattern to the actual edge indices.
    
    # Generate 60 patterns for vertices 0..5
    # Fix vertex 0.
    # Permute 1..5.
    # Cycle: 0 - p1 - p2 - p3 - p4 - p5 - 0
    # Edges: (0,p1), (p1,p2), (p2,p3), (p3,p4), (p4,p5), (p5,0)
    
    base_patterns = []
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = [0] + list(perm)
        edges_in_cycle = []
        for i in range(6):
            u, w = v[i], v[(i+1)%6]
            if u > w: u, w = w, u
            edges_in_cycle.append((u, w))
        # Check if this edge set is unique
        # We can use a set of frozensets of edges
        edge_set = frozenset(edges_in_cycle)
        if edge_set not in [frozenset(p) for p in base_patterns]:
            base_patterns.append(tuple(sorted(edge_set)))
            
    # There should be 60 unique patterns.
    # Map (u,w) in 0..5 to index in 0..14
    def get_edge_idx_6(u, w):
        if u > w: u, w = w, u
        # 0: 01, 1: 02, 2: 03, 3: 04, 4: 05
        # 5: 12, 6: 13, 7: 14, 8: 15
        # 9: 23, 10: 24, 11: 25
        # 12: 34, 13: 35
        # 14: 45
        # Formula:
        # If u=0: w-1
        # If u=1: 5 + (w-2)
        # If u=2: 9 + (w-3)
        # If u=3: 12 + (w-4)
        # If u=4: 14
        if u == 0: return w - 1
        if u == 1: return 5 + (w - 2)
        if u == 2: return 9 + (w - 3)
        if u == 3: return 12 + (w - 4)
        if u == 4: return 14
        return -1
        
    # Convert base_patterns to edge indices
    base_edge_indices = []
    for pat in base_patterns:
        idxs = []
        for (u, w) in pat:
            idxs.append(get_edge_idx_6(u, w))
        base_edge_indices.append(tuple(idxs))
        
    # Now generate all cycles for K_n
    # For each combination of 6 vertices
    for verts in itertools.combinations(range(n), 6):
        # Map local vertex 0..5 to global vertices
        # local i -> verts[i]
        for pat in base_edge_indices:
            global_edges = []
            for local_e_idx in pat:
                # Recover u, w from local_e_idx?
                # We lost the u,w mapping.
                # We need to store the (u,w) pairs in base_patterns.
                pass
                
    # Let's redo base_patterns to store (u,w) pairs.
    base_patterns_uw = []
    seen = set()
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = [0] + list(perm)
        edges_in_cycle = []
        for i in range(6):
            u, w = v[i], v[(i+1)%6]
            if u > w: u, w = w, u
            edges_in_cycle.append((u, w))
        edge_set = frozenset(edges_in_cycle)
        if edge_set not in seen:
            seen.add(edge_set)
            base_patterns_uw.append(tuple(sorted(edges_in_cycle)))
            
    # Now generate global cycles
    # This loop might be slow.
    # C(17,6) * 60 = 742,560 iterations.
    # Inside: 6 lookups.
    # Should be fast enough (< 1s).
    
    cycles_edges = []
    edge_to_cycles = [[] for _ in range(num_edges)]
    
    for verts in itertools.combinations(range(n), 6):
        # Map local 0..5 to global
        # local i -> verts[i]
        for pat in base_patterns_uw:
            global_e_idxs = []
            for (u_local, w_local) in pat:
                u_g = verts[u_local]
                w_g = verts[w_local]
                e_idx = edge_idx[(u_g, w_g)]
                global_e_idxs.append(e_idx)
            
            cycle_idx = len(cycles_edges)
            cycles_edges.append(tuple(global_e_idxs))
            
            for e in global_e_idxs:
                edge_to_cycles[e].append(cycle_idx)
                
    # Local Search
    # State: colors array
    # Cost: number of monochromatic cycles
    
    # Initialize colors randomly
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Initial cost calculation
    # is_mono[cycle_idx]
    is_mono = [False] * len(cycles_edges)
    cost = 0
    for c_idx, c_edges in enumerate(cycles_edges):
        c0 = colors[c_edges[0]]
        mono = True
        for e in c_edges[1:]:
            if colors[e] != c0:
                mono = False
                break
        if mono:
            is_mono[c_idx] = True
            cost += 1
            
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # We want to minimize cost.
    # If cost becomes 0, we found a solution.
    
    # To speed up, we can try hill climbing first.
    
    def get_delta_cost(edge_idx, new_color):
        old_color = colors[edge_idx]
        if old_color == new_color:
            return 0
            
        delta = 0
        for c_idx in edge_to_cycles[edge_idx]:
            # Check if cycle was mono
            if is_mono[c_idx]:
                delta -= 1 # Remove from count
                
            # Check if cycle will be mono with new_color
            # Get other 5 edges
            c_edges = cycles_edges[c_idx]
            # Find position of edge_idx
            # We can precompute positions?
            # Or just check all 6.
            # If all 6 are new_color, then it becomes mono.
            # Note: the edge we are changing is one of them.
            # So we check if the other 5 are new_color.
            # And the edge itself will be new_color.
            # So check if all 5 others are new_color.
            
            # Optimization: if is_mono was True, the old color was the one.
            # If new_color != old_color, it definitely won't be mono with old_color.
            # So we just need to check if all edges are new_color.
            
            mono_new = True
            for e in c_edges:
                if e == edge_idx:
                    continue
                if colors[e] != new_color:
                    mono_new = False
                    break
            
            if mono_new:
                delta += 1 # Add to count
                
        return delta
        
    # Run SA
    # We need to be careful with time.
    steps = 0
    while time.time() < deadline - 1.0:
        if cost == 0:
            return {'n': n, 'colors': colors}
            
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        old_c = colors[e]
        new_c = random.choice([c for c in range(4) if c != old_c])
        
        delta = get_delta_cost(e, new_c)
        
        # Accept if delta < 0 or with probability
        if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 0.001)):
            colors[e] = new_c
            # Update is_mono and cost
            # Actually, get_delta_cost already calculated the change.
            # But we need to update is_mono array.
            # We can re-evaluate is_mono for affected cycles.
            for c_idx in edge_to_cycles[e]:
                c_edges = cycles_edges[c_idx]
                c0 = colors[c_edges[0]]
                mono = True
                for ee in c_edges[1:]:
                    if colors[ee] != c0:
                        mono = False
                        break
                is_mono[c_idx] = mono
                
            cost += delta
            
        T *= cooling
        
        # Restart if stuck?
        # Simple restart: if no improvement for 1000 steps, reinitialize?
        # Or just keep going.
        
    return None

# To ensure we don't timeout on precomputation for large n, we can add a check.
# But n=17 precomputation is fast.

if __name__ == "__main__":
    res = construct()
    print(res['n'])
