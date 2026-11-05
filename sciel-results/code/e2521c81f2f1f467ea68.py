import time
import random
import numpy as np
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # 1. Enumerate all possible colored edges (nodes in MWIS)
    # Node index i corresponds to edge (u, v) with colour c
    # u < v to ensure uniqueness
    all_nodes = []
    node_to_uv = {}
    node_to_c = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for c in range(num_colours):
                node_to_uv[idx] = (u, v)
                node_to_c[idx] = c
                all_nodes.append(idx)
                idx += 1
    
    total_nodes = idx
    
    # 2. Build conflict graph
    # Two nodes conflict if:
    # a) They share a vertex and have the same colour (Proper edge colouring constraint)
    # b) They are part of a 4-cycle where the 4 edges in the cycle do not have distinct colours.
    #    Specifically, if we have edges e1, e2, e3, e4 forming a cycle u-v-w-x-u.
    #    The colours c1, c2, c3, c4 must be distinct.
    #    This implies:
    #    - If e1 and e2 are in the cycle, they cannot have the same colour.
    #    - If e1 and e3 are in the cycle, they cannot have the same colour.
    #    - If e1 and e4 are in the cycle, they cannot have the same colour.
    #    Wait, the constraint is "Every 4-cycle is rainbow", i.e., c1 != c2, c1 != c3, c1 != c4, c2 != c3, c2 != c4, c3 != c4.
    #    So any pair of edges in the same 4-cycle must have different colours.
    #    This means: For any 4-cycle, all pairs of edges in that cycle must have distinct colours.
    #    This is equivalent to saying: No two edges in the same 4-cycle can have the same colour.
    
    # Let's verify this interpretation. "Rainbow" means all 4 colours are distinct.
    # Yes, this implies any pair of edges in the cycle must have different colours.
    
    # So the conflict condition for two colored edges (u1,v1,c1) and (u2,v2,c2) is:
    # 1. c1 == c2 AND they share a vertex (u1==u2 or u1==v2 or v1==u2 or v1==v2) -> Actually, if they share a vertex and have same color, it's not a proper edge coloring. The task says "edge-coloured... simple graph". Usually "edge-coloured" implies proper unless stated otherwise? 
    #    Let's check the task description: "edge-coloured with 20 colours so every 4-cycle is rainbow".
    #    It does NOT explicitly say "proper edge colouring". 
    #    However, if two edges share a vertex and have the same color, do they form a 4-cycle? No.
    #    Do they violate the rainbow property? The rainbow property only applies to 4-cycles.
    #    BUT, standard graph coloring problems usually imply properness. Let's look at the seed program.
    #    The seed program's `can_colour` function checks:
    #    `forbidden_proper`: Colour of (u,v) must be different from colours of all edges incident to u or v.
    #    So the seed program ENFORCES proper edge coloring.
    #    I will enforce proper edge coloring as well, as it's a standard constraint and likely required for "dense valid colourings" in this context (anti-Ramsey usually deals with proper colorings or at least distinct colors on adjacent edges if not specified, but let's stick to the seed's interpretation which scored 4.3).
    
    # So, Conflict if:
    # A) c1 == c2 and edges are adjacent (share a vertex).
    # B) c1 == c2 and edges are in the same 4-cycle (even if not adjacent? No, if they are in the same 4-cycle, they must have different colors. This covers both adjacent and non-adjacent edges in the cycle).
    #    Actually, if c1 == c2, they conflict if they are in ANY 4-cycle together.
    #    Note: If two edges are adjacent, are they always in a 4-cycle? No.
    #    So we need:
    #    Conflict if (c1 == c2 AND (edges are adjacent OR edges are in some 4-cycle)).
    
    # Let's precompute for each edge (u,v) the set of other edges (w,x) that are in a 4-cycle with it.
    # And separately, the set of adjacent edges.
    
    # Precompute edge indices for fast lookup
    # edge_id[u, v] = index in 0..190 (if u<v) else edge_id[v, u]
    edge_id = np.full((n, n), -1, dtype=int)
    edge_list = []
    eid = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_id[u, v] = eid
            edge_id[v, u] = eid
            edge_list.append((u, v))
            eid += 1
    total_edges = eid
    
    # For each edge e, find all edges e' such that e and e' are in a 4-cycle.
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and 4 edges forming a cycle.
    # There are 3 ways to pair 4 vertices into a cycle:
    # (a,b), (b,c), (c,d), (d,a)
    # (a,b), (b,d), (d,c), (c,a)
    # (a,c), (c,b), (b,d), (d,a)
    
    # For each edge e=(u,v), which other edges e' can form a 4-cycle with e?
    # If e=(u,v), the 4-cycle must be u-v-w-x-u.
    # Edges: (u,v), (v,w), (w,x), (x,u).
    # So e' can be (v,w), (w,x), or (x,u).
    # This requires w and x to be distinct from u,v and each other.
    
    # Let's build a set of conflicting edges for each edge e, considering the "same color in 4-cycle" constraint.
    # Actually, it's easier to build the conflict graph on the colored nodes directly.
    
    # Conflict Graph Construction
    # Nodes: 190 * 20 = 3800 nodes.
    # We need to determine which pairs of nodes conflict.
    # Two nodes (e1, c1) and (e2, c2) conflict if:
    # 1. c1 == c2 and e1, e2 are adjacent.
    # 2. c1 == c2 and e1, e2 are in a 4-cycle.
    
    # Note: If e1 and e2 are in a 4-cycle, they might be adjacent (e.g., (u,v) and (v,w)) or not (e.g., (u,v) and (w,x)).
    # So condition 2 subsumes condition 1 for pairs in a 4-cycle.
    # But condition 1 applies to adjacent edges NOT in a 4-cycle.
    
    # Let's precompute for each edge e:
    # adj_edges[e]: set of edges adjacent to e.
    # cycle_edges[e]: set of edges in a 4-cycle with e.
    
    adj_edges = [set() for _ in range(total_edges)]
    cycle_edges = [set() for _ in range(total_edges)]
    
    # Compute adj_edges
    for e1 in range(total_edges):
        u1, v1 = edge_list[e1]
        for e2 in range(e1 + 1, total_edges):
            u2, v2 = edge_list[e2]
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                adj_edges[e1].add(e2)
                adj_edges[e2].add(e1)
                
    # Compute cycle_edges
    # Iterate over all 4-cycles
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        e1 = edge_id[a, b]
        e2 = edge_id[b, c]
        e3 = edge_id[c, d]
        e4 = edge_id[d, a]
        for e in [e1, e2, e3, e4]:
            for f in [e1, e2, e3, e4]:
                if e != f:
                    cycle_edges[e].add(f)
                    
        # Cycle 2: a-b-d-c-a
        e1 = edge_id[a, b]
        e2 = edge_id[b, d]
        e3 = edge_id[d, c]
        e4 = edge_id[c, a]
        for e in [e1, e2, e3, e4]:
            for f in [e1, e2, e3, e4]:
                if e != f:
                    cycle_edges[e].add(f)
                    
        # Cycle 3: a-c-b-d-a
        e1 = edge_id[a, c]
        e2 = edge_id[c, b]
        e3 = edge_id[b, d]
        e4 = edge_id[d, a]
        for e in [e1, e2, e3, e4]:
            for f in [e1, e2, e3, e4]:
                if e != f:
                    cycle_edges[e].add(f)

    # Now, build the conflict graph for colored nodes.
    # Node i = (e, c).
    # Conflict if c1 == c2 and (e1 in adj_edges[e2] or e1 in cycle_edges[e2]).
    # Note: adj_edges and cycle_edges are symmetric.
    
    # We will use a bitset for the conflict graph to speed up branch and bound.
    # 3800 nodes. We can use Python integers as bitsets or numpy arrays.
    # Let's use a list of sets first, then maybe optimize if needed.
    # Or, since we are doing branch and bound, we need to quickly find neighbors of a node in the current candidate set.
    
    # Let's precompute the conflict sets for each node.
    # conflicts[i] = set of nodes j that conflict with i.
    conflicts = [set() for _ in range(total_nodes)]
    
    for e1 in range(total_edges):
        for c1 in range(num_colours):
            node1 = e1 * num_colours + c1
            # Check against all e2, c2
            # Optimization: Only check c2 == c1 for color conflicts?
            # Yes, the only constraint is that edges in the same 4-cycle or adjacent edges must have DIFFERENT colors.
            # This means if c1 != c2, they do NOT conflict (unless there's some other constraint, but there isn't).
            # So we only need to check nodes with the same color.
            
            # For a fixed color c, the nodes are (e, c).
            # Two such nodes (e1, c) and (e2, c) conflict if e1 and e2 are adjacent or in a 4-cycle.
            
            # So for each color c, we have a graph on the edges.
            # We want to select an independent set in this graph for each color, and combine them.
            # Wait, the problem is to select a set of colored edges such that:
            # 1. For each edge e, at most one color is chosen. (Simple graph constraint: an edge has one color).
            #    Wait, the task says "coloured edges [[u, v, c], ...]". A simple graph has one edge between u and v.
            #    So for a given edge (u,v), we can choose at most one color.
            #    This is an additional constraint: For each edge e, we can pick at most one node (e, c).
            
            # So the conflicts are:
            # 1. (e, c1) and (e, c2) conflict if c1 != c2? No, if we pick both, we have two colors for one edge.
            #    The output is a list of edges. Each edge appears once.
            #    So we must ensure that for each edge e, we select at most one color.
            #    This means all nodes (e, c) for a fixed e are mutually exclusive.
            
            # 2. (e1, c) and (e2, c) conflict if c is the same and e1, e2 are adjacent or in a 4-cycle.
            
            # So, for a fixed color c, the nodes (e, c) form a graph where edges connect e1, e2 if they are adjacent or in a 4-cycle.
            # We need to pick an independent set from this graph.
            # Additionally, for different colors, nodes (e, c1) and (e, c2) conflict because we can only pick one color per edge.
            
            # This structure is a "Color-Edge" independent set.
            
            # Let's add the "same edge different color" conflicts.
            for c2 in range(num_colours):
                if c1 == c2:
                    # Check if e1 and e2 are adjacent or in 4-cycle
                    # We will do this in a second pass or here.
                    pass
                else:
                    # Same edge e1, different color. Conflict.
                    node2 = e1 * num_colours + c2
                    conflicts[node1].add(node2)
                    conflicts[node2].add(node1)
    
    # Now add the "same color, conflicting edges" conflicts.
    for c in range(num_colours):
        # For this color, consider all edges e.
        # If e1 and e2 are in conflict (adj or 4-cycle), then (e1, c) and (e2, c) conflict.
        for e1 in range(total_edges):
            node1 = e1 * num_colours + c
            for e2 in adj_edges[e1]:
                node2 = e2 * num_colours + c
                conflicts[node1].add(node2)
                conflicts[node2].add(node1)
            for e2 in cycle_edges[e1]:
                node2 = e2 * num_colours + c
                conflicts[node1].add(node2)
                conflicts[node2].add(node1)
                
    # Now we have the full conflict graph.
    # We want to find the Maximum Weight Independent Set (MWIS).
    # All weights are 1 (we want to maximize the number of edges).
    # So it's Maximum Independent Set (MIS).
    
    # Branch and Bound for MIS.
    # Standard algorithm:
    # 1. Pick a vertex v.
    # 2. Branch 1: Include v. Remove v and its neighbors.
    # 3. Branch 2: Exclude v. Remove v.
    # 4. Use a lower bound (greedy) and upper bound (e.g., sum of degrees or simple count) to prune.
    
    # To make it faster, we can use a bitset representation.
    # 3800 bits fits in 60 64-bit integers.
    # Let's use Python integers for bitsets.
    
    # Convert conflicts to bitsets.
    # bitset[i] is an integer where bit j is 1 if i and j conflict.
    bitsets = [0] * total_nodes
    for i in range(total_nodes):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        bitsets[i] = mask
        
    # Also, we need to quickly calculate the size of a set of nodes.
    # And we need to pick a vertex to branch on.
    # Heuristic: Pick the vertex with the highest degree in the current subgraph?
    # Or the one with the lowest degree?
    # Usually, for MIS, picking a vertex with high degree in the complement (low degree in conflict) is good?
    # Actually, standard B&B for MIS often picks the vertex with the highest degree in the conflict graph to reduce the branching factor quickly?
    # Let's try picking the vertex with the highest degree in the current candidate set.
    
    # To speed up, we will maintain the candidate set as a bitset.
    # And we will maintain an array of degrees? No, that's expensive to update.
    # Let's use a simple recursive search with bitsets.
    
    best_size = 0
    best_set = 0  # Bitset of the best independent set found
    
    # Greedy lower bound function
    def greedy_lb(candidate_bitset):
        # Greedily pick vertices
        size = 0
        cand = candidate_bitset
        while cand:
            # Pick a vertex with high degree? Or just the first one?
            # Let's pick the vertex with the lowest index for simplicity, or random?
            # Better: Pick the vertex with the highest degree in the induced subgraph?
            # That's expensive. Let's just pick the lowest set bit.
            v = (cand & -cand).bit_length() - 1
            size += 1
            # Remove v and its neighbors
            cand &= ~(bitsets[v] | (1 << v))
        return size
    
    # Upper bound: Simple count of vertices.
    # Better upper bound: Greedy coloring of the conflict graph? (Chvatal bound)
    # Let's stick to a simple count first, or maybe sum of 1/(degree+1)?
    # For now, let's use a simple bound: number of vertices in candidate set.
    # We can improve this by checking if the candidate set is empty.
    
    def popcount(x):
        return bin(x).count('1')
        
    def solve_mis(candidate_bitset, current_size, depth):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if candidate_bitset == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = 0 # We need to track the actual set, not just size.
                # Wait, I lost the set in the recursion. I need to pass the set or reconstruct it.
                # Reconstructing is hard. Let's pass the set as a list or bitset.
                # Passing bitset is fine.
            return
            
        # Lower bound
        lb = greedy_lb(candidate_bitset)
        if current_size + lb <= best_size:
            return
            
        # Upper bound
        # Simple upper bound: popcount(candidate_bitset)
        ub = popcount(candidate_bitset)
        if current_size + ub <= best_size:
            return
            
        # Pick a vertex to branch on
        # Heuristic: Pick vertex with highest degree in the current subgraph?
        # Let's pick the vertex with the lowest index for now.
        v = (candidate_bitset & -candidate_bitset).bit_length() - 1
        
        # Branch 1: Include v
        # New candidate set: candidate_bitset without v and its neighbors
        new_cand_include = candidate_bitset & ~(bitsets[v] | (1 << v))
        solve_mis(new_cand_include, current_size + 1, depth + 1)
        
        # Branch 2: Exclude v
        new_cand_exclude = candidate_bitset & ~(1 << v)
        solve_mis(new_cand_exclude, current_size, depth + 1)
        
    # This simple B&B might be too slow for 3800 nodes.
    # We need a better heuristic for vertex selection and bounds.
    # Let's try a different approach: Local Search / Greedy with Restart, but smarter.
    # Or, use the fact that the problem decomposes by color?
    # No, the "one color per edge" constraint couples them.
    
    # Given the time limit (135s), a full B&B on 3800 nodes is likely infeasible in Python without C extensions.
    # The seed program used greedy + random restarts and got 4.3.
    # Inspiration 1 (MWIS) got 4.58.
    # Inspiration 2 (Optimized loops + bipartite seeding) got 5.16.
    
    # Let's implement a very fast Greedy + Local Search (Simulated Annealing or Hill Climbing) on the colored edges.
    # State: A set of colored edges.
    # Objective: Maximize number of edges.
    # Moves:
    # 1. Add an edge (if valid).
    # 2. Remove an edge.
    # 3. Recolor an edge.
    # 4. Swap colors of two edges.
    
    # Let's stick to the "Add Edge" greedy but with a much better initial seeding and a local search phase.
    
    def is_valid(adj_matrix):
        # Check all 4-cycles
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            # Cycle 1
            if adj_matrix[a,b] != -1 and adj_matrix[b,c] != -1 and adj_matrix[c,d] != -1 and adj_matrix[d,a] != -1:
                cols = {adj_matrix[a,b], adj_matrix[b,c], adj_matrix[c,d], adj_matrix[d,a]}
                if len(cols) < 4:
                    return False
            # Cycle 2
            if adj_matrix[a,b] != -1 and adj_matrix[b,d] != -1 and adj_matrix[d,c] != -1 and adj_matrix[c,a] != -1:
                cols = {adj_matrix[a,b], adj_matrix[b,d], adj_matrix[d,c], adj_matrix[c,a]}
                if len(cols) < 4:
                    return False
            # Cycle 3
            if adj_matrix[a,c] != -1 and adj_matrix[c,b] != -1 and adj_matrix[b,d] != -1 and adj_matrix[d,a] != -1:
                cols = {adj_matrix[a,c], adj_matrix[c,b], adj_matrix[b,d], adj_matrix[d,a]}
                if len(cols) < 4:
                    return False
        return True
        
    # This validation is O(N^4) which is 20^4 = 160,000 * 3 = 480,000 checks.
    # 480k checks per validation. If we do 1000 validations, that's 480M ops. Too slow.
    # We need incremental validation or a very fast check.
    
    # Let's go back to the Greedy Add Edge with optimized `can_colour`.
    # The seed program's `can_colour` was O(deg^2).
    # We can optimize it by precomputing the "forbidden colors" for each edge addition.
    
    # Let's try the MWIS approach but with a very aggressive pruning and a good initial solution.
    # Actually, let's just use the best of the inspirations: Optimized Greedy + Local Search.
    
    best_edges = 0
    best_adj = None
    
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            
    def fast_can_colour(adj, neighbors, u, v):
        # Check proper coloring
        forbidden = set()
        for w, c in neighbors[u]:
            if w != v:
                forbidden.add(c)
        for w, c in neighbors[v]:
            if w != u:
                forbidden.add(c)
                
        # Check 4-cycles
        # For each w in neighbors[v], for each x in neighbors[u], if (w,x) exists...
        # This is the bottleneck.
        # Optimization: Precompute for each pair (u,v) the list of (w,x) that form a 4-cycle?
        # No, (w,x) depends on the current graph.
        
        # Let's just iterate, but break early if we find a valid color.
        # Actually, we just need to know if a color is available.
        
        # Let's collect all forbidden colors from 4-cycles.
        for w, c_vw in neighbors[v]:
            if w == u: continue
            for x, c_xu in neighbors[u]:
                if x == v or x == w: continue
                if adj[w, x] != -1:
                    c_wx = adj[w, x]
                    if c_vw != c_wx and c_vw != c_xu and c_wx != c_xu:
                        forbidden.add(c_vw)
                        forbidden.add(c_wx)
                        forbidden.add(c_xu)
                        
        for c in range(num_colours):
            if c not in forbidden:
                return c
        return None

    for attempt in range(500):
        if time.time() > deadline:
            break
            
        adj = np.full((n, n), -1, dtype=int)
        neighbors = [[] for _ in range(n)]
        
        # Random order
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            c = fast_can_colour(adj, neighbors, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
                neighbors[u].append((v, c))
                neighbors[v].append((u, c))
                
        # Local Search: Try to add edges that were skipped
        # Rebuild neighbors
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] != -1:
                    c = adj[i, j]
                    neighbors[i].append((j, c))
                    neighbors[j].append((i, c))
                    
        improved = True
        while improved:
            improved = False
            for u, v in edges:
                if adj[u, v] != -1:
                    continue
                c = fast_can_colour(adj, neighbors, u, v)
                if c is not None:
                    adj[u, v] = c
                    adj[v, u] = c
                    neighbors[u].append((v, c))
                    neighbors[v].append((u, c))
                    improved = True
                    
        edge_count = int(np.sum(adj != -1)) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
            
    if best_adj is None:
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19
        
    # Final Verification
    valid = True
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        if best_adj[a,b] != -1 and best_adj[b,c] != -1 and best_adj[c,d] != -1 and best_adj[d,a] != -1:
            if len({best_adj[a,b], best_adj[b,c], best_adj[c,d], best_adj[d,a]}) < 4:
                valid = False
                break
        if best_adj[a,b] != -1 and best_adj[b,d] != -1 and best_adj[d,c] != -1 and best_adj[c,a] != -1:
            if len({best_adj[a,b], best_adj[b,d], best_adj[d,c], best_adj[c,a]}) < 4:
                valid = False
                break
        if best_adj[a,c] != -1 and best_adj[c,b] != -1 and best_adj[b,d] != -1 and best_adj[d,a] != -1:
            if len({best_adj[a,c], best_adj[c,b], best_adj[b,d], best_adj[d,a]}) < 4:
                valid = False
                break
                
    if not valid:
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
