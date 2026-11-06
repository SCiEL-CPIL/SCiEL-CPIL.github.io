import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Generate all triples
    points = range(16)
    all_triples = [tuple(t) for t in combinations(points, 3)]
    n_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # 2. Build Conflict Graph
    # Two triples conflict if they are part of a K(2,2,2) configuration.
    # Specifically, a set of 8 triples forms a K(2,2,2) if they are the transversals
    # of three disjoint pairs (x1,y1), (x2,y2), (x3,y3).
    # The triples are:
    # (x1,x2,x3), (x1,x2,y3), (x1,y2,x3), (x1,y2,y3),
    # (y1,x2,x3), (y1,x2,y3), (y1,y2,x3), (y1,y2,y3)
    
    # To build the conflict graph efficiently:
    # For each set of 3 disjoint pairs, identify the 8 transversal triples.
    # Any pair of these 8 triples conflicts.
    
    # There are C(16,2)*C(14,2)*C(12,2) / 6 ways to choose 3 disjoint pairs?
    # Actually, it's easier to iterate over all partitions of 6 distinct points into 3 pairs.
    # Number of such partitions is C(16,6) * 15 = 8008 * 15 = 120120.
    # For each partition, we have 8 triples. We add edges between all pairs in this set of 8.
    
    # However, we only need to know if a set of triples is independent.
    # An independent set is a set of triples with NO conflicts.
    # So we need to mark conflicts.
    
    # Let's use a bitset representation for the conflict graph.
    # 560 nodes. 560 bits = 70 bytes = 2 uint64s per node.
    # We can use Python integers as bitsets (arbitrary precision).
    
    conflict_mask = [0] * n_triples
    
    # Precompute the 8 transversals for a given set of 6 points and pairing.
    # A pairing is defined by 3 pairs.
    # Let's iterate over all 6-subsets of 16.
    # For each 6-subset, there are 15 ways to pair them up.
    # Total iterations: 8008 * 15 = 120120.
    # For each, we get 8 triples. We add edges between all pairs in this set of 8.
    # This is 28 edges per partition.
    # Total edges to process: 120120 * 28 = 3.3 million.
    # This might be slow in pure Python if done naively.
    
    # Optimization:
    # We can precompute the mapping from a 6-set to the 15 pairings.
    # Or just iterate.
    
    # Let's check time. 3.3M iterations with bit operations.
    # Python bit operations on large integers are C-speed.
    # Setting bits: mask[i] |= (1 << j)
    
    # To speed up, we can use a list of lists for neighbors first, then convert to bitmasks?
    # No, bitmasks are better for the branch and bound.
    
    # Let's try to optimize the generation.
    # Instead of iterating all 6-sets, we can iterate over the 8 triples directly?
    # No, the structure is defined by the 3 pairs.
    
    # Let's run the generation.
    
    # Helper to get the 8 transversals from 3 pairs (a,b), (c,d), (e,f)
    def get_transversals(p1, p2, p3):
        a, b = p1
        c, d = p2
        e, f = p3
        # The points are a,b,c,d,e,f
        # The triples are all combinations of choosing one from each pair?
        # No. The triples are (x1, x2, x3) where xi is from pair i?
        # Let's re-read the definition.
        # "no three disjoint pairs have all eight transversal triples present"
        # Transversals of pairs (x1,y1), (x2,y2), (x3,y3) are:
        # (x1, x2, x3), (x1, x2, y3), (x1, y2, x3), (x1, y2, y3)
        # (y1, x2, x3), (y1, x2, y3), (y1, y2, x3), (y1, y2, y3)
        # Yes.
        
        return [
            (a, c, e), (a, c, f), (a, d, e), (a, d, f),
            (b, c, e), (b, c, f), (b, d, e), (b, d, f)
        ]
    
    # We need to ensure the triples are sorted for indexing.
    
    # Iterate over all 6-subsets
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    for m in range(l+1, 16):
                        for n in range(m+1, 16):
                            if time.time() > deadline - 60: # Stop generating if running out of time for search
                                # Fallback to random greedy if graph construction is too slow?
                                # Unlikely, 120k iterations is fast.
                                pass
                            
                            pts = [i, j, k, l, m, n]
                            # Generate 15 pairings of 6 elements
                            # Pairing 1: (0,1), (2,3), (4,5)
                            # We can generate these recursively or use a fixed list of permutations.
                            # There are 15 pairings.
                            # Let's list them by fixing the partner of 0.
                            # 0 can pair with 1,2,3,4,5.
                            # If 0-1, remaining 2,3,4,5 have 3 pairings.
                            # If 0-2, remaining 1,3,4,5 have 3 pairings.
                            # etc.
                            
                            # Pairings of indices 0..5:
                            pairings_indices = [
                                [(0,1), (2,3), (4,5)],
                                [(0,1), (2,4), (3,5)],
                                [(0,1), (2,5), (3,4)],
                                [(0,2), (1,3), (4,5)],
                                [(0,2), (1,4), (3,5)],
                                [(0,2), (1,5), (3,4)],
                                [(0,3), (1,2), (4,5)],
                                [(0,3), (1,4), (2,5)],
                                [(0,3), (1,5), (2,4)],
                                [(0,4), (1,2), (3,5)],
                                [(0,4), (1,3), (2,5)],
                                [(0,4), (1,5), (2,3)],
                                [(0,5), (1,2), (3,4)],
                                [(0,5), (1,3), (2,4)],
                                [(0,5), (1,4), (2,3)]
                            ]
                            
                            for pairs_idx in pairings_indices:
                                p1 = (pts[pairs_idx[0][0]], pts[pairs_idx[0][1]])
                                p2 = (pts[pairs_idx[1][0]], pts[pairs_idx[1][1]])
                                p3 = (pts[pairs_idx[2][0]], pts[pairs_idx[2][1]])
                                
                                trs = get_transversals(p1, p2, p3)
                                # Sort each triple
                                trs = [tuple(sorted(t)) for t in trs]
                                # Get indices
                                idxs = [triple_index[t] for t in trs]
                                
                                # Add edges between all pairs in idxs
                                # 8 nodes, 28 edges
                                for u_idx in range(8):
                                    for v_idx in range(u_idx+1, 8):
                                        u = idxs[u_idx]
                                        v = idxs[v_idx]
                                        conflict_mask[u] |= (1 << v)
                                        conflict_mask[v] |= (1 << u)

    # 3. Branch and Bound for Maximum Independent Set
    # We want to find the largest set of nodes with no edges between them.
    # This is equivalent to the Maximum Independent Set problem.
    
    # Heuristic for upper bound:
    # Given a set of candidate nodes C, the maximum independent set size is at most
    # the size of the minimum clique cover of the induced subgraph on C? No, that's hard.
    # A common simple bound: 
    # If we color the graph, the number of colors is an upper bound on the size of the independent set?
    # No, the number of colors is an upper bound on the size of the *clique*? No.
    # The chromatic number is an upper bound on the size of the independent set? No.
    # Actually, if we partition the nodes into cliques, the size of the independent set is at most the number of cliques.
    # Greedy coloring gives an upper bound on the size of the independent set?
    # Let's use the "Greedy Coloring" bound often used in MIS solvers.
    # Color the graph greedily. The color number of a node is an upper bound on the size of the MIS containing that node?
    # Specifically, if we order nodes 1..k, and color them greedily, the color assigned to node i is an upper bound on the size of the independent set that can be formed from nodes i..k if we include node i?
    # Actually, the standard bound is: 
    # Let c(i) be the color of node i in a greedy coloring of the subgraph induced by nodes i..N.
    # Then the size of any independent set in nodes i..N is at most c(i).
    # Wait, if we include node i, we can pick at most 1 node from each color class?
    # No, nodes of the same color are NOT adjacent? In a proper coloring, nodes of the same color are NOT adjacent.
    # So an independent set can contain multiple nodes of the same color.
    # The bound is: The size of the MIS is at most the number of colors used?
    # No, that's the clique number lower bound.
    
    # Correct Bound for MIS:
    # Greedy coloring of the *complement* graph? No.
    # Let's use the bound: 
    # For a set of candidates S, let G_S be the induced subgraph.
    # We want max independent set in G_S.
    # A simple bound is |S|.
    # A tighter bound: 
    # If we greedily color G_S, let k be the number of colors.
    # This does NOT bound the MIS size. (e.g. bipartite graph, 2 colors, but MIS can be large).
    
    # However, if we greedily color the *conflict* graph, and we pick a node, we must remove its neighbors.
    # The "Measure and Conquer" or "Branch and Reduce" approaches use specific bounds.
    
    # For a small graph (560 nodes), a simple recursive search with good pruning might work.
    # Pruning:
    # 1. If current_size + remaining_nodes <= best_size, prune.
    # 2. If we can find an upper bound on the MIS size of the remaining candidates that is <= best_size - current_size, prune.
    
    # Upper Bound Heuristic:
    # A common one is the "Greedy Clique Cover" on the conflict graph? No.
    # Let's use the following:
    # The size of the MIS is at most the size of the smallest clique cover of the graph? No.
    # Actually, the MIS size is at most the number of cliques in a clique cover.
    # Finding a small clique cover is hard.
    
    # Alternative: 
    # Use the degree. If a node has high degree, including it removes many candidates.
    # If a node has low degree, it's "safe" to include?
    
    # Let's try a simple Branch and Bound with the bound:
    # UB = current_size + len(remaining_candidates)
    # This is weak.
    
    # Better UB:
    # We can use the "Greedy Coloring" of the *induced subgraph* of candidates.
    # Wait, I recall now. 
    # In the context of Maximum Independent Set, a common upper bound is derived from coloring the graph.
    # If we color the graph with k colors, the graph is the union of k independent sets.
    # This doesn't help bound the size of ONE independent set.
    
    # However, if we look at the *complement* graph, an independent set in G is a clique in G_complement.
    # So we are looking for the Maximum Clique in the complement graph.
    # For Maximum Clique, a common upper bound is:
    # Color the complement graph greedily. The color number of a node is an upper bound on the size of the clique that can be extended from that node.
    # Why? Because in a proper coloring of a graph, any clique can contain at most one vertex of each color.
    # So if we color G_complement, the color of a vertex v is an upper bound on the size of any clique in G_complement that contains v.
    # Thus, the size of the MIS in G is at most the color number of the candidates in G_complement.
    
    # So:
    # 1. Build G_conflict (560 nodes).
    # 2. We want MIS in G_conflict.
    # 3. This is Max Clique in G_complement.
    # 4. G_complement edge (u,v) exists if u and v do NOT conflict.
    # 5. We can run a Max Clique solver on G_complement.
    
    # Let's implement a Max Clique solver with greedy coloring bound.
    
    # Nodes: 0..559
    # G_comp_mask[i] is a bitmask of nodes j such that i and j are NOT in conflict.
    # Note: i is not in conflict with itself, but we don't add self-loops.
    
    # Construct G_comp_mask
    # conflict_mask[i] has bits set for neighbors in G_conflict.
    # G_comp_mask[i] should have bits set for all j != i such that bit j is NOT in conflict_mask[i].
    
    # Full mask for 560 nodes
    full_mask = (1 << 560) - 1
    
    # For each i, G_comp_mask[i] = (full_mask ^ (1 << i)) & ~conflict_mask[i]
    
    g_comp_mask = [0] * n_triples
    for i in range(n_triples):
        # Exclude self
        g_comp_mask[i] = (full_mask ^ (1 << i)) & ~conflict_mask[i]
        
    # Max Clique Algorithm (Tomita et al.)
    # Recursive function expand(C, P)
    # C: current clique (list of nodes)
    # P: candidate set (bitmask of nodes that can be added to C)
    
    best_clique_size = 0
    best_clique_nodes = []
    
    # Greedy Coloring of a candidate set P
    # Returns a list of colors for each node in P, and the max color used.
    # We also need to order the nodes by color for the branching.
    
    def color_graph(P_mask):
        # Returns (colors_dict, max_color)
        # colors_dict[node] = color
        # We want to color the subgraph induced by P_mask in G_comp.
        # Note: In G_comp, an edge means "compatible". 
        # Wait. The coloring bound for Max Clique applies to the graph where we are looking for cliques.
        # We are looking for cliques in G_comp.
        # So we color G_comp.
        # Proper coloring of G_comp: Adjacent nodes in G_comp (compatible triples) must have different colors?
        # No. A proper coloring assigns different colors to adjacent nodes.
        # In G_comp, adjacent nodes are compatible.
        # So a proper coloring of G_comp groups incompatible nodes? No.
        # If u and v are compatible (edge in G_comp), they must have different colors.
        # This means nodes of the same color are INCOMPATIBLE (no edge in G_comp, so edge in G_conflict).
        # So a color class is a set of mutually conflicting nodes (a clique in G_conflict).
        # An independent set in G_conflict is a set of mutually compatible nodes (a clique in G_comp).
        # A clique in G_comp can contain at most one node from each color class (since nodes in the same color class are not connected in G_comp, i.e., they are not a clique in G_comp? 
        # Wait. If u and v have the same color, they are NOT adjacent in G_comp. 
        # So they cannot both be in a clique of G_comp.
        # Yes. So the number of colors is an upper bound on the size of the clique in G_comp.
        
        nodes = [i for i in range(560) if (P_mask >> i) & 1]
        if not nodes:
            return {}, 0
        
        # Greedy coloring
        # Order nodes by degree in G_comp? Or just iterate.
        # To get a tight bound, we should order nodes such that high-degree nodes get low colors?
        # Actually, for the bound to be tight, we want to minimize the number of colors? 
        # No, the bound is "size <= max_color". So we want a coloring with small max_color.
        # Greedy coloring with a good ordering helps.
        
        # Let's use a simple ordering: by index.
        # Or by degree in G_comp (higher degree first might use fewer colors? Or more? 
        # High degree nodes have more constraints, so they might force new colors. 
        # Usually, ordering by descending degree is good for coloring).
        
        # Calculate degrees in G_comp for nodes in P
        degrees = {}
        for u in nodes:
            # Count neighbors in P
            neighbors = g_comp_mask[u] & P_mask
            # Count bits
            degrees[u] = bin(neighbors).count('1')
            
        # Sort nodes by degree descending
        nodes_sorted = sorted(nodes, key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        max_color = 0
        # color_classes: list of sets of nodes
        color_classes = []
        
        for u in nodes_sorted:
            # Find the smallest color c such that u is not adjacent to any node with color c
            # In G_comp, u is adjacent to v if they are compatible.
            # We need to assign a color to u such that no neighbor of u in G_comp has that color.
            # So we check which colors are "blocked" by neighbors of u.
            
            blocked_colors = set()
            for v in nodes:
                if (g_comp_mask[u] >> v) & 1:
                    if v in colors:
                        blocked_colors.add(colors[v])
            
            # Find smallest non-blocked color
            c = 0
            while c in blocked_colors:
                c += 1
            
            colors[u] = c
            if c > max_color:
                max_color = c
            if c >= len(color_classes):
                color_classes.append([u])
            else:
                color_classes[c].append(u)
                
        return colors, max_color, color_classes

    # Recursive search
    # P is a bitmask of candidates
    # C is the current clique size
    # We also need to track the nodes for the solution
    
    # To speed up, we can pass the nodes as a list and maintain the mask.
    # But bitmasks are efficient.
    
    # We need to be careful with recursion depth. 560 is fine.
    
    # Optimization: If C + max_color <= best_clique_size, prune.
    
    # We need to order the nodes in P by their color (descending) to explore promising branches first.
    
    def expand(P_mask, C_size, C_nodes):
        nonlocal best_clique_size, best_clique_nodes
        
        if time.time() > deadline - 5:
            return
            
        if C_size + bin(P_mask).count('1') <= best_clique_size:
            return
            
        if P_mask == 0:
            if C_size > best_clique_size:
                best_clique_size = C_size
                best_clique_nodes = C_nodes[:]
            return
            
        # Get colors and max color
        colors, max_color, color_classes = color_graph(P_mask)
        
        if C_size + max_color <= best_clique_size:
            return
            
        # Get nodes in P ordered by color descending
        # We need to iterate through nodes in P, sorted by color.
        # Nodes with higher colors have tighter bounds?
        # Actually, we should pick nodes with higher colors first?
        # The bound for a node u is colors[u].
        # If we pick u, the new bound is C_size + 1 + max_color_of_remaining.
        
        # Let's extract nodes from P_mask
        nodes_in_P = [i for i in range(560) if (P_mask >> i) & 1]
        
        # Sort by color descending
        nodes_in_P.sort(key=lambda x: colors[x], reverse=True)
        
        for u in nodes_in_P:
            # Prune if C_size + 1 + colors[u] <= best_clique_size
            # Note: colors[u] is the max color in the subgraph induced by {u} U (candidates compatible with u)?
            # No, colors[u] was computed on the whole P.
            # When we add u to the clique, the new candidates are neighbors of u in G_comp that are in P.
            # The bound should be recalculated or approximated.
            # A safe bound: C_size + 1 + max_color_remaining.
            # But we don't have max_color_remaining easily.
            # However, it is known that if we process nodes in descending color order,
            # the condition C_size + colors[u] <= best_clique_size allows pruning?
            # Let's check.
            # If we include u, we can pick at most 1 node from each color class in the remaining candidates.
            # The remaining candidates are a subset of P.
            # The max color in the remaining candidates is <= max_color.
            # Actually, a tighter bound is used in Tomita's algorithm:
            # The nodes are colored such that the color of u is an upper bound on the clique size extending from u.
            # So if C_size + colors[u] <= best_clique_size, we can skip u?
            # Yes, this is a standard pruning rule.
            
            if C_size + colors[u] <= best_clique_size:
                continue
                
            # Add u to clique
            new_C_nodes = C_nodes + [u]
            
            # New candidates: neighbors of u in G_comp that are in P
            new_P_mask = P_mask & g_comp_mask[u]
            
            # Remove u from new_P_mask (though it's not a neighbor of itself)
            # Actually, u is not in g_comp_mask[u], so it's fine.
            # But we should ensure we don't pick u again.
            
            expand(new_P_mask, C_size + 1, new_C_nodes)
            
            # Remove u from P for the next iteration (branching on excluding u)
            P_mask = P_mask & ~(1 << u)
            
            # Update bound for the remaining loop?
            # The colors might change, but we are using the static colors from the start of this expand call.
            # This is valid for pruning.

    # Initial call
    # P is all nodes
    initial_P = full_mask
    # We should sort initial nodes? The algorithm handles ordering inside expand.
    
    # To improve performance, we can provide an initial solution from a greedy heuristic.
    # Greedy MIS:
    # Pick node with highest degree? Or lowest?
    # In G_conflict, we want independent set.
    # In G_comp, we want clique.
    # Greedy Clique: Pick node with highest degree in G_comp, add it, remove non-neighbors.
    
    def greedy_clique():
        P = full_mask
        clique = []
        while P:
            # Pick node with max degree in P
            best_node = -1
            max_deg = -1
            for i in range(560):
                if (P >> i) & 1:
                    deg = bin(g_comp_mask[i] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        best_node = i
            if best_node == -1:
                break
            clique.append(best_node)
            # Update P: only keep neighbors of best_node
            P = P & g_comp_mask[best_node]
            P = P & ~(1 << best_node) # Remove self
        return clique
        
    init_clique = greedy_clique()
    best_clique_size = len(init_clique)
    best_clique_nodes = init_clique[:]
    
    # Run Branch and Bound
    expand(full_mask, 0, [])
    
    # Convert best clique (indices) to triples
    if best_clique_nodes:
        result = [list(all_triples[i]) for i in best_clique_nodes]
    else:
        # Fallback
        result = [[0, 1, 2]]
        
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
