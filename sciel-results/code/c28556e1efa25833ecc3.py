# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses an exact Maximum Independent Set solver (Tomita algorithm) with a fallback 
    to heuristic local search if the exact solver exceeds a safe time budget.
    """
    m = 205
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency list for the conflict graph
    # Node i conflicts with node j if (i-j)%m is in forbidden
    adj = [[] for _ in range(m)]
    for i in range(m):
        for d in forbidden:
            j = (i - d) % m
            if i != j:
                adj[i].append(j)
                # Ensure symmetry is handled, though we build full adjacency
            j2 = (i + d) % m
            if i != j2:
                adj[i].append(j2)
        
        # Remove duplicates
        adj[i] = list(set(adj[i]))
    
    # The Tomita algorithm for Maximum Independent Set
    # It operates by finding the maximum clique in the complement graph,
    # or directly via branch and bound on the independent set.
    # Here we implement a standard branch and bound for MIS.
    
    best_set = []
    best_size = 0
    
    # We use a recursive approach with a coloring bound.
    # For N=205, we can represent sets as bitmasks or lists. Lists are easier to debug, 
    # but bitmasks are faster. Given N=205, Python integers can hold the bitmask.
    
    # Precompute neighbor masks for faster intersection
    neighbor_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        neighbor_mask[i] = mask
    
    # We will search for the MIS.
    # State: current_set (list), available_nodes (mask of nodes that can be added)
    
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def greedy_coloring(nodes):
        """
        Greedy coloring of the subgraph induced by 'nodes'.
        Returns a list of colors and the max color used (bound).
        This is used for the upper bound in the branch and bound.
        """
        if not nodes:
            return 0, []
        
        colors = [0] * len(nodes)
        max_color = 0
        # Simple greedy: assign smallest available color
        # For better bound, we can use a more sophisticated coloring, 
        # but for N=205, even basic greedy works well with good branching order.
        
        # To improve the bound, we sort nodes by degree in the induced subgraph?
        # Or just iterate.
        
        for i, node in enumerate(nodes):
            # Find colors used by neighbors of 'node' that are already colored
            used_colors = set()
            for j in range(i):
                if colors[j] > 0: # Only consider colored nodes
                    # Check if nodes[j] is a neighbor of node
                    # We need to know if nodes[j] is adjacent to node in the original graph
                    # 'nodes' is a list of node indices.
                    if (neighbor_mask[node] >> nodes[j]) & 1:
                        used_colors.add(colors[j])
            
            # Find smallest color not in used_colors
            c = 1
            while c in used_colors:
                c += 1
            colors[i] = c
            if c > max_color:
                max_color = c
            
        return max_color, colors

    def branch_and_bound(current_set, available_mask, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 5:
            return
            
        # Get list of available nodes
        nodes = []
        temp_mask = available_mask
        while temp_mask:
            # Extract lowest set bit
            lsb = temp_mask & -temp_mask
            node = lsb.bit_length() - 1
            nodes.append(node)
            temp_mask ^= lsb
            
        if not nodes:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Bound: current_size + upper_bound_on_remaining
        # Upper bound is the clique number of the complement of the induced subgraph?
        # For MIS, the bound is the size of the maximum clique in the graph induced by available nodes?
        # No, for MIS, we want to bound how many MORE nodes we can add.
        # The maximum number of nodes we can add from 'available' is bounded by the size of a maximum clique in the graph G[available]?
        # Wait. If we select nodes for MIS, they must be independent.
        # So we are looking for an independent set in G[available].
        # The size of the maximum independent set in G[available] is at most the number of nodes in G[available].
        # A better bound comes from coloring the graph G[available].
        # If we color G[available] with k colors, then the MIS size is at most k?
        # No, that's for Clique.
        # For Independent Set, if we color the COMPLEMENT graph?
        # Actually, a common bound for MIS is based on the clique cover of the graph.
        # Or simply: Greedy coloring of the graph G[available] gives a lower bound on the chromatic number.
        # This doesn't directly bound MIS.
        
        # Let's use the standard Tomita bound for MIS:
        # The maximum size of an independent set in a graph is bounded by the number of colors in a greedy coloring of the graph?
        # No. 
        # Let's stick to a simpler, effective heuristic bound: 
        # Sort nodes by degree (descending) in the induced subgraph.
        # The bound is often just the number of remaining nodes, which is weak.
        
        # Alternative: Use the fact that we are doing this for N=205.
        # We can use a simpler branch and bound with a good heuristic ordering.
        # Order: Descending degree in the induced subgraph.
        
        # Calculate degrees in induced subgraph
        degrees = []
        for node in nodes:
            deg = 0
            for other in nodes:
                if node != other and (neighbor_mask[node] >> other) & 1:
                    deg += 1
            degrees.append(deg)
            
        # Sort nodes by degree descending
        sorted_indices = sorted(range(len(nodes)), key=lambda i: degrees[i], reverse=True)
        sorted_nodes = [nodes[i] for i in sorted_indices]
        
        # A simple bound: We can add at most len(sorted_nodes) nodes.
        # If len(current_set) + len(sorted_nodes) <= best_size, prune.
        if len(current_set) + len(sorted_nodes) <= best_size:
            return
            
        # Branch on the first node
        # Heuristic: Pick the node with highest degree.
        # If we pick a node, we must exclude its neighbors.
        # If we don't pick it, we remove it from available.
        
        # We can also use a coloring bound on the complement?
        # Let's try a simpler approach first: Recursive with memoization? No, state space too big.
        
        # Let's use the "Greedy Coloring Bound" properly.
        # For Maximum Clique, greedy coloring of the graph gives an upper bound on clique size.
        # For Maximum Independent Set, we can find Max Clique in the Complement Graph.
        # Let's switch perspective: Find Max Clique in Complement Graph G_comp.
        # Adjacency in G_comp: i ~ j if i NOT adjacent to j in G.
        
        # Let's restart the solver logic to use Max Clique on Complement.
        
        pass # Placeholder, will implement below

    # --- Max Clique Solver on Complement Graph ---
    
    # Build complement adjacency
    comp_adj_mask = [0] * m
    all_mask = (1 << m) - 1
    for i in range(m):
        # Neighbors in G are neighbor_mask[i]
        # Neighbors in Comp are all_mask ^ neighbor_mask[i] ^ (1 << i)
        comp_adj_mask[i] = all_mask ^ neighbor_mask[i] ^ (1 << i)
        
    def greedy_coloring_clique(nodes):
        """
        Greedy coloring for the subgraph induced by 'nodes' in the COMPLEMENT graph.
        Returns max color. This serves as an upper bound for the clique size.
        """
        if not nodes:
            return 0
        
        colors = {}
        max_color = 0
        # Order nodes by degree in complement graph (descending) to improve bound
        # Degree in comp graph = (len(nodes) - 1) - degree_in_G
        # We already have neighbor_mask for G.
        
        # To keep it fast, we just iterate.
        for node in nodes:
            used_colors = set()
            for other in nodes:
                if node == other:
                    continue
                if (comp_adj_mask[node] >> other) & 1:
                    if other in colors:
                        used_colors.add(colors[other])
            
            c = 1
            while c in used_colors:
                c += 1
            colors[node] = c
            if c > max_color:
                max_color = c
                
        return max_color

    def expand_clique(clique_set, available_mask):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 5:
            return
            
        nodes = []
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            node = lsb.bit_length() - 1
            nodes.append(node)
            temp_mask ^= lsb
            
        if not nodes:
            if len(clique_set) > best_size:
                best_size = len(clique_set)
                best_set = clique_set[:]
            return
            
        # Bound: clique_size + max_color <= best_size
        max_color = greedy_coloring_clique(nodes)
        if len(clique_set) + max_color <= best_size:
            return
            
        # Sort nodes by degree in the complement graph (descending)
        # Degree in comp graph for node v in induced subgraph:
        # Count neighbors in 'nodes'
        degrees = []
        for node in nodes:
            deg = 0
            for other in nodes:
                if node != other and (comp_adj_mask[node] >> other) & 1:
                    deg += 1
            degrees.append(deg)
            
        sorted_indices = sorted(range(len(nodes)), key=lambda i: degrees[i], reverse=True)
        sorted_nodes = [nodes[i] for i in sorted_indices]
        
        for node in sorted_nodes:
            # If we add 'node' to clique, we can only add its neighbors in Comp
            new_available = available_mask & comp_adj_mask[node]
            new_clique = clique_set + [node]
            expand_clique(new_clique, new_available)
            
            # Remove 'node' from available for the next branch (don't pick node)
            available_mask &= ~(1 << node)
            
            # Re-evaluate bound after removing node?
            # The bound calculation is expensive. We can just continue.
            # But we should check if the remaining nodes can still beat best_size
            # A quick check: if len(clique_set) + popcount(available_mask) <= best_size: break
            if len(clique_set) + bin(available_mask).count('1') <= best_size:
                break

    # Initialize
    start_mask = (1 << m) - 1
    expand_clique([], start_mask)
    
    if best_size > 0:
        return np.array(best_set, dtype=np.int64)
        
    # Fallback to heuristic if exact solver failed or found nothing (should not happen)
    # Use the heuristic from the current program
    def heuristic_fallback():
        forbidden_set = forbidden
        conflict_mask = [0] * m
        for i in range(m):
            mask = 0
            for d in forbidden_set:
                j = (i - d) % m
                mask |= (1 << j)
            conflict_mask[i] = mask
            
        def greedy_construct(order):
            selected_mask = 0
            selected_list = []
            for candidate in order:
                if (selected_mask & conflict_mask[candidate]) == 0:
                    selected_mask |= (1 << candidate)
                    selected_list.append(candidate)
            return selected_list, selected_mask

        def local_search_1move(selected_list, selected_mask):
            selected_list = list(selected_list)
            improved = True
            while improved:
                improved = False
                for a in range(m):
                    if (selected_mask & (1 << a)) == 0:
                        if (selected_mask & conflict_mask[a]) == 0:
                            selected_mask |= (1 << a)
                            selected_list.append(a)
                            improved = True
            return selected_list, selected_mask

        rng = random.Random(42)
        best_sol = []
        best_mask = 0
        for _ in range(20):
            order = list(range(m))
            rng.shuffle(order)
            sol, mask = greedy_construct(order)
            sol, mask = local_search_1move(sol, mask)
            if len(sol) > len(best_sol):
                best_sol = sol
                best_mask = mask
                
        return np.array(best_sol, dtype=np.int64)
        
    return heuristic_fallback()

# EVOLVE-BLOCK-END
