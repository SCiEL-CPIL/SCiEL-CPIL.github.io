# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    This implementation uses an optimized branch-and-bound algorithm (Tomita-style) to find the
    Maximum Independent Set (MIS) of the conflict graph. The conflict graph has vertices
    0..204, with an edge between u and v if (u-v) % 205 is a forbidden square.
    We find the MIS by finding the Maximum Clique in the complement graph.
    """
    m = 205
    deadline = time.time() + 580  # Safety margin for 600s limit

    # 1. Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # 2. Build adjacency masks for the conflict graph G
    # adj_mask[u] has bit v set if (u-v) % m is in forbidden
    adj_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v and (u - v) % m in forbidden:
                mask |= (1 << v)
        adj_mask[u] = mask

    # 3. Build adjacency masks for the complement graph H
    # comp_mask[u] has bit v set if u and v are NOT adjacent in G (and u != v)
    # This is the graph in which we seek the Maximum Clique.
    full_mask = (1 << m) - 1
    comp_mask = [0] * m
    for u in range(m):
        # Start with all nodes, remove self and neighbors in G
        comp_mask[u] = full_mask & ~(1 << u) & ~adj_mask[u]

    # 4. Tomita Algorithm for Maximum Clique
    best_clique_mask = 0
    best_size = 0

    # Helper to popcount efficiently
    def popcount(x):
        return x.bit_count()

    # Precompute list of bits for quick iteration? 
    # Since m=205, we can just use bit operations.

    def greedy_color(cand_mask):
        """
        Performs greedy coloring on the induced subgraph of H defined by cand_mask.
        Returns a list of nodes ordered by decreasing color, and a dictionary/list of colors.
        We return nodes sorted by color descending, and the color of each node.
        """
        # Extract nodes
        cand_nodes = []
        temp = cand_mask
        while temp:
            lsb = temp & -temp
            node = lsb.bit_length() - 1
            cand_nodes.append(node)
            temp ^= lsb
        
        if not cand_nodes:
            return [], {}

        # Order by degree in H induced by cand_mask (descending) to help coloring
        # deg(v) = popcount(comp_mask[v] & cand_mask)
        # We compute degrees only for nodes in cand_nodes
        degs = {v: popcount(comp_mask[v] & cand_mask) for v in cand_nodes}
        cand_nodes.sort(key=lambda x: degs[x], reverse=True)

        colors = {}
        # To speed up color assignment, we can maintain a set of used colors for each node's neighbors
        # But since N is small, simple check is fine.
        
        # Standard greedy: assign smallest available color
        for v in cand_nodes:
            # Find colors used by neighbors of v in H that are in cand_nodes
            neigh_mask = comp_mask[v] & cand_mask
            used_colors = set()
            
            # Iterate through neighbors
            temp_neigh = neigh_mask
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                u = lsb.bit_length() - 1
                temp_neigh ^= lsb
                if u in colors:
                    used_colors.add(colors[u])
            
            # Assign smallest available color
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c

        # Sort nodes by color descending
        # This allows us to prune: if current_size + colors[v] <= best, we can stop
        # because all subsequent nodes have color <= colors[v]
        sorted_nodes = sorted(cand_nodes, key=lambda x: colors[x], reverse=True)
        
        return sorted_nodes, colors

    def expand(clique_mask, cand_mask):
        nonlocal best_clique_mask, best_size

        # Time check
        if time.time() > deadline:
            return

        # Pruning: if current clique size + max possible (cand size) <= best, return
        # Note: popcount(cand_mask) is a loose upper bound. Coloring provides a tighter one.
        if popcount(clique_mask) + popcount(cand_mask) <= best_size:
            return

        if cand_mask == 0:
            if popcount(clique_mask) > best_size:
                best_size = popcount(clique_mask)
                best_clique_mask = clique_mask
            return

        # Greedy Coloring
        sorted_nodes, colors = greedy_color(cand_mask)

        # Branch on nodes in reverse order of their color
        for v in sorted_nodes:
            # Pruning based on color bound
            # If we pick v, the max size we can add is bounded by colors[v]
            # (actually, it's bounded by the size of the clique we can form in the subgraph
            # induced by v and its neighbors with color <= colors[v] in the specific ordering,
            # but the standard Tomita bound is simply that the clique size from this branch
            # cannot exceed current_size + colors[v])
            
            if popcount(clique_mask) + colors[v] <= best_size:
                # Since nodes are sorted by color descending, all subsequent nodes
                # have color <= colors[v], so we can break
                break
            
            # Branch on v
            new_clique_mask = clique_mask | (1 << v)
            
            # New candidates are neighbors of v in H that are in the current cand_mask
            # AND have not been processed yet?
            # In Tomita, we typically restrict to neighbors that are "after" v in the ordering
            # to avoid duplicates? No, the standard recursive step is:
            # expand(clique + v, cand & N(v))
            # However, to maintain the coloring bound's validity and avoid re-exploring,
            # we should intersect with the set of nodes that were "available" when we started
            # this level, but specifically, we should only consider nodes that are adjacent to v.
            
            # The key optimization in Tomita is that the new candidate set is:
            # new_cand_mask = cand_mask & comp_mask[v]
            # But we must ensure we don't pick nodes that are "before" v in the specific
            # ordering used for the coloring bound? 
            # Actually, the bound `colors[v]` is an upper bound on the clique size in the
            # subgraph induced by {v} U {u in cand_nodes | u adjacent to v and color[u] <= colors[v]}.
            # To be safe and correct, we just use `cand_mask & comp_mask[v]`.
            # The pruning `break` handles the efficiency.
            
            new_cand_mask = cand_mask & comp_mask[v]
            
            expand(new_clique_mask, new_cand_mask)
            
            # Remove v from cand_mask for subsequent iterations
            # This ensures that when we consider the next node w, we are exploring
            # the branch where v is NOT in the clique.
            cand_mask &= ~(1 << v)

    # Initial call
    # Candidates are all nodes 0..204
    init_cand_mask = full_mask
    expand(0, init_cand_mask)

    # Extract the best clique
    result = []
    temp = best_clique_mask
    while temp:
        lsb = temp & -temp
        node = lsb.bit_length() - 1
        result.append(node)
        temp ^= lsb

    # If no clique found (should not happen, at least size 1), return [0]
    if not result:
        result = [0]

    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
