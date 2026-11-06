import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 85
    
    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # Build the "compatibility" graph: edge (i,j) exists if lcm(i,j) > 256
    # We want a maximum weight clique in this graph (weight = 1/value).
    
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(1, v) for v in nodes]
    
    # Build adjacency as bitsets (Python ints) for fast operations
    # adj[i] is a bitmask where bit j is set if i and j are compatible
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            if a * b > 256 * gcd(a, b):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy coloring for upper bound in branch and bound
    # We color the candidate subgraph to get an upper bound on clique size (weighted)
    
    def greedy_color_weighted(candidates_mask, order):
        """
        Color the subgraph induced by candidates using greedy coloring.
        Returns a list of color classes (each a list of node indices),
        where each class is an independent set in the compatibility graph
        (i.e., a set of mutually incompatible nodes).
        The number of colors is an upper bound on clique size.
        For weighted version, we return the sum of max weights per color class
        as an upper bound on the weighted clique.
        """
        # Order candidates for coloring
        uncolored = [i for i in order if (candidates_mask >> i) & 1]
        color_classes = []  # list of sets
        color_of = {}
        
        for v in uncolored:
            # Find the first color class where v doesn't conflict with any member
            # In the compatibility graph, two nodes are "compatible" if there's an edge.
            # For coloring in the context of maximum clique, we color the complement,
            # but here we're working directly with the compatibility graph.
            # Actually, for maximum clique, we use coloring on the subgraph of candidates.
            # Two candidates can be in the same color class if they are NOT compatible
            # (i.e., no edge in compatibility graph between them).
            placed = False
            for ci, cls in enumerate(color_classes):
                # Check if v is compatible with any node in cls
                # If v is compatible with some node in cls, v cannot be in this class
                # (because in a clique, all pairs must be compatible, so a color class
                #  in the clique coloring must be an independent set in the compatibility graph)
                can_place = True
                for u in cls:
                    if (adj[v] >> u) & 1:
                        can_place = False
                        break
                if can_place:
                    cls.add(v)
                    color_of[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append({v})
                color_of[v] = len(color_classes) - 1
        
        # Upper bound: sum over color classes of max weight in each class
        ub = Fraction(0)
        for cls in color_classes:
            max_w = max(weights[v] for v in cls)
            ub += max_w
        
        return ub, color_classes
    
    # Branch and bound for maximum weight clique
    best_clique = []
    best_weight = Fraction(0)
    
    def branch_and_bound(candidates_mask, current_clique, current_weight):
        nonlocal best_clique, best_weight
        
        if time.time() > deadline - 2:
            return
        
        # If no candidates, update best
        if candidates_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(current_clique)
            return
        
        # Upper bound: current weight + sum of all remaining weights (loose)
        # Better: use greedy coloring bound
        candidate_list = [i for i in range(n) if (candidates_mask >> i) & 1]
        
        # Quick upper bound: current_weight + sum of weights of all candidates
        quick_ub = current_weight + sum(weights[i] for i in candidate_list)
        if quick_ub <= best_weight:
            return
        
        # Greedy coloring bound
        # Order candidates by weight descending for better bounds
        candidate_list.sort(key=lambda i: -weights[i].numerator / weights[i].denominator)
        
        ub, _ = greedy_color_weighted(candidates_mask, candidate_list)
        if current_weight + ub <= best_weight:
            return
        
        # Branch on the first candidate in the ordering
        for idx in candidate_list:
            if time.time() > deadline - 2:
                return
            
            # Try including idx
            new_clique = current_clique + [idx]
            new_weight = current_weight + weights[idx]
            
            # New candidates must be compatible with idx and in current candidates
            new_candidates = candidates_mask & adj[idx]
            # Remove idx itself
            new_candidates &= ~(1 << idx)
            
            branch_and_bound(new_candidates, new_clique, new_weight)
            
            # Remove idx from candidates for the next iteration
            candidates_mask &= ~(1 << idx)
            
            # Re-check bound after removal
            if candidates_mask == 0:
                break
            
            candidate_list_remaining = [i for i in candidate_list if i != idx and (candidates_mask >> i) & 1]
            if not candidate_list_remaining:
                break
            
            candidate_list_remaining.sort(key=lambda i: -weights[i].numerator / weights[i].denominator)
            ub2, _ = greedy_color_weighted(candidates_mask, candidate_list_remaining)
            if current_weight + ub2 <= best_weight:
                break
    
    # Initial greedy solution to set a good lower bound
    # Greedy: repeatedly pick the node with highest weight that is compatible with all chosen
    greedy_set = []
    remaining = set(range(n))
    while remaining:
        # Pick the node with highest weight among remaining that is compatible with all in greedy_set
        best_node = None
        best_w = Fraction(-1)
        for i in sorted(remaining, key=lambda x: -weights[x].numerator / weights[x].denominator):
            compatible = True
            for g in greedy_set:
                if not ((adj[i] >> g) & 1):
                    compatible = False
                    break
            if compatible:
                if weights[i] > best_w:
                    best_w = weights[i]
                    best_node = i
                break  # Since sorted by weight, first compatible is best
        if best_node is None:
            break
        greedy_set.append(best_node)
        remaining.discard(best_node)
        # Remove all nodes incompatible with best_node from remaining
        to_remove = set()
        for r in list(remaining):
            if not ((adj[best_node] >> r) & 1):
                to_remove.add(r)
        remaining -= to_remove
    
    if greedy_set:
        best_weight = sum(weights[i] for i in greedy_set)
        best_clique = list(greedy_set)
    
    # Run branch and bound
    full_mask = (1 << n) - 1
    branch_and_bound(full_mask, [], Fraction(0))
    
    # Convert back to values
    result = [nodes[i] for i in best_clique]
    
    # If empty for some reason, return a valid single element
    if not result:
        result = [2]
    
    return result
