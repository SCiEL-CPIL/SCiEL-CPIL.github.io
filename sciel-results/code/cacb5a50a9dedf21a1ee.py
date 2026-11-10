# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Bitset-based Branch and Bound with Greedy Coloring for Maximum Independent Set.
    """
    m = 205
    
    # 1. Precompute forbidden differences and adjacency
    forbidden = set(pow(x, 2, m) for x in range(1, m))
    forbidden.discard(0)
    
    # Build conflict mask: conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # 2. Branch and Bound with Greedy Coloring Upper Bound
    # We want Maximum Independent Set in the conflict graph.
    # Equivalently, Maximum Clique in the complement graph.
    # 
    # B&B approach:
    # - candidates: bitmask of vertices still available
    # - current: bitmask of vertices in current solution
    # - At each step, pick a vertex v from candidates
    #   Branch 1: Include v -> remove v and N(v) from candidates, add v to current
    #   Branch 2: Exclude v -> remove v from candidates
    # - Upper bound: |current| + greedy_color_bound(candidates)
    #   where greedy_color_bound computes chromatic number of conflict graph on candidates
    #   (which upper bounds the independence number)
    
    def greedy_color_bound(candidates_mask):
        """
        Returns an upper bound on the maximum independent set size in the subgraph
        induced by candidates_mask.
        
        Uses greedy coloring of the conflict graph restricted to candidates.
        The number of colors used is an upper bound on the independence number
        (since each color class is an independent set, and we need at least
        alpha colors to cover all vertices with independent sets... wait, that's wrong).
        
        Actually: The chromatic number chi(G) >= alpha(G) is NOT true in general.
        What's true: alpha(G) <= n, and chi(G) * alpha(G) >= n.
        So alpha(G) <= n / chi(G)? No, that's also not right.
        
        Let me reconsider. For a graph G:
        - alpha(G) = maximum independent set size
        - chi(G) = minimum number of colors (partition into independent sets)
        
        We have: chi(G) * alpha(G) >= n (since each color class has size at most alpha)
        So: alpha(G) <= n / chi(G) is FALSE.
        Actually: n <= chi(G) * alpha(G), so alpha(G) >= n/chi(G). That's a lower bound.
        
        For an UPPER bound on alpha(G):
        - Trivial: alpha(G) <= n
        - Better: Use the fact that in any graph, alpha(G) <= n - delta(G) where delta is min degree? No.
        
        Standard B&B for MIS uses:
        - Color the COMPLEMENT graph greedily
        - The number of colors in the complement = clique number of complement = independence number of original
        - Wait, we want to bound alpha(G) from above.
        
        Let me use a different approach. A common upper bound for MIS in B&B:
        Process vertices in some order. For each vertex, if it can't be added to the
        current independent set, skip it. The greedy construction gives a LOWER bound.
        
        For an UPPER bound, one technique is:
        - Order vertices
        - For each vertex v in order, compute how many of the remaining vertices
          can coexist with v. This gives a bound.
        
        Actually, the standard "color bound" for MIS B&B works as follows:
        - We have a set of candidate vertices
        - We greedily color the INDUCED SUBGRAPH on candidates using the CONFLICT relation
        - If we use k colors, then the maximum independent set in this subgraph has size at most k
          (because in a proper coloring, no two adjacent vertices share a color, so an independent
          set can have at most one vertex per color... wait, that's for the clique problem).
        
        I'm confusing myself. Let me be very clear:
        
        Problem: Maximum Independent Set in graph G = (V, E)
        - Independent set: subset of V with no edges between any pair
        - We want to maximize |S| where S is independent
        
        B&B with coloring bound:
        - We maintain a set C of candidate vertices
        - We want to bound alpha(G[C]) = max independent set in G[C]
        - Greedy coloring of G[C]: assign colors such that adjacent vertices get different colors
        - If we use k colors, then G[C] can be partitioned into k independent sets
        - Therefore, alpha(G[C]) <= k? NO! That's wrong.
        - Actually: if G[C] has a proper coloring with k colors, then each color class is an
          independent set, but alpha(G[C]) could be larger than k.
        
        I think I've been confusing this with the CLIQUE problem. Let me restart.
        
        For Maximum Clique in graph H:
        - Greedy coloring of H gives an upper bound on the clique number
        - Because a clique must have all distinct colors in any proper coloring
        
        For Maximum Independent Set in graph G:
        - This is Maximum Clique in the complement graph G'
        - So we should greedily color G' (the complement of G)
        - The number of colors used is an upper bound on the clique number of G'
        - Which equals the independence number of G
        
        So: to bound alpha(G[C]), we should color the COMPLEMENT of G[C] greedily.
        The complement of G[C] has an edge between u,v if u,v are NOT adjacent in G.
        
        Let me implement this properly.
        """
        if candidates_mask == 0:
            return 0
        
        # Get list of vertices in candidates
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
        
        n_cand = len(verts)
        if n_cand == 0:
            return 0
        
        # Greedy coloring of the COMPLEMENT graph on verts
        # In the complement, u and v are adjacent if they are NOT adjacent in the original conflict graph
        # i.e., if (conflict_mask[u] & (1 << v)) == 0 and u != v
        
        # For greedy coloring, we process vertices in some order and assign the smallest available color
        # We want to minimize the number of colors to get a tight upper bound
        
        # Order vertices by degree in the complement (descending) for better coloring
        # Degree in complement = (n_cand - 1) - degree_in_original
        
        deg_in_original = {}
        for v in verts:
            deg_in_original[v] = bin(candidates_mask & conflict_mask[v]).count('1')
        
        # Sort by complement degree descending (= n_cand - 1 - deg_in_original)
        verts_sorted = sorted(verts, key=lambda v: (n_cand - 1 - deg_in_original[v]), reverse=True)
        
        # Greedy coloring
        color_of = {}  # vertex -> color
        next_color = 0
        
        for v in verts_sorted:
            # Find colors used by neighbors in the complement
            # Neighbors in complement: u in verts, u != v, and (conflict_mask[v] & (1 << u)) == 0
            used_colors = set()
            for u in verts:
                if u == v:
                    continue
                if (candidates_mask & (1 << u)) == 0:
                    continue
                # u is in candidates and u != v
                # Are v and u adjacent in the complement?
                # Yes if they are NOT adjacent in the original conflict graph
                if (conflict_mask[v] & (1 << u)) == 0:
                    if u in color_of:
                        used_colors.add(color_of[u])
            
            # Assign the smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        
        return next_color
    
    def branch_and_bound(candidates_mask, current_mask, current_size, best_size, best_mask, deadline):
        """
        Recursive B&B for Maximum Independent Set.
        Returns (best_size, best_mask)
        """
        # Check deadline
        if time.time() > deadline:
            return best_size, best_mask
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return best_size, best_mask
        
        # Upper bound
        ub = current_size + greedy_color_bound(candidates_mask)
        if ub <= best_size:
            return best_size, best_mask
        
        # Get vertices in candidates
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
        
        # Order vertices by degree in candidates (descending) - high degree first
        # Vertices with high degree are more "constraining"
        def degree_in_cand(v):
            return bin(candidates_mask & conflict_mask[v]).count('1')
        
        verts_sorted = sorted(verts, key=degree_in_cand, reverse=True)
        
        # Try including the first vertex
        v = verts_sorted[0]
        
        # Branch 1: Include v
        new_current_mask = current_mask | (1 << v)
        new_current_size = current_size + 1
        # Remove v and its neighbors from candidates
        new_candidates = candidates_mask & ~(1 << v) & ~conflict_mask[v]
        
        best_size, best_mask = branch_and_bound(
            new_candidates, new_current_mask, new_current_size,
            best_size, best_mask, deadline
        )
        
        # Branch 2: Exclude v
        new_candidates2 = candidates_mask & ~(1 << v)
        
        best_size, best_mask = branch_and_bound(
            new_candidates2, current_mask, current_size,
            best_size, best_mask, deadline
        )
        
        return best_size, best_mask
    
    # 3. Initial greedy solution
    def greedy_init():
        """Greedy construction of an independent set."""
        curr_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for v in candidates:
            if (curr_mask & conflict_mask[v]) == 0:
                curr_mask |= (1 << v)
        
        return curr_mask
    
    # 4. Local search to improve initial solution
    def local_search(mask):
        """Improve mask via 1-swaps and additions."""
        curr_mask = mask
        improved = True
        max_iters = 500
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try to add a vertex
            for v in range(m):
                if not (curr_mask & (1 << v)):
                    if (curr_mask & conflict_mask[v]) == 0:
                        curr_mask |= (1 << v)
                        improved = True
                        break
                        
            if improved:
                continue
            
            # Try 1-swap
            set_bits = []
            temp = curr_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                set_bits.append(idx)
                temp ^= lsb
            
            random.shuffle(set_bits)
            
            for u in set_bits:
                new_mask_without_u = curr_mask & ~(1 << u)
                for v in range(m):
                    if not (new_mask_without_u & (1 << v)):
                        if (new_mask_without_u & conflict_mask[v]) == 0:
                            curr_mask = new_mask_without_u | (1 << v)
                            improved = True
                            break
                if improved:
                    break
        
        return curr_mask
    
    # Main execution
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Phase 1: Get initial solution via greedy + local search
    best_mask = greedy_init()
    best_mask = local_search(best_mask)
    best_size = bin(best_mask).count('1')
    
    # Phase 2: B&B to find optimal or near-optimal solution
    # Start with all vertices as candidates, empty current set
    # But we can use the current best as a lower bound for pruning
    
    # To make B&B more efficient, we can start from the greedy solution
    # and try to improve it. But standard B&B starts from scratch.
    
    # Let's try B&B with the greedy solution as initial best
    full_mask = (1 << m) - 1
    
    # B&B from scratch with the greedy solution as initial lower bound
    best_size_bb, best_mask_bb = branch_and_bound(full_mask, 0, 0, best_size, best_mask, deadline)
    
    if best_size_bb > best_size:
        best_size = best_size_bb
        best_mask = best_mask_bb
    
    # Convert best_mask to array
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
    
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
