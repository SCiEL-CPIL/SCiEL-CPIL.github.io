"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill
    
    n = 20
    num_colors = 20
    vertices = list(range(n))
    
    # Precompute all potential 4-cycles
    # A 4-cycle is a set of 4 edges.
    # We represent a 4-cycle by its 4 vertices (a,b,c,d) and the specific pairing.
    # In K_n, for any 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles:
    # 1. (a,b), (b,c), (c,d), (d,a)
    # 2. (a,b), (b,d), (d,c), (c,a)
    # 3. (a,c), (c,b), (b,d), (d,a)
    
    # To check validity, we need to ensure that for every 4-cycle, the 4 edge colors are distinct.
    # Strategy: Build the graph incrementally.
    # When adding an edge (u,v) with color c, we must ensure that we don't complete a 4-cycle
    # where the other 3 edges already have colors that conflict with c or each other in a way
    # that makes the set of 4 not distinct.
    
    # Actually, a simpler sufficient condition for "rainbow 4-cycle" is:
    # If we ensure that no 4-cycle has two edges of the same color, is that enough?
    # No, we need ALL 4 to be distinct.
    # So, if we add edge e with color c, and there is a 4-cycle containing e,
    # the other 3 edges must have colors distinct from c and from each other.
    
    # Let's maintain:
    # adj[u][v] = color or -1
    # For fast checking, we can precompute for each edge (u,v), the list of "partner" edges
    # that form a 4-cycle with it.
    # If (u,v) and (x,y) are in a 4-cycle, then {u,v,x,y} are distinct.
    # The other two edges in the cycle depend on the pairing.
    
    # Let's define the 4-cycles explicitly by their 4 edges.
    # There are C(20,4) * 3 = 4845 * 3 = 14535 4-cycles.
    
    # We will use a greedy approach with randomization and restarts.
    # We try to add edges in a random order.
    # For each edge, we try colors in a random order (or specific heuristic).
    # We check if the assignment is valid with respect to all 4-cycles that are "almost complete".
    # A 4-cycle is "almost complete" if 3 of its edges are assigned.
    # If we assign the 4th, we check if the 4 colors are distinct.
    # If 2 or 3 are assigned, we just check if the new color conflicts with existing ones in that cycle.
    
    # Optimization:
    # For each edge e, precompute the list of 4-cycles it belongs to.
    # When assigning color c to e:
    #   For each cycle C containing e:
    #     Let others = C \ {e}
    #     Let assigned_others = [adj[u][v] for (u,v) in others if adj[u][v] != -1]
    #     If c in assigned_others: Conflict!
    #     If len(assigned_others) == 3:
    #        # The cycle is now complete.
    #        # Check if all 4 are distinct.
    #        # Since c not in assigned_others, we just need assigned_others to be distinct.
    #        if len(set(assigned_others)) != 3: Conflict! (This means the other 3 already had a duplicate)
    #        # Wait, if the other 3 had a duplicate, the cycle was already invalid?
    #        # No, if the other 3 had a duplicate, say c1, c1, c2, then the cycle is not rainbow.
    #        # But we should have prevented that when the 3rd edge was added.
    #        # So if we maintain invariant "no partial cycle has duplicate colors",
    #        # then when we add the 4th, we just need c != any of the 3.
    #        # And since the 3 are distinct (by invariant), the 4 will be distinct.
    
    # So the invariant is: **No 4-cycle contains two edges of the same color.**
    # If we maintain this invariant, then any completed 4-cycle will have 4 distinct colors?
    # Let's check.
    # Cycle edges e1, e2, e3, e4.
    # Invariant: ci != cj for i != j.
    # Yes, that is exactly the definition of a rainbow cycle.
    
    # So the problem reduces to: **Edge coloring such that no 4-cycle is monochromatic or has any repeated color.**
    # Which is just: **No two edges in any 4-cycle share the same color.**
    
    # This is a strong constraint.
    
    # Algorithm:
    # 1. Generate all edges.
    # 2. Shuffle edges.
    # 3. For each edge, try to assign a color.
    # 4. If a color leads to a conflict (i.e., there exists another edge in a 4-cycle with this edge that has the same color), reject the color.
    # 5. If no color works, remove the edge from the graph (or restart).
    
    # To maximize edges, we want to minimize removals.
    # Heuristic: Try to keep the graph dense.
    
    # Conflict Check:
    # When assigning color c to edge e=(u,v):
    #   We need to check if there is any other edge e'=(x,y) such that:
    #   1. e and e' are in a common 4-cycle.
    #   2. color(e') == c.
    
    # How to check this fast?
    # Precompute for each edge e, the set of edges e' that share a 4-cycle with e.
    # Let's call this `cycle_neighbors[e]`.
    # When assigning c to e:
    #   For each e' in cycle_neighbors[e]:
    #     If color(e') == c: Conflict.
    
    # Size of cycle_neighbors[e]:
    # Edge (u,v).
    # 4-cycles containing (u,v) involve two other vertices x,y from V\{u,v}.
    # The other edges in the cycle are (x,y) and the connections to u,v.
    # Specifically, for a fixed pair {x,y}, the cycles are:
    # 1. u-v-x-y-u. Edges: (u,v), (v,x), (x,y), (y,u).
    #    Neighbors: (v,x), (x,y), (y,u).
    # 2. u-v-y-x-u. Edges: (u,v), (v,y), (y,x), (x,u).
    #    Neighbors: (v,y), (y,x), (x,u).
    
    # So for each pair {x,y} (153 pairs), we have 6 neighbor edges.
    # Total neighbors = 153 * 6 = 918?
    # Let's verify uniqueness.
    # Are these neighbors distinct?
    # (v,x) is distinct from (v,y) if x!=y.
    # (x,y) is the same in both.
    # (y,u) and (x,u) are distinct.
    # So for a pair {x,y}, the neighbors are:
    # (v,x), (x,y), (u,y) from cycle 1.
    # (v,y), (x,y), (u,x) from cycle 2.
    # Unique set: {(v,x), (u,y), (v,y), (u,x), (x,y)}.
    # So 5 unique neighbors per pair {x,y}.
    # Total neighbors = 153 * 5 = 765.
    
    # So for each edge, we check ~765 other edges.
    # If we maintain a list of edges by color, `edges_by_color[c]`,
    # we can check:
    #   For each e' in edges_by_color[c]:
    #     If e' in cycle_neighbors[e]: Conflict.
    
    # To make this O(1) or O(k), we can use a bitset for cycle_neighbors[e].
    # num_edges = 190.
    # Bitset is 190 bits -> 3 integers or a Python int (arbitrary precision).
    # `neighbor_mask[e]` is a Python int with bits set for neighbor edges.
    
    # Check:
    #   `color_mask[c]` is a Python int with bits set for edges of color c.
    #   If `neighbor_mask[e] & color_mask[c]` is not 0, then there is a conflict.
    
    # This is extremely fast!
    
    # Algorithm:
    # 1. Precompute `neighbor_mask` for all 190 edges.
    # 2. `color_mask` = [0] * 20.
    # 3. `adj` = [[-1]*20 for _ in range(20)].
    # 4. `result_edges` = [].
    # 5. `edge_order` = shuffled list of all 190 edges.
    # 6. For each edge e in edge_order:
    #      Try colors in a random order (or 0..19).
    #      For each color c:
    #         If `neighbor_mask[e] & color_mask[c]` == 0:
    #             Assign c to e.
    #             `color_mask[c] |= (1 << e)`
    #             `adj[u][v] = c`
    #             `result_edges.append([u,v,c])`
    #             Break
    #      If no color works:
    #         Skip edge e (do not add to graph).
    # 7. Return result_edges.
    
    # This greedy approach is O(E * C * (BitOp)).
    # E=190, C=20. BitOp is fast.
    # This should run in milliseconds.
    
    # To improve density, we can try multiple random orderings and color orderings.
    # We will run multiple restarts within the time limit.
    
    # Precomputation:
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Build neighbor masks
    # neighbor_mask[i] is a bitmask of edges that share a 4-cycle with edge i.
    neighbor_mask = [0] * num_edges
    
    # For each edge (u,v) with index i:
    for i, (u, v) in enumerate(all_edges):
        # Find all pairs (x,y) from remaining vertices
        remaining = [w for w in range(n) if w != u and w != v]
        for x_idx in range(len(remaining)):
            for y_idx in range(x_idx + 1, len(remaining)):
                x = remaining[x_idx]
                y = remaining[y_idx]
                
                # Cycle 1: u-v-x-y-u
                # Neighbors: (v,x), (x,y), (y,u)
                e_vx = edge_to_idx.get((min(v,x), max(v,x)))
                e_xy = edge_to_idx.get((min(x,y), max(x,y)))
                e_yu = edge_to_idx.get((min(y,u), max(y,u)))
                
                if e_vx != -1: neighbor_mask[i] |= (1 << e_vx)
                if e_xy != -1: neighbor_mask[i] |= (1 << e_xy)
                if e_yu != -1: neighbor_mask[i] |= (1 << e_yu)
                
                # Cycle 2: u-v-y-x-u
                # Neighbors: (v,y), (y,x), (x,u)
                e_vy = edge_to_idx.get((min(v,y), max(v,y)))
                # e_xy already added
                e_xu = edge_to_idx.get((min(x,u), max(x,u)))
                
                if e_vy != -1: neighbor_mask[i] |= (1 << e_vy)
                if e_xu != -1: neighbor_mask[i] |= (1 << e_xu)
                
                # Note: (x,y) is the same in both cycles, so we don't need to add it twice.
                # The union of neighbors for pair {x,y} is {(v,x), (x,y), (y,u), (v,y), (x,u)}.
    
    # Self-check: An edge is not a neighbor of itself.
    # The masks above only add other edges.
    
    best_result = []
    best_count = 0
    
    # Run multiple restarts
    # Time budget per restart?
    # Since each restart is very fast, we can do many.
    # Let's do 1000 restarts or until time limit.
    
    random.seed(42)
    
    for restart in range(2000):
        if time.time() > DEADLINE - 5:
            break
            
        # Shuffle edge order
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        # Shuffle color order for each edge? Or global?
        # Let's use a random permutation of colors for the trial.
        colors_order = list(range(num_colors))
        random.shuffle(colors_order)
        
        color_mask = [0] * num_colors
        adj = [[-1]*n for _ in range(n)]
        current_edges = []
        
        for e_idx in edge_order:
            u, v = all_edges[e_idx]
            assigned = False
            for c in colors_order:
                # Check conflict
                if neighbor_mask[e_idx] & color_mask[c]:
                    continue
                # No conflict
                color_mask[c] |= (1 << e_idx)
                adj[u][v] = c
                adj[v][u] = c
                current_edges.append([u, v, c])
                assigned = True
                break
            
            # If not assigned, we skip this edge.
            # This is a greedy choice. It might be better to backtrack?
            # But backtracking is slow.
            # Let's stick to greedy for now.
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_result = current_edges
            # If we reached a very high density, we might stop early?
            # Max possible is 190.
            # If we get > 150, it's good.
            # Let's keep searching for a bit.
            
    # Fallback: If best_result is empty or very small, use a simple C4-free construction?
    # The greedy method should always find at least a matching or something.
    # Actually, if it fails to assign any edge, it returns empty.
    # But with 20 colors and sparse constraints initially, it should assign many.
    
    if not best_result:
        # Fallback: Star graph
        best_result = [[0, i, 0] for i in range(1, n)]
        
    return best_result

# EVOLVE-BLOCK-END
