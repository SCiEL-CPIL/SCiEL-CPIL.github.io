import time
import random
import numpy as np
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 135s for computation
    deadline = start_time + 135.0
    
    n = 20
    num_colours = 20
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute all 4-cycles and map edges to cycles
    # A 4-cycle is defined by 4 distinct vertices. There are 3 ways to connect 4 vertices in a cycle.
    # For vertices a,b,c,d, the cycles are:
    # 1. a-b-c-d-a  (edges: ab, bc, cd, da)
    # 2. a-b-d-c-a  (edges: ab, bd, dc, ca)
    # 3. a-c-b-d-a  (edges: ac, cb, bd, da)
    
    # We need to check if a set of 4 edges forms a rainbow cycle.
    # Strategy:
    # State: adj matrix (n x n), -1 for no edge, else color.
    # Move: Add edge (u,v) with color c, or Remove edge (u,v).
    # Validity: All 4-cycles must be rainbow.
    
    # To make moves fast, we precompute for each edge (u,v) the list of 4-cycles it participates in.
    # However, checking validity of a *new* edge addition requires checking all cycles containing that edge.
    # If we add (u,v), we only need to check cycles containing (u,v).
    # If we remove (u,v), the graph remains valid (removing edges cannot create new invalid cycles).
    
    # Precompute cycle structure
    # cycles_by_edge[edge] = list of tuples of 3 other edges in the cycle
    # Note: An edge can be in multiple cycles.
    # For a specific 4-cycle {e1, e2, e3, e4}, if we add e1, we must check if {c1, c2, c3, c4} are distinct.
    
    # Let's build a list of all 4-cycles as tuples of 4 edges (frozenset for uniqueness? No, order matters for color check? No, just the set of edges).
    # Actually, for a specific cycle instance, we just need the 4 edges.
    # Let's store: cycles = list of (edge1, edge2, edge3, edge4)
    # And edge_to_cycles: dict mapping edge -> list of cycle_indices
    
    cycles = []
    edge_to_cycles = {e: [] for e in all_possible_edges}
    
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        e1 = (a, b) if a < b else (b, a)
        e2 = (b, c) if b < c else (c, b)
        e3 = (c, d) if c < d else (d, c)
        e4 = (d, a) if d < a else (a, d)
        cyc = (e1, e2, e3, e4)
        # Ensure no duplicate edges (simple graph, distinct vertices in combo ensures distinct edges)
        idx = len(cycles)
        cycles.append(cyc)
        for e in cyc:
            edge_to_cycles[e].append(idx)
            
        # Cycle 2: a-b-d-c-a
        e1 = (a, b) if a < b else (b, a)
        e2 = (b, d) if b < d else (d, b)
        e3 = (d, c) if d < c else (c, d)
        e4 = (c, a) if c < a else (a, c)
        cyc = (e1, e2, e3, e4)
        idx = len(cycles)
        cycles.append(cyc)
        for e in cyc:
            edge_to_cycles[e].append(idx)
            
        # Cycle 3: a-c-b-d-a
        e1 = (a, c) if a < c else (c, a)
        e2 = (c, b) if c < b else (b, c)
        e3 = (b, d) if b < d else (d, b)
        e4 = (d, a) if d < a else (a, d)
        cyc = (e1, e2, e3, e4)
        idx = len(cycles)
        cycles.append(cyc)
        for e in cyc:
            edge_to_cycles[e].append(idx)

    def is_valid_state(adj):
        """Check if the current graph is valid (all 4-cycles are rainbow)."""
        # Optimization: Only check cycles that are "active" (all 4 edges present).
        # But we don't track active cycles. 
        # For n=20, total cycles ~ 3 * C(20,4) = 3 * 4845 = 14535.
        # Checking all 14k cycles is fast in numpy/python if optimized.
        # Let's just iterate all cycles.
        for e1, e2, e3, e4 in cycles:
            if adj[e1[0], e1[1]] != -1 and adj[e2[0], e2[1]] != -1 and \
               adj[e3[0], e3[1]] != -1 and adj[e4[0], e4[1]] != -1:
                c1 = adj[e1[0], e1[1]]
                c2 = adj[e2[0], e2[1]]
                c3 = adj[e3[0], e3[1]]
                c4 = adj[e4[0], e4[1]]
                if len({c1, c2, c3, c4}) < 4:
                    return False
        return True

    def can_add_edge_fast(adj, u, v):
        """
        Try to find a valid color for edge (u,v).
        Returns color or None.
        Uses backtracking/forward checking logic implicitly by just trying colors 
        and checking conflicts with existing cycles containing (u,v).
        """
        e_new = (u, v) if u < v else (v, u)
        # Get all cycles containing this edge
        relevant_cycles = edge_to_cycles[e_new]
        
        # For each color c in 0..19:
        # Check if adding (u,v) with color c creates a non-rainbow cycle.
        # A cycle is violated if:
        # 1. The other 3 edges exist.
        # 2. Their colors are not all distinct from each other AND from c.
        
        # Optimization: Precompute the "forbidden" colors for each cycle?
        # No, depends on current state.
        
        # Let's iterate colors. 
        # To speed up, we can check if ANY cycle is already broken by the other 3 edges?
        # If the other 3 edges have a repeated color, then NO color c can fix it.
        # So if there exists a cycle where the other 3 edges exist and have a color repeat, return None.
        
        for c in range(num_colours):
            valid = True
            for idx in relevant_cycles:
                e1, e2, e3, e4 = cycles[idx]
                # Identify the other 3 edges
                other_edges = []
                for e in (e1, e2, e3, e4):
                    if e != e_new:
                        other_edges.append(e)
                
                # Check if all other 3 edges exist
                exists = True
                colors = []
                for e in other_edges:
                    if adj[e[0], e[1]] == -1:
                        exists = False
                        break
                    colors.append(adj[e[0], e[1]])
                
                if not exists:
                    continue
                
                # Check if colors are distinct from each other and from c
                # {c, colors[0], colors[1], colors[2]} must have size 4
                if len({c, colors[0], colors[1], colors[2]}) < 4:
                    valid = False
                    break
            
            if valid:
                return c
        return None

    # Simulated Annealing
    # State: adj matrix
    # Objective: Maximize edges.
    # We start with a random valid graph (e.g., greedy construction) or empty.
    # Greedy construction is fast and gives a decent start.
    
    def greedy_construct():
        adj = np.full((n, n), -1, dtype=int)
        edges = all_possible_edges[:]
        random.shuffle(edges)
        
        # To speed up greedy, we can use a simpler check or the same can_add_edge_fast
        # can_add_edge_fast is O(Cycles_per_edge * NumColors * 3). 
        # Cycles_per_edge is small.
        for u, v in edges:
            c = can_add_edge_fast(adj, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
        return adj

    # Initial state
    adj = greedy_construct()
    current_edges = int(np.sum(adj != -1)) // 2
    best_adj = adj.copy()
    best_edges = current_edges
    
    T = 10.0 # Initial temperature
    cooling = 0.999
    min_T = 0.1
    
    # Precompute list of existing edges for removal moves
    def get_existing_edges(adj):
        return [(u, v) for u, v in all_possible_edges if adj[u, v] != -1]

    def get_missing_edges(adj):
        return [(u, v) for u, v in all_possible_edges if adj[u, v] == -1]

    while time.time() < deadline:
        if T < min_T:
            break
            
        # Choose move type: Add (prob 0.6) or Remove (prob 0.4)
        # If we are stuck (no adds possible), force removes.
        
        # Get lists
        existing = get_existing_edges(adj)
        missing = get_missing_edges(adj)
        
        if not existing:
            break
            
        move_type = random.random()
        
        new_adj = adj.copy()
        
        if move_type < 0.6:
            # Try to add an edge
            if not missing:
                continue # No missing edges, can't add
            u, v = random.choice(missing)
            c = can_add_edge_fast(new_adj, u, v)
            if c is None:
                continue # Can't add this edge with any color
            new_adj[u, v] = c
            new_adj[v, u] = c
            new_edges = current_edges + 1
            delta = 1
        else:
            # Try to remove an edge
            u, v = random.choice(existing)
            new_adj[u, v] = -1
            new_adj[v, u] = -1
            new_edges = current_edges - 1
            delta = -1
            # Note: Removal is always valid if the previous state was valid.
            # We assume current state is valid.
            
        # Calculate acceptance
        if delta > 0:
            accept = True
        else:
            # Metropolis criterion
            # P = exp(-delta / T) = exp(1/T) for delta=-1
            # Actually, we want to maximize edges, so "energy" is -edges.
            # Delta E = -Delta Edges.
            # If we remove an edge, Delta Edges = -1, Delta E = 1.
            # P = exp(-Delta E / T) = exp(-1/T)
            prob = np.exp(-1.0 / T)
            accept = random.random() < prob
            
        if accept:
            adj = new_adj
            current_edges = new_edges
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
                
        T *= cooling

    # Final verification of best_adj
    # If best_adj is invalid (shouldn't be, but just in case), fall back to greedy
    if not is_valid_state(best_adj):
        best_adj = greedy_construct()
        best_edges = int(np.sum(best_adj != -1)) // 2
        if not is_valid_state(best_adj):
             # Extremely unlikely fallback: empty graph
             best_adj = np.full((n, n), -1, dtype=int)
             best_edges = 0

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
