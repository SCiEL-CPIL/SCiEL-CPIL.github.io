import time
import random
import sys
import numpy as np

def construct():
    """
    Construct a dense edge-coloured graph on 20 vertices (0..19) with 20 colours (0..19)
    such that every 4-cycle is rainbow.
    
    Strategy:
    1. Define all possible edges (190 total).
    2. Model the problem as finding a subset of edges and assigning colours such that no 4-cycle has duplicate colours.
    3. Since colouring is hard to optimize jointly with edge selection in simple SA, we use a "conflict graph" approach:
       - A "state" is a set of edges and their colours.
       - However, checking all 4-cycles is expensive.
       - Alternative: Use a conflict graph where nodes are (edge, colour) pairs? No, that's 190*20 = 3800 nodes.
       
    Let's refine the SA approach:
    State: A list of (u, v, c) for active edges.
    Objective: Maximize number of edges.
    Constraint: No 4-cycle (u,v,w,x) has c(uv)==c(vw), c(uv)==c(wx), etc.
    
    To make this tractable, we precompute all 4-cycles in K20.
    Number of 4-cycles in K20 is C(20,4) * 3 = 4845 * 3 = 14535 distinct 4-cycles (unoriented).
    Actually, the number of distinct 4-cycles (subgraphs) is C(n,4) * 3.
    
    We will use a Simulated Annealing approach where:
    - State: adj_matrix (20x20) with -1 or color.
    - Moves: 
        1. Add an edge with a random color.
        2. Change color of an existing edge.
        3. Remove an edge.
    - Energy Function: -num_edges * weight_edge + num_violations * weight_violation.
    - We want to minimize Energy (maximize edges, minimize violations).
    
    To speed up violation checking:
    When adding/changing edge (u,v,c), we only need to check 4-cycles containing (u,v).
    How many 4-cycles contain a specific edge (u,v)?
    A 4-cycle containing (u,v) is of the form u-v-w-x-u.
    w and x must be distinct from u,v and each other.
    Number of such cycles = C(n-2, 2) * 2? 
    For fixed u,v, we need to pick w, x from remaining 18 vertices.
    The cycle is u-v-w-x-u.
    The edges are (u,v), (v,w), (w,x), (x,u).
    For a pair (w,x), the cycle exists if all 4 edges are present.
    So, for a new edge (u,v,c), we check all pairs (w,x) such that (v,w), (w,x), (x,u) are present.
    There are at most C(18,2) = 153 pairs (w,x).
    For each pair, we check if the colors are distinct.
    This is O(153) per move. Very fast.
    
    Initial State: 
    Start with a valid sparse graph (e.g., star or matching) to ensure 0 violations, 
    or start with empty and add edges.
    Better: Start with a random valid-ish structure or just empty and rely on SA to grow.
    Actually, starting with a known dense structure (like the bipartite one from seed) but with random colors might be a good start, 
    but we want to explore non-bipartite.
    
    Let's start with an empty graph and use SA to add edges.
    To encourage density, we can bias the moves towards adding edges.
    
    Precomputation:
    List of all pairs (w,x) for each edge (u,v)? 
    Actually, for a given edge (u,v), the relevant 4-cycles are determined by neighbors.
    We can just iterate over all w != u,v and x != u,v,w.
    
    Optimization:
    We can precompute for each edge (u,v) the list of potential "witnesses" (w,x) that would form a 4-cycle IF the other edges existed.
    But the existence of the other edges changes.
    
    Let's stick to the O(153) check. 190 edges * 153 checks * 4 color comparisons ~ 100k ops per move.
    If we do 100k moves in 100s, that's 1 move per ms. Feasible.
    
    We will track the best valid graph found.
    
    Deadline: 140s.
    """
    
    n = 20
    num_colours = 20
    start_time = time.time()
    deadline = start_time + 140.0
    
    # adj[u][v] = colour or -1
    adj = [[-1] * n for _ in range(n)]
    
    def count_edges():
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    count += 1
        return count
    
    def check_violations_for_edge(u, v, c):
        """
        Check if adding/changing edge (u,v) to colour c creates any non-rainbow 4-cycle.
        Returns number of violations created.
        """
        violations = 0
        # A 4-cycle involving (u,v) is u-v-w-x-u
        # Edges: (u,v), (v,w), (w,x), (x,u)
        # We need to check all w, x such that (v,w), (w,x), (x,u) exist.
        
        # To avoid double counting or confusion, we fix the order u-v-w-x-u.
        # Since the graph is undirected, we can just iterate all w, x.
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w] == -1:
                continue
            c2 = adj[v][w]
            # If c == c2, it's a violation (two adjacent edges in cycle have same color? No, cycle is u-v-w-x-u. 
            # Colors are c(uv), c(vw), c(wx), c(xu).
            # Violation if any two are equal.
            
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if adj[w][x] == -1:
                    continue
                if adj[x][u] == -1:
                    continue
                
                c3 = adj[w][x]
                c4 = adj[x][u]
                
                # Check distinctness of {c, c2, c3, c4}
                if len({c, c2, c3, c4}) < 4:
                    violations += 1
                    
        return violations

    def total_violations():
        """
        Count total violations in the current graph.
        This is expensive, so we only use it for final verification or initial state.
        For SA, we use incremental checks.
        """
        v_count = 0
        # We need to count each 4-cycle once.
        # A 4-cycle is defined by 4 vertices {u,v,w,x} and a pairing.
        # There are 3 pairings for 4 vertices.
        # We can iterate over all 4-subsets.
        # C(20,4) = 4845.
        # For each subset, check the 3 possible 4-cycles.
        # This is much faster than O(E^2) if E is large.
        
        import itertools
        for u, v, w, x in itertools.combinations(range(n), 4):
            # Cycle 1: u-v-w-x-u
            if adj[u][v] != -1 and adj[v][w] != -1 and adj[w][x] != -1 and adj[x][u] != -1:
                if len({adj[u][v], adj[v][w], adj[w][x], adj[x][u]}) < 4:
                    v_count += 1
            # Cycle 2: u-v-x-w-u
            if adj[u][v] != -1 and adj[v][x] != -1 and adj[x][w] != -1 and adj[w][u] != -1:
                if len({adj[u][v], adj[v][x], adj[x][w], adj[w][u]}) < 4:
                    v_count += 1
            # Cycle 3: u-w-v-x-u
            if adj[u][w] != -1 and adj[w][v] != -1 and adj[v][x] != -1 and adj[x][u] != -1:
                if len({adj[u][w], adj[w][v], adj[v][x], adj[x][u]}) < 4:
                    v_count += 1
        return v_count

    # Initial State: Start with a valid sparse graph to ensure we have a baseline.
    # A star graph is valid (no 4-cycles).
    # Or better, start with a random matching or just empty.
    # Let's start with a random valid coloring of a bipartite graph to get a good head start?
    # The seed program did this. Let's replicate the seed's initial state as the start of SA.
    
    # Replicate seed initial state
    A = list(range(10))
    B = list(range(10, 20))
    for a in A:
        for b in B:
            b_idx = b - 10
            c = (2 * a + b_idx) % 20
            adj[a][b] = c
            adj[b][a] = c
            
    # Remove violations from this initial state using the seed's method to get a valid start
    # This ensures we start with a valid, dense-ish graph.
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        # Find first violation
        found = False
        for u in range(n):
            if found: break
            for v in range(n):
                if found: break
                if v == u or adj[u][v] == -1: continue
                for w in range(n):
                    if found: break
                    if w == u or w == v or adj[v][w] == -1: continue
                    for x in range(n):
                        if x == u or x == v or x == w: continue
                        if adj[w][x] == -1 or adj[x][u] == -1: continue
                        c1 = adj[u][v]
                        c2 = adj[v][w]
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if len({c1, c2, c3, c4}) < 4:
                            # Remove edge (u,v)
                            adj[u][v] = -1
                            adj[v][u] = -1
                            found = True
                            changed = True
                            break
                            
    # Now we have a valid graph. Let's record it.
    best_adj = [row[:] for row in adj]
    best_edges = count_edges()
    
    # Simulated Annealing
    # State: adj
    # We will try to improve by adding edges or recoloring.
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    current_edges = count_edges()
    
    # Precompute list of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Add an edge
            # Pick a random missing edge
            missing_edges = [(u, v) for u, v in all_edges if adj[u][v] == -1]
            if not missing_edges:
                continue
            u, v = random.choice(missing_edges)
            c = random.randint(0, num_colours - 1)
            
            # Calculate delta violations
            # Old violations: 0 for this edge (it wasn't there)
            # New violations: check_violations_for_edge(u, v, c)
            delta_v = check_violations_for_edge(u, v, c)
            delta_e = 1
            
            # Energy delta: -delta_e + delta_v (we want to minimize energy = -edges + violations)
            # Actually, let's use a simple acceptance criterion:
            # Accept if it increases edges and doesn't increase violations.
            # Or accept with probability exp(-delta_E / T) if it increases violations.
            
            # We prioritize edge count.
            # Let's define Energy = -100 * edges + violations.
            # Delta E = -100 * delta_e + delta_v
            
            delta_E = -100 * delta_e + delta_v
            
            if delta_E <= 0 or (delta_v == 0 and delta_e > 0) or (random.random() < np.exp(-delta_E / T)):
                # Accept
                adj[u][v] = c
                adj[v][u] = c
                current_edges += delta_e
                
                # Check if this is a new best valid graph
                # We don't want to call total_violations() every step.
                # We can track violations incrementally? Hard.
                # Instead, we can periodically check if the current graph is valid.
                # If it is valid, update best.
                # To avoid expensive checks, we can check every 100 moves.
                
        elif move_type < 0.8:
            # Change color of an existing edge
            existing_edges = [(u, v) for u, v in all_edges if adj[u][v] != -1]
            if not existing_edges:
                continue
            u, v = random.choice(existing_edges)
            old_c = adj[u][v]
            new_c = random.randint(0, num_colours - 1)
            if new_c == old_c:
                continue
            
            # Calculate delta violations
            # This is tricky because changing a color affects all 4-cycles containing this edge.
            # We need to count violations BEFORE and AFTER.
            # This requires checking all 4-cycles containing (u,v) in the current state.
            # Let's implement a helper to count violations for a specific edge in the current graph.
            
            def count_violations_for_edge_in_graph(u, v):
                count = 0
                for w in range(n):
                    if w == u or w == v: continue
                    if adj[v][w] == -1: continue
                    c2 = adj[v][w]
                    for x in range(n):
                        if x == u or x == v or x == w: continue
                        if adj[w][x] == -1: continue
                        if adj[x][u] == -1: continue
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        c1 = adj[u][v]
                        if len({c1, c2, c3, c4}) < 4:
                            count += 1
                return count
            
            # This is O(153) per call.
            # We need to do this for the old color and new color.
            # But the graph state changes...
            # We can simulate the change:
            # 1. Temporarily set adj[u][v] = new_c
            # 2. Count violations for (u,v)
            # 3. Restore adj[u][v] = old_c
            # 4. Count violations for (u,v)
            # 5. Delta = new_v - old_v
            
            # However, this only counts violations *involving* (u,v).
            # Changing (u,v) color does not affect violations of 4-cycles NOT containing (u,v).
            # So this is correct.
            
            old_v_count = count_violations_for_edge_in_graph(u, v)
            
            adj[u][v] = new_c
            adj[v][u] = new_c
            new_v_count = count_violations_for_edge_in_graph(u, v)
            
            delta_v = new_v_count - old_v_count
            delta_e = 0
            
            delta_E = -100 * delta_e + delta_v
            
            if delta_E <= 0 or (random.random() < np.exp(-delta_E / T)):
                # Keep new color
                pass
            else:
                # Revert
                adj[u][v] = old_c
                adj[v][u] = old_c
                
        else:
            # Remove an edge
            existing_edges = [(u, v) for u, v in all_edges if adj[u][v] != -1]
            if not existing_edges:
                continue
            u, v = random.choice(existing_edges)
            c = adj[u][v]
            
            # Delta violations: removing an edge removes all violations involving it.
            # So delta_v is negative (good).
            # But we lose an edge.
            
            # We only want to remove if it allows adding more later?
            # For now, let's rarely remove.
            if random.random() < 0.1: # 10% chance to actually process removal
                old_v_count = count_violations_for_edge_in_graph(u, v)
                
                adj[u][v] = -1
                adj[v][u] = -1
                # new_v_count is 0 for this edge
                new_v_count = 0
                
                delta_v = new_v_count - old_v_count
                delta_e = -1
                
                delta_E = -100 * delta_e + delta_v
                # delta_E = 100 + delta_v
                # If old_v_count was large, this is good.
                
                if delta_E <= 0 or (random.random() < np.exp(-delta_E / T)):
                    current_edges -= delta_e
                else:
                    adj[u][v] = c
                    adj[v][u] = c

        # Periodic check for best valid graph
        if random.random() < 0.01: # 1% of the time
            if total_violations() == 0:
                curr_e = count_edges()
                if curr_e > best_edges:
                    best_edges = curr_e
                    best_adj = [row[:] for row in adj]
                    # Restart SA from this good state?
                    # Or just keep going.
                    # Let's reset temperature to explore more?
                    # T = max(T_min, T * 1.5)
                    
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Safety check
        if time.time() > deadline:
            break
            
    # Final verification of best_adj
    # We assume best_adj is valid because we only updated it when total_violations()==0
    # But let's double check to be safe.
    if total_violations() == 0:
        curr_e = count_edges()
        if curr_e > best_edges:
            best_edges = curr_e
            best_adj = [row[:] for row in adj]
            
    # Construct output from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] != -1:
                edges.append([u, v, best_adj[u][v]])
                
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
