import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit

    n = 20
    num_colours = 20

    # 1. Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d and one of 3 orderings.
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        # The 3 distinct 4-cycles on these vertices:
        # 1. a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # 2. a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # 3. a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    num_cycles = len(cycles)

    # 2. Define the nodes of the Conflict Graph
    # A node represents a potential colored edge (u, v, c) with u < v.
    # Total nodes: C(20, 2) * 20 = 190 * 20 = 3800.
    
    edge_ids = []
    node_to_uv = {}
    node_to_c = {}
    id_to_node = {}
    
    node_idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for c in range(num_colours):
                edge_ids.append((u, v, c))
                node_to_uv[node_idx] = (u, v)
                node_to_c[node_idx] = c
                id_to_node[(u, v, c)] = node_idx
                node_idx += 1
                
    total_nodes = node_idx

    # 3. Build Conflict Graph
    # Two nodes conflict if they cannot both be in the solution.
    # Conflicts arise from:
    # A. Same edge (u,v) with different colors (Simple graph constraint)
    # B. Same vertex incident edges with same color (Proper edge coloring)
    # C. 4-cycle rainbow constraint: If a 4-cycle is formed, all 4 colors must be distinct.
    
    # For efficiency, we will build the conflict graph using bitsets or adjacency lists.
    # Given 3800 nodes, a full adjacency matrix is 3800*3800 ~ 14M entries, which fits in memory.
    # But building it naively is slow. We will use a list of sets for adjacency.
    
    conflicts = [set() for _ in range(total_nodes)]
    
    # Optimization: Precompute for each edge (u,v) the list of nodes (u,v,c)
    nodes_for_edge = {}
    for idx in range(total_nodes):
        u, v = node_to_uv[idx]
        if (u,v) not in nodes_for_edge:
            nodes_for_edge[(u,v)] = []
        nodes_for_edge[(u,v)].append(idx)
        
    # Optimization: Precompute for each vertex u and color c, the list of nodes incident to u with color c
    nodes_for_vc = {} # (v, c) -> [node indices]
    for idx in range(total_nodes):
        u, v = node_to_uv[idx]
        c = node_to_c[idx]
        key1 = (u, c)
        key2 = (v, c)
        if key1 not in nodes_for_vc:
            nodes_for_vc[key1] = []
        nodes_for_vc[key1].append(idx)
        if key2 not in nodes_for_vc:
            nodes_for_vc[key2] = []
        nodes_for_vc[key2].append(idx)
            
    # Optimization: Precompute for each edge (u,v) the list of 4-cycles it belongs to
    # And for each cycle, the 4 edges involved.
    # Actually, it's better to iterate over cycles and add conflicts.
    
    # Add Conflicts A and B (Local constraints)
    # A: Same (u,v), different c
    for (u,v), nodes in nodes_for_edge.items():
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                n1 = nodes[i]
                n2 = nodes[j]
                conflicts[n1].add(n2)
                conflicts[n2].add(n1)
                
    # B: Same vertex, same color, different edges
    for (v,c), nodes in nodes_for_vc.items():
        # Nodes in this list are edges incident to v with color c.
        # Any pair of these edges conflicts because they share vertex v and color c.
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                n1 = nodes[i]
                n2 = nodes[j]
                conflicts[n1].add(n2)
                conflicts[n2].add(n1)

    # Add Conflicts C (4-cycle rainbow)
    # This is the heavy part.
    # For each 4-cycle (e1, e2, e3, e4):
    # The constraint is: If e1, e2, e3, e4 are all present, their colors must be distinct.
    # This creates conflicts between specific pairs of colored edges.
    # Specifically, if we choose (e1, c), we cannot choose (e2, c) IF e3 and e4 are also chosen with distinct colors?
    # No, the constraint is global.
    # "Every 4-cycle is rainbow" means: For any 4-cycle C, the set of colors of its edges must have size 4.
    # This is a constraint on the *set* of chosen edges.
    # In the MWIS formulation, we need to ensure that no subset of nodes forms a "bad" configuration.
    # A "bad" configuration is a 4-cycle where two edges have the same color.
    # So, if we pick node (e1, c) and node (e2, c), we MUST NOT pick nodes for e3 and e4 such that they complete the cycle?
    # Actually, if we pick (e1, c) and (e2, c), the cycle is NOT rainbow regardless of e3, e4.
    # Wait. The condition is "Every 4-cycle in the graph is rainbow".
    # If the graph contains edges e1, e2, e3, e4 forming a cycle, and color(e1) == color(e2), it is invalid.
    # Therefore, we cannot have a solution containing (e1, c) and (e2, c) AND (e3, c3) AND (e4, c4) for ANY c3, c4?
    # No. If we have (e1, c) and (e2, c), the cycle is invalid ONLY IF e3 and e4 are also present.
    # If e3 or e4 is missing, the 4-cycle doesn't exist in the graph, so it's not a violation.
    # This makes it a "clique" constraint of size 4? No, it's a constraint on the existence of the subgraph.
    
    # This is tricky for standard MWIS. MWIS handles binary conflicts (pairwise).
    # The constraint here is: NOT ( (e1,c1) AND (e2,c2) AND (e3,c3) AND (e4,c4) AND (c1==c2 OR c1==c3 OR ...) )
    # This is a 4-uniform hypergraph constraint.
    
    # However, we can simplify.
    # If we select (e1, c) and (e2, c), we are "blocking" the completion of any cycle containing e1 and e2.
    # Specifically, for any cycle containing e1 and e2, we cannot select ANY node for e3 and ANY node for e4?
    # No, we just cannot select e3 and e4 *together* with e1 and e2 if they form a cycle.
    # Actually, if e1 and e2 have the same color, they form a "conflict" that prevents the cycle from being valid.
    # So, if we pick (e1, c) and (e2, c), we must ensure that for every cycle containing e1 and e2, at least one of the other two edges is NOT in the graph.
    # This is not a pairwise conflict.
    
    # Alternative approach:
    # The "Rainbow 4-cycle" constraint is equivalent to:
    # For any 4-cycle C = {e1, e2, e3, e4}, the colors must be distinct.
    # This implies that if we choose a color c for e1, we cannot choose color c for e2, e3, or e4 IF the other edges are also chosen.
    
    # Let's look at the structure again.
    # If we fix the graph structure (which edges are present), the coloring must be such that every 4-cycle is rainbow.
    # This is a constraint satisfaction problem on the colors.
    # But we are also choosing the edges.
    
    # Maybe a different formulation is better?
    # What if we iterate over possible edge sets? No, too many.
    
    # Let's stick to the MWIS idea but handle the 4-cycle constraint carefully.
    # Actually, a common trick for "Rainbow" constraints in edge coloring:
    # If we pick (e1, c) and (e2, c), they are "bad" only if e3 and e4 are present.
    # This is a "forbidden subgraph" constraint.
    # Standard MWIS solvers don't handle this directly.
    
    # However, note that the current program uses a greedy approach.
    # The inspiration suggests MWIS.
    # Perhaps we can reduce the problem size?
    # Or use a solver that handles clauses? (SAT/ILP).
    # But we are limited to Python stdlib + NumPy/SciPy. No Gurobi/CPLEX.
    # We can write a custom Branch-and-Bound.
    
    # Let's refine the Conflict Graph approach.
    # We can add "soft" conflicts or handle the 4-cycle constraint in the search step.
    # Or, we can observe that if we ensure the graph is C4-free, the condition is vacuously true.
    # But we want MAX edges. C4-free graphs have at most n^2/4 edges.
    # For n=20, max C4-free is ~100 edges.
    # The current program gets ~86 edges (score 4.3 * 20 = 86).
    # If we allow C4s, we might get more edges, provided we color them rainbow.
    # Is it possible to have > 100 edges?
    # Yes, if we color the 4-cycles properly.
    
    # Let's try a different strategy:
    # 1. Start with a dense graph (e.g., complete graph or random dense graph).
    # 2. Use Local Search / Simulated Annealing to fix colorings and remove edges if necessary.
    # Objective: Maximize edges.
    # Penalty: Number of non-rainbow 4-cycles.
    
    # This is a different design family: Local Search on the space of (Graph, Coloring).
    
    # State:
    # - adj matrix (0/1) for edges.
    # - color matrix for edges.
    # 
    # Moves:
    # 1. Add an edge (if possible to color).
    # 2. Remove an edge.
    # 3. Change color of an edge.
    # 4. Swap colors of two edges.
    
    # This allows us to "fix" violations by recoloring, rather than just removing edges.
    # The greedy approach in the current program removes edges if they can't be colored.
    # Local search can try to recolor existing edges to make room for new ones.
    
    # Let's implement a Local Search strategy.
    
    # Precompute cycles for fast checking.
    # We need to quickly identify which 4-cycles are violated.
    
    # State representation:
    # adj: np.array (n, n) int, -1 for no edge, else color.
    
    def get_violations(adj):
        """Returns list of violating 4-cycles. Each cycle is a tuple of 4 edges."""
        violations = []
        for cycle in cycles:
            e1, e2, e3, e4 = cycle
            c1 = adj[e1[0], e1[1]]
            c2 = adj[e2[0], e2[1]]
            c3 = adj[e3[0], e3[1]]
            c4 = adj[e4[0], e4[1]]
            
            if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                continue
                
            if len({c1, c2, c3, c4}) < 4:
                violations.append(cycle)
        return violations

    def count_edges(adj):
        return int(np.sum(adj != -1)) // 2

    # Initial Solution: Greedy
    def generate_initial():
        adj = np.full((n, n), -1, dtype=int)
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        
        # Simple greedy coloring
        for u, v in edges:
            # Try to find a valid color
            forbidden = set()
            # Proper coloring
            for w in range(n):
                if w != u and adj[u, w] != -1:
                    forbidden.add(adj[u, w])
                if w != v and adj[w, v] != -1:
                    forbidden.add(adj[w, v])
            
            # 4-cycle constraint
            # We need to check if adding (u,v) with color c creates a violation.
            # A violation is created if there is a 4-cycle containing (u,v) where the other 3 edges have colors that block c,
            # or if the other 3 edges already have a conflict.
            
            # Actually, let's just try all colors and pick the first one that doesn't immediately create a violation
            # involving the new edge.
            # This is O(20 * Cost_Check).
            
            best_c = None
            for c in range(num_colours):
                if c in forbidden:
                    continue
                    
                # Check if adding (u,v,c) creates a non-rainbow 4-cycle
                # We only need to check cycles containing (u,v).
                key = (u, v) if u < v else (v, u)
                # We need a mapping from edge to cycles.
                # We built edge_to_cycles in the first draft. Let's rebuild it here for the local search.
                
                # For efficiency, let's precompute edge_to_cycles outside.
                
                valid = True
                for idx in edge_to_cycles.get(key, []):
                    cyc = cycles[idx]
                    colors = []
                    for e in cyc:
                        eu, ev = e
                        if (eu == u and ev == v) or (eu == v and ev == u):
                            colors.append(c)
                        else:
                            cc = adj[eu, ev]
                            if cc == -1:
                                valid = False
                                break
                            colors.append(cc)
                    if not valid:
                        break
                    if len(set(colors)) < 4:
                        valid = False
                        break
                        
                if valid:
                    best_c = c
                    break
                    
            if best_c is not None:
                adj[u, v] = best_c
                adj[v, u] = best_c
                
        return adj

    # Precompute edge_to_cycles for the local search
    edge_to_cycles = {}
    for i, cycle in enumerate(cycles):
        for e in cycle:
            u, v = e
            if u > v:
                u, v = v, u
            key = (u, v)
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(i)

    # Local Search
    def local_search(adj, time_limit):
        current_adj = adj.copy()
        current_edges = count_edges(current_adj)
        best_adj = current_adj.copy()
        best_edges = current_edges
        
        # Precompute violations
        violations = get_violations(current_adj)
        # In the greedy phase, we ensured no violations. So violations should be empty.
        # But if we start with a random dense graph, we'd have violations.
        # Let's start with the greedy result.
        
        # Moves:
        # 1. Try to add a missing edge.
        # 2. If adding fails, try to recolor existing edges to make it possible?
        #    Or remove an edge to resolve a conflict?
        
        # Actually, if we start with a maximal greedy graph, we can't add any edge without violating constraints.
        # To add an edge, we must remove some existing edges or recolor.
        # Recoloring is cheaper.
        
        # Strategy:
        # Loop:
        #   Pick a random missing edge (u, v).
        #   Try to assign a color.
        #   If no color works, identify the blocking cycles.
        #   Try to recolor edges in those cycles to free up a color.
        #   If recoloring succeeds, add the edge.
        #   If not, try to remove an edge from the blocking cycles (sacrifice an edge to gain one? No, net loss).
        #   Better: Try to recolor to resolve conflicts, allowing us to add MORE edges later.
        
        # Actually, a simpler local search:
        # Objective: Maximize edges.
        # Constraint: 0 violations.
        
        # We can use Simulated Annealing.
        # State: adj matrix.
        # Energy: - (num_edges * 100) + (num_violations * 1000)
        # We want to minimize Energy (so maximize edges, minimize violations).
        
        T = 10.0
        cooling = 0.99
        steps = 0
        max_steps = 100000
        
        while time.time() < time_limit and steps < max_steps:
            steps += 1
            
            # Generate neighbor
            new_adj = current_adj.copy()
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Add or Remove edge
                # Pick random edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                
                if new_adj[u, v] == -1:
                    # Try to add
                    # Pick random color
                    c = random.randint(0, num_colours-1)
                    new_adj[u, v] = c
                    new_adj[v, u] = c
                else:
                    # Remove
                    new_adj[u, v] = -1
                    new_adj[v, u] = -1
                    
            elif move_type < 0.8:
                # Recolor edge
                # Pick random existing edge
                edges_present = [(u, v) for u in range(n) for v in range(u+1, n) if current_adj[u, v] != -1]
                if not edges_present: continue
                u, v = random.choice(edges_present)
                new_c = random.randint(0, num_colours-1)
                new_adj[u, v] = new_c
                new_adj[v, u] = new_c
                
            else:
                # Swap colors of two edges
                edges_present = [(u, v) for u in range(n) for v in range(u+1, n) if current_adj[u, v] != -1]
                if len(edges_present) < 2: continue
                e1, e2 = random.sample(edges_present, 2)
                c1 = new_adj[e1[0], e1[1]]
                c2 = new_adj[e2[0], e2[1]]
                new_adj[e1[0], e1[1]] = c2
                new_adj[e1[1], e1[0]] = c2
                new_adj[e2[0], e2[1]] = c1
                new_adj[e2[1], e2[0]] = c1
            
            # Evaluate
            new_edges = count_edges(new_adj)
            new_violations = len(get_violations(new_adj))
            
            curr_edges = count_edges(current_adj)
            curr_violations = len(get_violations(current_adj))
            
            # Score: We want high edges, low violations.
            # Let's use a penalty method.
            # Value = edges - 10 * violations
            curr_val = curr_edges - 10 * curr_violations
            new_val = new_edges - 10 * new_violations
            
            delta = new_val - curr_val
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_adj = new_adj
                if new_edges > best_edges and new_violations == 0:
                    best_adj = new_adj.copy()
                    best_edges = new_edges
            
            T *= cooling
            
        return best_adj, best_edges

    # Run Local Search
    initial_adj = generate_initial()
    best_adj, best_edges = local_search(initial_adj, deadline)
    
    # Final check
    if get_violations(best_adj):
        # Fallback to a valid graph if SA failed
        # Use the greedy result if it was valid
        if not get_violations(initial_adj):
            best_adj = initial_adj
            best_edges = count_edges(best_adj)
        else:
            # Emergency: Star
            adj = np.full((n, n), -1, dtype=int)
            for v in range(1, n):
                adj[0, v] = 0
                adj[v, 0] = 0
            best_adj = adj
            best_edges = 19

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
