"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import defaultdict

def construct():
    n = 20
    colors = 20
    
    # We want to maximize edges such that every 4-cycle is rainbow.
    # A 4-cycle involves 4 vertices and 4 edges.
    # If we can ensure that for any 4 vertices, the edges between them (if they form a cycle) are distinct in color, we are good.
    # Actually, the condition is: for every 4-cycle in the graph, the 4 edges must have distinct colors.
    
    # Strategy:
    # Start with a dense graph (e.g., complete graph K20) and try to assign colors such that the condition holds.
    # Or, start with a sparse graph and add edges.
    # Given the constraint is local to 4-cycles, a random assignment on a dense graph might fail often.
    # However, if we use a structured coloring, we might succeed.
    
    # Let's try a specific construction:
    # If we color edge (u, v) with (u + v) % 20, does it work?
    # Consider a 4-cycle u-v-w-x-u.
    # Edges: (u,v), (v,w), (w,x), (x,u).
    # Colors: (u+v)%20, (v+w)%20, (w+x)%20, (x+u)%20.
    # We need these 4 to be distinct.
    # This is not guaranteed. For example, if u=0, v=1, w=2, x=3:
    # 1, 3, 5, 3 -> 3 is repeated. So this simple scheme fails.
    
    # Let's try a Max-SAT / Local Search approach.
    # Variables:
    # 1. Presence of edge (u, v) for u < v. Let E[u][v] be 0 or 1.
    # 2. Color of edge (u, v) if present. Let C[u][v] be in 0..19.
    
    # Objective: Maximize sum of E[u][v].
    # Penalty: For every 4-cycle (u,v,w,x), if all 4 edges are present, and their colors are not all distinct, add a large penalty.
    
    # Since n=20, the number of potential edges is 190.
    # The number of 4-cycles in K20 is large: C(20,4) * 3 (ways to pair 4 vertices into a cycle) = 4845 * 3 = 14535.
    # Wait, number of 4-cycles in K_n is (n choose 4) * 3.
    # 20 choose 4 = 4845. 4845 * 3 = 14535 cycles.
    
    # We can use a simple local search:
    # 1. Initialize E and C randomly or heuristically.
    # 2. Calculate score = edges - penalty.
    # 3. Mutate: flip an edge presence, or change a color.
    # 4. Accept if score improves (or with some probability for annealing).
    
    # To make it fast, we can precompute the list of 4-cycles.
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a specific ordering.
    # Actually, a 4-cycle in a simple graph is defined by 4 distinct vertices and the 4 edges connecting them in a cycle.
    # For 4 vertices, there are 3 possible cycles: (a-b-c-d-a), (a-b-d-c-a), (a-c-b-d-a).
    # But since the graph is undirected, these correspond to the 3 ways to partition the 4 vertices into 2 pairs of opposite vertices? No.
    # In K4, there are 3 distinct 4-cycles (Hamiltonian cycles).
    # Let's list them for vertices 0,1,2,3:
    # 0-1-2-3-0 (edges: 01, 12, 23, 30)
    # 0-1-3-2-0 (edges: 01, 13, 32, 20) -> same set of edges? No. 01, 13, 23, 02.
    # 0-2-1-3-0 (edges: 02, 21, 13, 30) -> 02, 12, 13, 03.
    
    # So for each set of 4 vertices, there are 3 specific sets of 4 edges that form a cycle.
    # We need to check these 3 specific edge sets for each 4-vertex subset.
    
    # Precompute all 4-cycles (as tuples of 4 edges).
    cycles = []
    vertices = range(n)
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # The 3 cycles are:
        # 1. a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
        # 2. a-b-d-c-a: edges (a,b), (b,d), (d,c), (c,a)
        # 3. a-c-b-d-a: edges (a,c), (c,b), (b,d), (d,a)
        
        e1 = ((a,b), (b,c), (c,d), (d,a))
        e2 = ((a,b), (b,d), (d,c), (c,a))
        e3 = ((a,c), (c,b), (b,d), (d,a))
        
        # Normalize edge representation (u < v)
        def norm(e):
            u, v = e
            return (u, v) if u < v else (v, u)
        
        cycles.append(tuple(sorted([norm(e) for e in e1])))
        cycles.append(tuple(sorted([norm(e) for e in e2])))
        cycles.append(tuple(sorted([norm(e) for e in e3])))
        
    # Remove duplicates? The 3 cycles for a quad are distinct sets of edges.
    # So total cycles = C(20,4) * 3 = 14535.
    
    # State:
    # present[e] = 1 if edge e is in the graph, else 0.
    # color[e] = color of edge e (only valid if present).
    
    # Let's represent edges by index 0..189.
    edge_list = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_list.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
            
    m = len(edge_list) # 190
    
    # Initialize
    present = [1] * m # Start with complete graph
    color = [random.randint(0, colors-1) for _ in range(m)]
    
    # Precompute for each edge, which cycles it participates in.
    edge_to_cycles = defaultdict(list)
    for i, cyc in enumerate(cycles):
        for e in cyc:
            idx_e = edge_to_idx[e]
            edge_to_cycles[idx_e].append(i)
            
    # Function to calculate penalty for a given cycle
    # A cycle is violated if all 4 edges are present and the 4 colors are not distinct.
    
    # We will use a local search.
    # Score = number of edges - K * (number of violated cycles)
    # K should be large enough to prioritize validity.
    
    def get_score():
        edge_count = sum(present)
        penalty = 0
        for cyc in cycles:
            idxs = [edge_to_idx[e] for e in cyc]
            if all(present[i] for i in idxs):
                cols = [color[i] for i in idxs]
                if len(set(cols)) < 4:
                    penalty += 1
        return edge_count - 100 * penalty # 100 is arbitrary large weight
        
    # Optimize score
    # Simple hill climbing with random restarts or simulated annealing.
    # Given 120s limit, we can do many iterations.
    
    best_score = get_score()
    best_present = present[:]
    best_color = color[:]
    
    # To speed up, we can maintain the penalty count incrementally.
    # But for simplicity and reliability, let's try a few random restarts with local search.
    
    # Actually, let's try a smarter initialization.
    # If we use a proper edge coloring of K20, it's not possible with 20 colors (chromatic index of K20 is 19 or 20? K_n for even n has chromatic index n. So 20 colors are enough for K20).
    # If we use a 1-factorization of K20, each color class is a perfect matching.
    # In a 1-factorization, any 4-cycle uses 4 edges. Are they distinct colors?
    # In a 1-factorization, any two edges with the same color are disjoint.
    # In a 4-cycle, edges are (u,v), (v,w), (w,x), (x,u).
    # (u,v) and (w,x) are disjoint. They *could* have the same color.
    # (v,w) and (x,u) are disjoint. They *could* have the same color.
    # (u,v) and (v,w) share v, so they must have different colors.
    # So in a 1-factorization, a 4-cycle has colors c1, c2, c3, c4.
    # c1 != c2, c2 != c3, c3 != c4, c4 != c1.
    # But c1 can equal c3, and c2 can equal c4.
    # So a 1-factorization does NOT guarantee rainbow 4-cycles.
    
    # We need a stronger condition.
    # What if we restrict the graph to be a union of matchings such that no 4-cycle is formed by edges from only 2 or 3 colors?
    
    # Let's stick to the local search.
    
    # Improve: Simulated Annealing
    T = 100.0
    T_min = 1.0
    cooling = 0.99
    
    current_score = get_score()
    
    # We need to be careful with the time. 14535 cycles * 190 edges is a lot.
    # Let's optimize the penalty calculation.
    # Maintain a count of violations per cycle?
    
    # Let's try a different approach: Randomized greedy construction.
    # Add edges one by one.
    # But the task says "Maximize the number of edges".
    
    # Let's try to find a valid dense graph.
    # If we take a random subset of edges from K20, and random colors, what is the probability of a valid 4-cycle?
    
    # Let's run the local search for a fixed number of iterations.
    
    # Optimization: Precompute cycle indices.
    cycle_indices = []
    for cyc in cycles:
        cycle_indices.append([edge_to_idx[e] for e in cyc])
        
    # State: present (list of 0/1), color (list of ints)
    # We will mutate one variable at a time.
    
    # To make it faster, we can precompute the "conflict" status of each cycle.
    # But updating conflicts after a change is complex.
    
    # Let's just run a simple hill climbing for a few seconds.
    import time
    start_time = time.time()
    max_time = 100.0 # Leave buffer for overhead
    
    current_score = get_score()
    
    while time.time() - start_time < max_time:
        # Choose a random edge index
        i = random.randint(0, m-1)
        
        # Decide mutation: flip presence or change color
        if random.random() < 0.5:
            # Flip presence
            old_val = present[i]
            present[i] = 1 - old_val
            new_score = get_score()
            if new_score > current_score or random.random() < 0.01: # Accept worse with small prob
                current_score = new_score
            else:
                present[i] = old_val
        else:
            # Change color
            old_col = color[i]
            color[i] = random.randint(0, colors-1)
            if color[i] == old_col:
                continue
            new_score = get_score()
            if new_score > current_score or random.random() < 0.01:
                current_score = new_score
            else:
                color[i] = old_col
                
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_present = present[:]
            best_color = color[:]
            
    # Extract result
    result = []
    for i in range(m):
        if best_present[i]:
            u, v = edge_list[i]
            c = best_color[i]
            result.append([u, v, c])
            
    return result

# EVOLVE-BLOCK-END
