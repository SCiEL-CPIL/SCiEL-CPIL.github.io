import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s limit
    
    n = 20
    colors = 20
    
    # Helper to check if a 4-cycle is rainbow
    # A 4-cycle is (u, v, w, x, u) where edges are (u,v), (v,w), (w,x), (x,u)
    # We represent the graph as an adjacency matrix of colors, or a dict of edges.
    # For speed in the GA, we'll use a numpy array for adjacency (colors) and a list of active edges.
    
    import numpy as np
    
    # Initialize population
    pop_size = 40
    generation = 0
    
    # Pre-compute all possible 4-cycles? No, too many. 
    # C4 count in K20 is ~ 20*19*18*17 / (4*2) ? No.
    # Number of 4-cycles in K_n is n(n-1)(n-2)(n-3)/24.
    # For n=20: 20*19*18*17/24 = 19380.
    # This is manageable. Let's precompute all 4-cycles as tuples of 4 vertices (u,v,w,x) 
    # such that u<v, w<x, and {u,v} != {w,x} and no shared vertices? 
    # A 4-cycle subgraph is defined by 4 distinct vertices u,v,w,x and the edges (u,v), (v,w), (w,x), (x,u).
    # Actually, the definition of "4-cycle" usually means the cycle C4 as a subgraph.
    # The vertices must be distinct. The edges must exist.
    # There are 3 ways to connect 4 vertices into a cycle. 
    # Let's just iterate over all combinations of 4 distinct vertices (u,v,w,x) and check the 3 possible cycles?
    # Or better: Precompute all unique 4-cycles in K20.
    # A 4-cycle is determined by 4 vertices and the pairing of edges.
    # Let's generate all sets of 4 vertices, and for each set, the 3 possible 4-cycles.
    
    cycles = []
    vertices = list(range(n))
    from itertools import combinations
    
    # This might take a second or two, but it's a one-time cost.
    # 20 choose 4 is 4845. 3 cycles per set. ~14535 cycles.
    for quad in combinations(vertices, 4):
        u, v, w, x = quad
        # The 3 cycles on these 4 vertices:
        # 1: u-v-w-x-u  (Edges: uv, vw, wx, xu)
        # 2: u-v-x-w-u  (Edges: uv, vx, xw, wu)
        # 3: u-w-v-x-u  (Edges: uw, wv, vx, xu)
        
        # We need to ensure we don't double count or miss.
        # Standard representation: sorted edges.
        
        # Cycle 1: (u,v), (v,w), (w,x), (x,u)
        e1 = [tuple(sorted((u,v))), tuple(sorted((v,w))), tuple(sorted((w,x))), tuple(sorted((x,u)))]
        # Cycle 2: (u,v), (v,x), (x,w), (w,u)
        e2 = [tuple(sorted((u,v))), tuple(sorted((v,x))), tuple(sorted((x,w))), tuple(sorted((w,u)))]
        # Cycle 3: (u,w), (w,v), (v,x), (x,u)
        e3 = [tuple(sorted((u,w))), tuple(sorted((w,v))), tuple(sorted((v,x))), tuple(sorted((x,u)))]
        
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
        
    # Total cycles ~ 14535.
    
    # State representation: 
    # adj: numpy array (20, 20) of ints. 0 means no edge, 1-20 is color.
    # Since graph is simple and undirected, adj[u][v] == adj[v][u].
    
    def evaluate(adj):
        """Returns (num_edges, is_valid)"""
        num_edges = 0
        # Count edges (upper triangle)
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] > 0:
                    num_edges += 1
        
        # Check validity
        # For speed, we can break early on violation.
        for e1, e2, e3, e4 in cycles:
            c1 = adj[e1[0][0]][e1[0][1]]
            c2 = adj[e2[0][0]][e2[0][1]]
            c3 = adj[e3[0][0]][e3[0][1]]
            c4 = adj[e4[0][0]][e4[0][1]]
            
            # If any edge is missing (0), this specific 4-cycle doesn't exist in the subgraph sense?
            # The task says: "Every 4-cycle (as a subgraph) must receive four distinct colours".
            # This implies we only check cycles where ALL 4 edges exist.
            if c1 == 0 or c2 == 0 or c3 == 0 or c4 == 0:
                continue
                
            if len(set([c1, c2, c3, c4])) < 4:
                return num_edges, False
                
        return num_edges, True

    def repair(adj):
        """Removes edges to make the graph valid. Returns new adj, num_removed."""
        removed = 0
        # We need to find a valid subgraph.
        # Greedy removal: iterate through cycles, if invalid, remove one edge.
        # To be smart, remove the edge that is part of the most other invalid cycles? 
        # For simplicity in GA, let's just remove a random edge from the invalid cycle.
        # Or better: iterate through all cycles. If a cycle is present and invalid, remove one of its edges.
        # Which one? The one with the "highest" color? Or random?
        # Let's try: remove the edge that appears in the most *other* currently existing 4-cycles? 
        # That's expensive.
        # Let's just remove a random edge from the violating cycle.
        
        # To avoid infinite loops or inefficient removal, we can check cycles in a fixed order.
        # If a cycle is invalid, pick a random edge from it and set to 0.
        
        # Optimization: Only check cycles that are still "potentially" invalid.
        # Since we modify the graph, we have to re-check.
        # However, if we remove an edge, we fix all cycles containing that edge.
        
        # Strategy: 
        # 1. Identify all currently existing 4-cycles.
        # 2. Check which are invalid.
        # 3. If none, return.
        # 4. Pick a random invalid cycle. Remove a random edge from it.
        # 5. Repeat.
        
        # This could be slow if we do it naively.
        # Let's estimate: Max edges 190. Max cycles 14535.
        # In a dense graph, most cycles exist.
        # If we remove edges one by one, we might need to remove many.
        
        # Faster repair:
        # Maintain a list of active cycles? 
        # Let's just do a simple loop.
        
        changed = True
        while changed:
            changed = False
            for e1, e2, e3, e4 in cycles:
                c1 = adj[e1[0][0]][e1[0][1]]
                c2 = adj[e2[0][0]][e2[0][1]]
                c3 = adj[e3[0][0]][e3[0][1]]
                c4 = adj[e4[0][0]][e4[0][1]]
                
                if c1 == 0 or c2 == 0 or c3 == 0 or c4 == 0:
                    continue
                    
                if len(set([c1, c2, c3, c4])) < 4:
                    # Violation found. Remove a random edge.
                    edges_in_cycle = [(e1[0][0], e1[0][1]), (e2[0][0], e2[0][1]), (e3[0][0], e3[0][1]), (e4[0][0], e4[0][1])]
                    u, v = random.choice(edges_in_cycle)
                    adj[u][v] = 0
                    adj[v][u] = 0
                    removed += 1
                    changed = True
                    # Break out of cycle loop to re-evaluate from start? 
                    # Or continue? If we continue, we might remove more edges from the same "region".
                    # Breaking and restarting ensures we don't rely on stale state in the loop, 
                    # but it's slower. Let's continue, it's usually fine.
                    
        return adj, removed

    def random_individual():
        """Generates a random dense graph and repairs it."""
        adj = np.zeros((n, n), dtype=int)
        # Fill with random colors
        for i in range(n):
            for j in range(i+1, n):
                # High probability of edge
                if random.random() < 0.9:
                    c = random.randint(1, colors)
                    adj[i][j] = c
                    adj[j][i] = c
        adj, _ = repair(adj)
        return adj

    # Initialize population
    population = []
    for _ in range(pop_size):
        pop = random_individual()
        num_edges, is_valid = evaluate(pop)
        population.append((pop, num_edges, is_valid))
        
    # Sort by num_edges descending
    population.sort(key=lambda x: x[1], reverse=True)
    
    best_adj = population[0][0]
    best_score = population[0][1]
    
    # If best is not valid, we need to ensure we return a valid one.
    # The repair function should guarantee validity.
    # Let's double check the first one.
    if not population[0][2]:
        # Force repair again if something went wrong
        best_adj, _ = repair(best_adj.copy())
        best_score, is_valid = evaluate(best_adj)
        if not is_valid:
             # Fallback to star
             star = np.zeros((n, n), dtype=int)
             for v in range(1, n):
                 star[0][v] = 1
                 star[v][0] = 1
             return [[0, v, 1] for v in range(1, n)]

    while time.time() < deadline:
        generation += 1
        
        new_population = []
        
        # Elitism: keep top 2
        new_population.append(population[0])
        new_population.append(population[1])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Tournament selection
            p1 = random.choice(population[:10])
            p2 = random.choice(population[:10])
            
            if p1[1] >= p2[1]:
                parent = p1[0]
            else:
                parent = p2[0]
                
            # Mutation: Add a random edge with random color, or change color of existing edge, or remove edge
            child = parent.copy()
            
            mut_type = random.random()
            
            if mut_type < 0.5:
                # Add edge
                u, v = random.sample(range(n), 2)
                if child[u][v] == 0:
                    child[u][v] = random.randint(1, colors)
                    child[v][u] = child[u][v]
            elif mut_type < 0.8:
                # Change color
                u, v = random.sample(range(n), 2)
                if child[u][v] > 0:
                    child[u][v] = random.randint(1, colors)
                    child[v][u] = child[u][v]
            else:
                # Remove edge (less likely, but helps escape local optima if stuck in invalid states? 
                # Actually repair handles validity. Removing edges reduces score. 
                # Maybe swap colors?
                u, v = random.sample(range(n), 2)
                if child[u][v] > 0:
                     # Swap color with another edge?
                     # Just random color change is fine.
                     child[u][v] = random.randint(1, colors)
                     child[v][u] = child[u][v]
                     
            # Repair
            child, _ = repair(child)
            
            # Evaluate
            num_edges, is_valid = evaluate(child)
            new_population.append((child, num_edges, is_valid))
            
            if num_edges > best_score and is_valid:
                best_score = num_edges
                best_adj = child.copy()
                
        # Update population
        new_population.sort(key=lambda x: x[1], reverse=True)
        population = new_population
        
        # Check time
        if time.time() > deadline - 5:
             break

    # Final validation and formatting
    # Ensure best_adj is valid
    final_edges, is_valid = evaluate(best_adj)
    if not is_valid:
        # Try to repair best_adj one last time
        best_adj, _ = repair(best_adj.copy())
        final_edges, is_valid = evaluate(best_adj)
        if not is_valid:
             # Fallback to star
             star = np.zeros((n, n), dtype=int)
             for v in range(1, n):
                 star[0][v] = 1
                 star[v][0] = 1
             return [[0, v, 1] for v in range(1, n)]

    result = []
    for i in range(n):
        for j in range(i+1, n):
            c = best_adj[i][j]
            if c > 0:
                result.append([i, j, int(c)])
                
    return result

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
