import time
import random
import itertools
import sys

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy: 
    1. Start with a very dense graph (e.g., complete graph K20 or random dense graphs).
    2. The constraint is that every 4-cycle must have 4 distinct colors.
    3. We use a local search / backtracking hybrid to find a valid coloring for a given dense graph.
    4. We iterate over different dense graph structures (varying density and randomness) 
       and try to find a valid coloring.
    5. We keep the densest graph for which we find a valid coloring.
    
    Note: The space of valid colorings for dense graphs with 4-cycles is non-trivial.
    However, with 20 colors and n=20, there is significant slack. 
    A C4-free graph is a safe fallback, but we aim for higher density.
    """
    n = 20
    deadline = time.time() + 110
    best_edge_count = 0
    best_result = None
    
    # Precompute all 4-cycles for a given graph is expensive if done naively for every step.
    # Instead, we will generate candidate graphs, and for each, attempt to color it.
    
    def get_4_cycles(adj):
        """
        Returns a list of 4-cycles. Each cycle is a tuple of 4 vertices (a,b,c,d) 
        such that (a,b), (b,c), (c,d), (d,a) are edges.
        To avoid duplicates, we enforce a < b < c < d is NOT sufficient for cycles, 
        but we can enforce a canonical form: min vertex is first, and the second vertex is the smallest among its neighbors in the cycle?
        Actually, for checking, we just need the set of edges involved or the vertices.
        A 4-cycle is defined by 4 distinct vertices. 
        We can find them by iterating pairs of vertices (a,b) and checking common neighbors.
        If a and b have common neighbors c, d, then a-c-b-d-a is a potential 4-cycle? 
        No, a-c-b-d-a requires edges (a,c), (c,b), (b,d), (d,a).
        So for each pair (a,b), let C = N(a) ∩ N(b).
        For each pair (c,d) in C with c < d, the cycle is a-c-b-d-a.
        This generates each 4-cycle twice (once for pair (a,b) and once for (c,d)? No, (a,b) and (c,d) are opposite edges).
        Actually, if we iterate all pairs (a,b) as "diagonals" or "opposite edges", we might double count.
        Let's just generate all sets of 4 vertices {a,b,c,d} and check if they form a C4.
        With n=20, number of 4-sets is C(20,4) = 4845.
        For each set, check if it induces a C4.
        A set of 4 vertices induces a C4 if the subgraph is exactly a cycle of length 4 (or contains one? No, "every 4-cycle (as a subgraph)").
        The task says "Every 4-cycle (as a subgraph)". This usually means if the vertices form a cycle, the colors on those edges must be distinct.
        If the subgraph contains a C4 (i.e., has 4 or more edges forming a cycle), does it matter?
        "checked over all 4-cycles". A 4-cycle is a specific subgraph isomorphic to C4.
        So if vertices {a,b,c,d} have edges (a,b), (b,c), (c,d), (d,a), that is a 4-cycle.
        If they also have (a,c), it's a C4 with a chord. The cycle a-b-c-d-a is still a 4-cycle.
        So we must check all cycles of length 4.
        
        Efficient way:
        For each pair of vertices (u, v) that are NOT adjacent? No, u and v can be adjacent.
        Let's stick to the 4-sets method for clarity and speed (4845 sets is small).
        For each 4-set {a,b,c,d}:
          Check all 3 pairings of the 4 vertices into 2 pairs.
          Pairing 1: (a,b) & (c,d). If both are edges, and the other 4 edges form a cycle?
          Actually, a 4-cycle on vertices {a,b,c,d} is defined by a cyclic ordering.
          There are 3 distinct cyclic orderings (up to rotation/reflection):
          1. a-b-c-d-a (Edges: ab, bc, cd, da)
          2. a-b-d-c-a (Edges: ab, bd, dc, ca)
          3. a-c-b-d-a (Edges: ac, cb, bd, da)
          
          We need to check if any of these 3 configurations exist in the graph.
          If a configuration exists, we must ensure the 4 edge colors are distinct.
        """
        cycles = []
        vertices = range(n)
        for a, b, c, d in itertools.combinations(vertices, 4):
            # Check the 3 possible C4s on these 4 vertices
            # 1. a-b-c-d-a
            if (a,b) in adj and (b,c) in adj and (c,d) in adj and (d,a) in adj:
                cycles.append(((a,b), (b,c), (c,d), (d,a)))
            # 2. a-b-d-c-a
            if (a,b) in adj and (b,d) in adj and (d,c) in adj and (c,a) in adj:
                cycles.append(((a,b), (b,d), (d,c), (c,a)))
            # 3. a-c-b-d-a
            if (a,c) in adj and (c,b) in adj and (b,d) in adj and (d,a) in adj:
                cycles.append(((a,c), (c,b), (b,d), (d,a)))
        return cycles

    def try_color_graph(adj, edge_list, seed):
        """
        Attempts to find a valid coloring for the given graph.
        Uses randomized local search / backtracking.
        Returns colored edges if successful, None otherwise.
        """
        random.seed(seed)
        num_edges = len(edge_list)
        if num_edges == 0:
            return []
        
        # Map edge to index
        edge_idx = {}
        for i, (u, v) in enumerate(edge_list):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
            
        # Initialize colors randomly
        colors = [random.randint(0, 19) for _ in range(num_edges)]
        
        # Precompute constraints: for each 4-cycle, list of edge indices
        # We need the 4-cycles of THIS graph.
        # To save time, we can compute cycles once per graph structure.
        # But we are calling this for different graphs.
        
        # Let's pass cycles as argument to avoid recomputation if we restructure.
        # For now, compute inside.
        cycles = get_4_cycles(adj)
        
        # If no cycles, any coloring works.
        if not cycles:
            return [[u, v, colors[edge_idx[(u,v)]]] for u, v in edge_list]
        
        # Convert cycles to list of edge indices
        cycle_edge_indices = []
        for cyc in cycles:
            idxs = [edge_idx[e] for e in cyc]
            cycle_edge_indices.append(idxs)
            
        # Local Search: Minimize violations (number of 4-cycles with duplicate colors)
        # We can use a simple hill climbing or simulated annealing.
        # Given the small size (20 vertices, ~100-190 edges, ~100-1000 cycles),
        # random restarts with local search should be fast.
        
        best_violations = float('inf')
        best_colors = colors[:]
        
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Calculate current violations
            violations = 0
            violating_cycles = []
            for idxs in cycle_edge_indices:
                c0, c1, c2, c3 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
                if len({c0, c1, c2, c3}) < 4:
                    violations += 1
                    violating_cycles.append(idxs)
                    
            if violations == 0:
                return [[u, v, colors[edge_idx[(u,v)]]] for u, v in edge_list]
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Pick a random violating cycle
            if violating_cycles:
                idxs = random.choice(violating_cycles)
                # Try to change one of the colors in this cycle to fix it
                # Pick a random edge in the cycle
                e_idx = random.choice(idxs)
                # Try all other colors? Or just random new color?
                # Random new color is faster.
                old_c = colors[e_idx]
                new_c = random.randint(0, 19)
                colors[e_idx] = new_c
            else:
                # No violating cycles found? Should not happen if violations > 0
                # Just random move
                e_idx = random.randint(0, num_edges - 1)
                colors[e_idx] = random.randint(0, 19)
                
        # If we didn't find a perfect solution, return best found if it's valid?
        # No, must be valid.
        # Check best_colors
        # (Optional: verify best_colors, but if local search didn't hit 0, it's invalid)
        return None

    # Strategy:
    # 1. Try very dense graphs (e.g., K20, K20 minus few edges).
    # 2. Try random graphs with high edge probability.
    # 3. Try C4-free graphs as fallback (guaranteed valid, lower density).
    
    # Candidate 1: Complete Graph K20 (190 edges)
    # This has many 4-cycles. Coloring might be hard.
    # Candidate 2: Random Dense Graphs (p=0.9, 0.8, 0.7...)
    # Candidate 3: C4-free Greedy (Current approach)
    
    # Let's try a few random dense graphs first.
    densities = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
    
    for p in densities:
        if time.time() > deadline:
            break
        for restart in range(5):
            if time.time() > deadline:
                break
            # Generate random graph
            adj = [set() for _ in range(n)]
            edge_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
                        edge_list.append((u, v))
            
            m = len(edge_list)
            if m <= best_edge_count:
                continue
                
            result = try_color_graph(adj, edge_list, restart * 1000 + int(p*100))
            if result is not None:
                best_edge_count = m
                best_result = result
                # If we found a very dense one, we might stop early?
                # K20 is 190. If we get > 100, it's a good jump.
                if best_edge_count > 150:
                    return best_result

    # Fallback: C4-free greedy (known to work, ~70-80 edges)
    def try_greedy_c4_free(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                return 0, []
            
            common_uv = adj[u].intersection(adj[v])
            if len(common_uv) > 1:
                continue
            
            valid = True
            for x in adj[v]:
                if x == u: continue
                if not adj[u].isdisjoint(adj[x]):
                    valid = False
                    break
            if not valid:
                continue
            
            for x in adj[u]:
                if x == v: continue
                if not adj[v].isdisjoint(adj[x]):
                    valid = False
                    break
            if not valid:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        return len(edges), edges

    for i in range(50):
        if time.time() > deadline:
            break
        m, edges = try_greedy_c4_free(i)
        if m > best_edge_count:
            best_edge_count = m
            # Color with 0s
            best_result = [[u, v, 0] for u, v in edges]

    if best_result is None:
        # Absolute fallback: Star
        best_result = [[0, v, 0] for v in range(1, 20)]
        
    return best_result

# Quick test to ensure it doesn't crash
if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
