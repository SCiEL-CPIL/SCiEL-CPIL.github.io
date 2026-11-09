import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    def get_lazebnik_woldar():
        """
        Construct a 5-coloring of K_26 without monochromatic C4.
        Based on the Lazebnik-Woldar construction for R_5(C4).
        
        Vertices are pairs (x, y) where x in Z_5, y in Z_5 \cup {infinity}.
        Total 5 * 6 = 30? No, the construction for R_5(C4) usually yields 26.
        Let's recall the specific construction for 5 colors and n=26.
        
        The known bound for R_5(C4) is 26.
        A common construction uses the affine plane AG(2, 5) or similar.
        However, AG(2, 5) has 25 points. Adding a point at infinity gives 26?
        
        Let's use the explicit construction from the literature for n=26.
        The construction often maps vertices to elements of Z_5 x (Z_5 U {inf})?
        Actually, the standard Lazebnik-Woldar construction for R_k(C4) with k=5:
        It is known that R_5(C4) > 26.
        
        One specific construction:
        Vertices: V = Z_5 x Z_5 (25 vertices) + {infinity} = 26 vertices.
        Edges are colored based on slopes or differences.
        
        Let's try a simpler algebraic approach if possible, or hardcode the known structure if I can derive it quickly.
        Since I cannot look up the exact code, I will implement a robust randomized local search that is seeded to find n=26 quickly, 
        OR I will implement the specific known construction for n=11 (cyclic) and try to extend it?
        
        Wait, the prompt says "The algebraic construction of Lazebnik and Woldar gives 26 vertices".
        If I can't derive the exact algebraic formula in a few lines without error, 
        I should rely on a very efficient local search that is guaranteed to find n=26 if it exists, 
        or fall back to a known valid smaller n.
        
        However, the directive asks for a MATERIAL change. The current program does random local search.
        The "Greedy Randomized Construction" suggested in the strategy description is a good candidate.
        But the "Inspiration 1" mentions the Lazebnik-Woldar construction explicitly.
        
        Let's implement a high-quality local search that starts from a structured initialization (cyclic-like) 
        and uses a conflict-driven repair strategy, which is structurally different from pure random init.
        
        Actually, let's look at the "Idea target". It mentions "error_repair".
        The current program already does error repair (local search).
        
        Let's try the Greedy Randomized Construction (GRC) as suggested in the directive:
        "Use a greedy randomized construction that sequentially assigns colors to edges while maintaining a dynamic count of 'forbidden pairs' ... prioritizing edge-color assignments that minimize the creation of new forbidden pairs".
        
        This is a forward-looking heuristic (building up the graph) rather than backward-looking (repairing a random graph).
        
        Algorithm:
        1. Initialize n=26.
        2. Create list of all edges.
        3. Shuffle edges.
        4. For each edge, try each color 0-4.
        5. For each color, calculate the "cost" = number of new C4s created (i.e., if we add this edge with color c, how many pairs (u,v) will now have >=2 common neighbors in color c?).
           Actually, adding an edge (u,v) with color c creates a C4 if there is already a path u-w-v of length 2 in color c.
           Specifically, if we add edge (u,v) with color c, it completes a C4 for every w such that (u,w) and (v,w) are already colored c.
           So the immediate cost is the number of such w.
           We want to minimize this cost.
           If min cost > 0, we might still pick the best one, but we risk invalidity.
           To ensure validity, we must only pick a color if it creates 0 new C4s? 
           If we only pick colors that create 0 new C4s, we are doing a "greedy valid extension".
           If we get stuck (no color has cost 0), we can backtrack or restart.
           
        This is a CSP (Constraint Satisfaction Problem) approach with forward checking (greedy).
        
        Let's implement this.
        """
        n = 26
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We will try multiple restarts of the greedy process.
        # Time budget per restart: ~1s. Total 110s -> 100 restarts?
        # Greedy is O(E * C * N) per step? No, we maintain counts.
        
        def greedy_construct(n, time_limit):
            # State
            color = [[-1] * n for _ in range(n)]
            # common[c][u][v] = count of w s.t. color[u][w]==c and color[v][w]==c
            # Note: This count is over ALL w, regardless of whether edge(u,v) is colored yet.
            # When we color edge(u,v) with c, we need to check if it creates a C4.
            # A C4 is formed if there exists w such that color[u][w]==c and color[v][w]==c.
            # So the "immediate conflict" count is exactly common[c][u][v].
            # If common[c][u][v] > 0, adding edge (u,v) with color c creates a C4.
            # Wait, does it?
            # C4: u-w-v-x-u? No, C4 in the graph of color c.
            # A cycle of length 4 in the subgraph of color c.
            # If we add edge (u,v) to the subgraph of color c, it creates a cycle if u and v are already connected by a path of length 2 in color c?
            # No, a path of length 2 u-w-v plus the edge (u,v) creates a triangle (C3), not a C4.
            # A C4 is u-w-x-v-u? Or u-w-v-z-u?
            # The condition "No colour class may contain a four-cycle" means the subgraph G_c is C4-free.
            # Adding an edge (u,v) to G_c creates a C4 if and only if there is a path of length 3 between u and v in G_c?
            # Or if u and v have a common neighbor w, and w has a common neighbor x with v?
            # Let's check the definition in the prompt: "no two vertices with two common neighbours in one colour".
            # This is the standard definition of C4-free for the *line graph*? No.
            # "No two vertices with two common neighbours" is the condition that the graph is C4-free?
            # Let's verify.
            # If u and v have two common neighbors w and x, then u-w-v-x-u is a 4-cycle.
            # Yes, that is a C4.
            # So the condition is: For any color c, for any pair of vertices (u, v), the number of common neighbors in color c must be <= 1.
            
            # So, if we add edge (u,v) with color c:
            # Does it increase the common neighbor count for any pair?
            # It adds u as a neighbor of v (in color c) and v as a neighbor of u (in color c).
            # For any other vertex w:
            #   - If edge(u,w) is color c, then w and v now share u as a common neighbor in color c.
            #     So common[c][w][v] increases by 1.
            #     If common[c][w][v] was already 1, it becomes 2 -> CONFLICT (C4 formed: w-u-v-x-w where x is the other common neighbor).
            #   - If edge(v,w) is color c, then w and u now share v as a common neighbor in color c.
            #     So common[c][w][u] increases by 1.
            #     If common[c][w][u] was already 1, it becomes 2 -> CONFLICT.
            
            # So the "cost" of adding edge (u,v) with color c is the number of w such that:
            # 1. edge(u,w)==c AND common[c][w][v] == 1
            # 2. edge(v,w)==c AND common[c][w][u] == 1
            # (Note: w cannot be u or v).
            
            # We want to pick a color c that minimizes this cost. Ideally cost=0.
            
            # Data structures:
            # color[u][v]
            # common[c][u][v]
            
            # Initialization
            common = [[[0] * n for _ in range(n)] for _ in range(5)]
            
            # We process edges in a random order.
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edge_list)
            
            # To speed up, we can pre-calculate neighbors for each color?
            # Or just iterate.
            
            for u, v in edge_list:
                if time.time() > time_limit:
                    return None
                
                best_c = -1
                best_cost = float('inf')
                
                for c in range(5):
                    cost = 0
                    # Check w such that edge(u,w)==c and common[c][w][v]==1
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if color[u][w] == c:
                            if common[c][w][v] >= 1:
                                cost += 1
                        if color[v][w] == c:
                            if common[c][w][u] >= 1:
                                cost += 1
                        if cost >= best_cost:
                            break
                            
                    if cost < best_cost:
                        best_cost = cost
                        best_c = c
                        
                if best_c == -1:
                    return None # Should not happen
                
                # If best_cost > 0, we are creating a conflict. 
                # In a strict greedy valid extension, we would fail here.
                # But we can allow some conflicts and repair later?
                # The directive says "minimize the creation of new forbidden pairs to naturally avoid completing any C4".
                # If we allow cost > 0, we create C4s.
                # Let's stick to cost == 0 if possible. If not, pick min cost and hope repair fixes it?
                # Or, if min cost > 0, restart?
                # Given n=26 is tight, restarts might be frequent.
                
                # Let's try to pick best_c.
                c = best_c
                
                # Update state
                color[u][v] = c
                color[v][u] = c
                
                # Update common counts
                # For each w:
                #   if color[u][w] == c:
                #       common[c][w][v] += 1
                #       common[c][v][w] += 1
                #   if color[v][w] == c:
                #       common[c][w][u] += 1
                #       common[c][u][w] += 1
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if color[u][w] == c:
                        common[c][w][v] += 1
                        common[c][v][w] += 1
                    if color[v][w] == c:
                        common[c][w][u] += 1
                        common[c][u][w] += 1
                        
            # After construction, verify validity
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        if common[c][u][v] >= 2:
                            return None
                            
            # Extract result
            res_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    res_colors.append(color[i][j])
            return res_colors

        # Try greedy construction for n=26
        # Use multiple restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            t_start = time.time()
            t_limit = min(deadline - 1.0, t_start + 2.0) # 2s per attempt
            res = greedy_construct(26, t_limit)
            if res is not None:
                return {'n': 26, 'colors': res}
        
        # Fallback to local search for n=26 if greedy failed
        # (Re-using the logic from the current program but optimized)
        def find_coloring_ls(n, time_limit):
            # Random init
            colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            edge_color = [[-1] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = i * (n - 1) - i * (i + 1) // 2 + j - i - 1
                    c = colors[idx]
                    edge_color[i][j] = c
                    edge_color[j][i] = c

            common = [[[0] * n for _ in range(n)] for _ in range(5)]
            for u in range(n):
                for v in range(u + 1, n):
                    for c in range(5):
                        count = 0
                        for w in range(n):
                            if w != u and w != v:
                                if edge_color[u][w] == c and edge_color[v][w] == c:
                                    count += 1
                        common[c][u][v] = count
                        common[c][v][u] = count

            def is_valid():
                for c in range(5):
                    for u in range(n):
                        for v in range(u + 1, n):
                            if common[c][u][v] >= 2:
                                return False
                return True

            def get_conflict():
                conflicts = []
                for c in range(5):
                    for u in range(n):
                        for v in range(u + 1, n):
                            if common[c][u][v] >= 2:
                                conflicts.append((u, v, c))
                if not conflicts:
                    return None
                return random.choice(conflicts)

            def flip_edge(u, v, c_new):
                c_old = edge_color[u][v]
                if c_old == c_new:
                    return
                
                edge_color[u][v] = c_new
                edge_color[v][u] = c_new
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if edge_color[u][w] == c_old:
                        common[c_old][w][v] -= 1
                        common[c_old][v][w] -= 1
                    if edge_color[v][w] == c_old:
                        common[c_old][w][u] -= 1
                        common[c_old][u][w] -= 1
                            
                    if edge_color[u][w] == c_new:
                        common[c_new][w][v] += 1
                        common[c_new][v][w] += 1
                    if edge_color[v][w] == c_new:
                        common[c_new][w][u] += 1
                        common[c_new][u][w] += 1

            max_iter = 50000
            for _ in range(max_iter):
                if time.time() > time_limit:
                    break
                if is_valid():
                    res_colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            res_colors.append(edge_color[i][j])
                    return res_colors
                
                conflict = get_conflict()
                if conflict is None:
                    res_colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            res_colors.append(edge_color[i][j])
                    return res_colors
                
                u, v, c = conflict
                new_c = random.choice([x for x in range(5) if x != c])
                flip_edge(u, v, new_c)
            
            return None

        # Try LS for n=26 with remaining time
        if time.time() < deadline - 5:
            res = find_coloring_ls(26, deadline - 2)
            if res is not None:
                return {'n': 26, 'colors': res}
                
        # Fallback n=11
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    return get_lazebnik_woldar()
