import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety
    
    n = 18
    best_score = 18
    best_edges = []
    
    # Helper: Get edges from adjacency matrix
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Helper: Check if graph is K4-free using bitmask or simple check
    def is_k4_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    continue
                for w in range(v + 1, n):
                    if not adj[u][w] or not adj[v][w]:
                        continue
                    # u,v,w form a triangle. Check if any x completes K4
                    # x must be connected to u, v, w
                    # x can be < u, between u and v, between v and w, or > w
                    # Actually, just check all x != u,v,w
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[u][x] and adj[v][x] and adj[w][x]:
                            return False
        return True

    # Helper: Find maximum triangle-free induced subgraph size
    # This is the expensive part. We use a simple branch and bound or exhaustive check for small n.
    # Since n=18, C(18,9) is 48620, which is small. But we want the MAX size.
    # We can check sizes from n down to 1.
    # Optimization: If we find a triangle-free set of size k, we don't need to check smaller ones if we are looking for max.
    # But we want to MINIMIZE this max size. So we want to find the true max for the current graph.
    
    def max_triangle_free_size(adj):
        # Represent graph as bitmasks for faster triangle checks?
        # Or just use the matrix.
        # For n=18, checking all subsets of size k is feasible if k is small or we prune well.
        # Let's check k from n down to 1.
        for k in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), k):
                # Check if this subset induces a triangle-free graph
                is_tf = True
                # Convert subset to list for indexing
                verts = list(subset)
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        v = verts[j]
                        if adj[u][v]:
                            # Check if there is a w in subset such that u-v-w is a triangle
                            # Actually, if adj[u][v] is true, we just need to check if any other w in subset connects to both u and v.
                            # If no such w exists, this edge is fine. But we need the WHOLE subgraph to be triangle free.
                            # So if adj[u][v] is true, we must ensure no w in subset (w!=u,v) has adj[u][w] and adj[v][w].
                            for l in range(len(verts)):
                                if l == i or l == j:
                                    continue
                                w = verts[l]
                                if adj[u][w] and adj[v][w]:
                                    is_tf = False
                                    break
                            if not is_tf:
                                break
                    if not is_tf:
                        break
                if is_tf:
                    return k
        return 0

    # Initial Construction: Clebsch Graph on 16 vertices + 2 extra
    # Clebsch graph vertices can be represented as {0,1}^4.
    # Two vertices are adjacent if they differ in exactly 2 bits.
    # This graph is (16, 5, 0, 2) strongly regular.
    # It is K4-free? Max clique size is 2 (triangle free? No, lambda=0 means no triangles).
    # Wait, lambda=0 means adjacent vertices have 0 common neighbors. So it is triangle-free!
    # If the Clebsch graph is triangle-free, then the induced subgraph on all 16 vertices is triangle-free.
    # So max_triangle_free_size would be at least 16.
    # We need to minimize this. So we want to ADD edges to break up the large triangle-free sets,
    # while keeping the graph K4-free.
    
    # Let's start with the Clebsch graph on vertices 0..15.
    # Vertices 16, 17 are isolated initially? Or connected to some?
    # If 16, 17 are isolated, they can be added to any triangle-free set.
    # So we should connect them to break up sets.
    
    def generate_clebsch_adj():
        adj = [[0]*n for _ in range(n)]
        # Clebsch on 0..15
        for u in range(16):
            for v in range(u+1, 16):
                # Check Hamming distance
                diff = u ^ v
                # Count bits
                if bin(diff).count('1') == 2:
                    adj[u][v] = 1
                    adj[v][u] = 1
        # Vertices 16, 17
        # Connect them to some vertices to avoid them being in large TF sets.
        # Let's connect 16 to 0,1,2,3,4 and 17 to 5,6,7,8,9.
        # This is arbitrary, local search will fix it.
        for i in range(5):
            adj[16][i] = 1
            adj[i][16] = 1
        for i in range(5, 10):
            adj[17][i] = 1
            adj[i][17] = 1
        # Edge 16-17?
        adj[16][17] = 1
        adj[17][16] = 1
        return adj

    # Local Search
    # We want to minimize max_triangle_free_size.
    # Current graph: adj
    # Move: Flip an edge (u,v).
    # Constraint: Graph must remain K4-free.
    
    def flip_edge(adj, u, v):
        new_adj = [row[:] for row in adj]
        new_adj[u][v] ^= 1
        new_adj[v][u] = new_adj[u][v]
        return new_adj

    def score_graph(adj):
        # Check K4-free
        if not is_k4_free(adj):
            return None
        # Calculate max TF size
        return max_triangle_free_size(adj)

    # Start with Clebsch + extensions
    adj = generate_clebsch_adj()
    current_score = score_graph(adj)
    if current_score is not None:
        best_score = current_score
        best_edges = get_edges_from_adj(adj)
    else:
        # Should not happen if construction is valid, but just in case
        # Try random
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        # Remove K4s
        # Simple removal: while K4 exists, remove random edge from a K4
        changed = True
        while changed:
            changed = False
            for quad in itertools.combinations(range(n), 4):
                u,v,w,x = quad
                if adj[u][v] and adj[u][w] and adj[u][x] and adj[v][w] and adj[v][x] and adj[w][x]:
                    # Remove random edge
                    e = random.choice([(u,v),(u,w),(u,x),(v,w),(v,x),(w,x)])
                    adj[e[0]][e[1]] = 0
                    adj[e[1]][e[0]] = 0
                    changed = True
                    break
        current_score = score_graph(adj)
        if current_score is not None:
            best_score = current_score
            best_edges = get_edges_from_adj(adj)
        else:
            return []

    # Local Search Loop
    # We perform random restarts or just one long local search?
    # Given time limit, let's do a few restarts with different initializations if time permits.
    # But local search on a good initial guess might be enough.
    
    # To be safe, let's try a few different random initializations + local search.
    # But the Clebsch one is structured. Let's stick to improving that one first.
    
    # Actually, let's mix:
    # 1. Clebsch based
    # 2. Random dense + K4 removal
    # 3. Random sparse + K4 removal
    
    strategies = [
        "clebsch",
        "random_dense",
        "random_sparse"
    ]
    
    for strat in strategies:
        if time.time() > deadline - 10:
            break
            
        if strat == "clebsch":
            adj = generate_clebsch_adj()
        elif strat == "random_dense":
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.6:
                        adj[i][j] = 1
                        adj[j][i] = 1
            # Remove K4s
            changed = True
            while changed:
                changed = False
                for quad in itertools.combinations(range(n), 4):
                    u,v,w,x = quad
                    if adj[u][v] and adj[u][w] and adj[u][x] and adj[v][w] and adj[v][x] and adj[w][x]:
                        e = random.choice([(u,v),(u,w),(u,x),(v,w),(v,x),(w,x)])
                        adj[e[0]][e[1]] = 0
                        adj[e[1]][e[0]] = 0
                        changed = True
                        break
        elif strat == "random_sparse":
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.4:
                        adj[i][j] = 1
                        adj[j][i] = 1
            # Remove K4s
            changed = True
            while changed:
                changed = False
                for quad in itertools.combinations(range(n), 4):
                    u,v,w,x = quad
                    if adj[u][v] and adj[u][w] and adj[u][x] and adj[v][w] and adj[v][x] and adj[w][x]:
                        e = random.choice([(u,v),(u,w),(u,x),(v,w),(v,x),(w,x)])
                        adj[e[0]][e[1]] = 0
                        adj[e[1]][e[0]] = 0
                        changed = True
                        break
        
        current_score = score_graph(adj)
        if current_score is None:
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_edges = get_edges_from_adj(adj)
            
        # Local Search
        # Try to improve current graph
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try flipping random edges
            # We need to be careful: flipping an edge might create a K4.
            # We will try a flip, check if K4-free, then score.
            # To speed up, we can check K4-free only for the neighborhood of u,v.
            
            # Let's just pick a random edge to flip
            u, v = random.sample(range(n), 2)
            new_adj = flip_edge(adj, u, v)
            
            # Check K4-free
            if not is_k4_free(new_adj):
                continue
                
            new_score = score_graph(new_adj)
            if new_score is not None:
                if new_score < current_score:
                    current_score = new_score
                    adj = new_adj
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = get_edges_from_adj(adj)
                # Accept equal scores with some probability to escape local minima?
                # Or just hill climbing. Hill climbing is safer for time.
                # Let's do simulated annealing style? No, keep it simple.
                # If new_score == current_score, accept with prob 0.1 to move around.
                elif new_score == current_score and random.random() < 0.1:
                    adj = new_adj
                    # Don't mark as improved, but move
                    # Actually, if we move, we should continue searching from new state.
                    # But we didn't improve score.
                    # Let's just do strict improvement for now to avoid cycling.
                    pass

    return best_edges
