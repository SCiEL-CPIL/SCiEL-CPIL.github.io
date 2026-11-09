import random
import itertools

def construct():
    n = 20
    # Start with a dense random graph
    # We want to maximize edges, so start with p=0.7 or regular
    # Let's use a random regular graph of degree 10 if possible, else random
    # For simplicity and speed, use random p=0.6
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Generate initial graph
    # To ensure we start high, let's try p=0.7
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.7:
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append((u, v))
                
    # Function to count common neighbors of u, v, w
    def common_neighbors(u, v, w):
        count = 0
        for x in range(n):
            if adj[u][x] and adj[v][x] and adj[w][x]:
                count += 1
        return count

    # Find all bad triples
    def get_bad_triples():
        bad = []
        for u, v, w in itertools.combinations(range(n), 3):
            if common_neighbors(u, v, w) > 2:
                bad.append((u, v, w))
        return bad

    # Iterate to remove edges
    # To speed up, we can recompute bad triples each iteration
    # n=20 is small, 1140 triples.
    # Max edges 190. Iterations < 190.
    # 190 * 1140 * 20 ops ~ 4M ops. Very fast.
    
    # We will remove the edge that appears in the most "excess" common neighbor slots
    # Actually, simpler: Find a bad triple, pick a random edge from its common neighbor connections to remove?
    # Better: Find the edge (u,v) that is "most central" to the violations.
    
    # Let's use a simple greedy:
    # 1. Find all bad triples.
    # 2. If none, break.
    # 3. For each bad triple, the "cost" of keeping the graph is 1.
    #    We need to remove an edge incident to one of the common neighbors.
    #    Let's pick the edge that is in the most bad triples' "neighborhoods".
    
    # To keep it simple and robust:
    # While there are bad triples:
    #   Pick a random bad triple (u,v,w).
    #   Find all common neighbors X.
    #   Pick a random x in X.
    #   Pick a random vertex from {u,v,w} to disconnect from x?
    #   Removing (u,x) reduces count by 1.
    #   This might remove too many edges.
    
    # Better:
    # Calculate for each edge (i,j) how many bad triples it "supports".
    # An edge (i,j) supports a bad triple (u,v,w) if j is a common neighbor of u,v,w and i is one of u,v,w?
    # No, if j is a common neighbor, then edges (u,j), (v,j), (w,j) exist.
    # Removing ANY of these edges breaks the common neighbor status of j for that triple.
    # So edge (u,j) is associated with triple (u,v,w) if j in N(u) cap N(v) cap N(w).
    
    # Let's implement this scoring.
    
    while True:
        bad_triples = get_bad_triples()
        if not bad_triples:
            break
            
        # Score each edge
        edge_scores = {}
        for u, v, w in bad_triples:
            # Find common neighbors
            common = []
            for x in range(n):
                if adj[u][x] and adj[v][x] and adj[w][x]:
                    common.append(x)
            
            # For each common neighbor x, the edges (u,x), (v,x), (w,x) are candidates
            # We add 1 to the score of these edges
            for x in common:
                for y in (u, v, w):
                    if y != x: # Ensure no self loops, though adj[y][y] is 0
                        edge = (min(y, x), max(y, x))
                        edge_scores[edge] = edge_scores.get(edge, 0) + 1
        
        if not edge_scores:
            break
            
        # Find the edge with the highest score
        # Tie-break randomly
        max_score = max(edge_scores.values())
        candidates = [e for e, s in edge_scores.items() if s == max_score]
        edge_to_remove = random.choice(candidates)
        
        u, v = edge_to_remove
        adj[u][v] = 0
        adj[v][u] = 0
        if (u, v) in edges:
            edges.remove((u, v))
        elif (v, u) in edges:
            edges.remove((v, u))

    # Convert to list of lists
    res = [[u, v] for u, v in edges]
    return res
