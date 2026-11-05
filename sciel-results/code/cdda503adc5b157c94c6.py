"""C4-free graph with high minimum degree via greedy construction."""

def construct():
    """
    Construct a C4-free graph on as many vertices as possible with minimum degree >= n - 39.
    
    Strategy:
    1. Start with the incidence graph of PG(2,4) (42 vertices, 5-regular, C4-free).
    2. Try to add vertices one by one, connecting each new vertex to a maximum set of 
       existing vertices such that no C4 is created.
    3. The key constraint: for a new vertex v connected to vertices in set S, no two 
       vertices in S can have a common neighbor (other than v itself), because that 
       would create a C4: new_v - u - w - x - new_v where u,x in S and w is common neighbor.
       
    Actually, the C4 constraint for adding a new vertex v adjacent to S is:
    For any two vertices u, w in S, they must NOT share a common neighbor in the existing graph.
    Because if u and w share neighbor x, then v-u-x-w-v forms a C4.
    
    So we need to find a set S of existing vertices where no two share a common neighbor.
    This is equivalent to: the closed neighborhoods of vertices in S are pairwise disjoint 
    (except possibly at the vertices themselves... wait, let me think more carefully).
    
    If v is adjacent to u and w, and u and w share a common neighbor x (x != u, x != w), 
    then v-u-x-w-v is a C4. So we need: for all distinct u, w in S, N(u) ∩ N(w) = ∅.
    
    Wait, that's too strong. Let me reconsider.
    v-u-x-w-v: this is a 4-cycle if v≠x, u≠w, u≠x, w≠v. Since v is new, v≠x. 
    u≠w since distinct. u≠x since x is a neighbor of u. w≠v since v is new.
    So yes, if u and w are both in S and share a common neighbor x, we get a C4.
    
    So the condition is: for all distinct u, w in S, N(u) ∩ N(w) = ∅.
    
    In PG(2,4) incidence graph:
    - Two points: they share a line (the line through them). So N(point1) ∩ N(point2) contains that line.
    - Two lines: they share a point (the intersection). So N(line1) ∩ N(line2) contains that point.
    - A point and a line: if the point is on the line, they share... wait.
      N(point) = set of lines through the point. N(line) = set of points on the line.
      These are disjoint sets (one is points, one is lines in the bipartite sense).
      So N(point) ∩ N(line) = ∅ always!
      
    So in the incidence graph, any point and any line have disjoint neighborhoods.
    But two points share a line, two lines share a point.
    
    So a valid set S can contain:
    - At most 1 point (since any two points share a line)
    - At most 1 line (since any two lines share a point)
    - But a point and a line can both be in S (their neighborhoods are disjoint).
    
    So |S| ≤ 2, with at most 1 point and at most 1 line.
    
    This means we can add at most 1 vertex of degree 2 to the PG(2,4) incidence graph.
    For n=43, we need min degree >= 4, so degree 2 is not enough.
    
    Let me try a different approach: use a different base graph or a more sophisticated construction.
    
    Alternative: Use a polarity graph or a different algebraic construction.
    
    Let me try: Start with n=46 or n=47 and use a greedy/backtracking approach to build a C4-free graph.
    
    Actually, let me try a completely different approach: 
    Use the fact that we need a C4-free graph with min degree >= n-39.
    For n=46, min degree >= 7.
    For n=47, min degree >= 8.
    
    Let me try a random greedy approach:
    1. Start with n vertices.
    2. Greedily add edges while maintaining C4-freeness and trying to achieve min degree >= n-39.
    
    Or better: use a known construction. The Erdos-Renyi type construction or using finite fields.
    
    Let me try: For q = 4, the polarity graph of PG(2,4) has 21 vertices. Not enough.
    
    Let me try a different strategy: Use a bipartite graph construction where I can control degrees.
    
    Actually, let me try the following: 
    Take the complete bipartite graph K_{a,b} and remove edges to make it C4-free.
    K_{a,b} is C4-free iff a*b <= 3 (trivially) or... no, K_{2,2} is a C4.
    So K_{a,b} is C4-free iff a<=1 or b<=1. Not useful.
    
    Let me try: Start with a 5-regular C4-free graph on 42 vertices (PG(2,4) incidence),
    and try to add 4-5 more vertices with appropriate connections.
    
    Wait, I showed that I can only add vertices of degree at most 2. So that won't work for n=46.
    
    New idea: Use a different base. What about the incidence graph of a different structure?
    
    For q=3: PG(2,3) has 13 points and 13 lines, n=26, degree 4.
    For q=4: n=42, degree 5.
    For q=5: PG(2,5) has 31 points and 31 lines, n=62, degree 6. But n=62 > 47.
    
    So PG(2,5) is too large. But maybe I can take a subset?
    
    Let me try: Take PG(2,5) incidence graph (62 vertices, 6-regular) and find a large induced subgraph 
    that is still C4-free with min degree >= n-39.
    
    For n=46, need min degree >= 7. But PG(2,5) is 6-regular, so induced subgraphs have degree <= 6. Not enough.
    
    For n=42, need min degree >= 3. PG(2,5) induced subgraph of 42 vertices with min degree >= 3.
    But we already have n=42 with degree 5, which is better.
    
    Let me try yet another approach: Use a greedy algorithm to build a C4-free graph with high minimum degree.
    
    Algorithm:
    1. Start with n vertices, no edges.
    2. For each pair (u,v), decide whether to add edge (u,v) based on whether it creates a C4.
    3. Try to maximize the minimum degree.
    
    This is a combinatorial optimization problem. Let me use a simple greedy:
    - Order vertices randomly.
    - For each vertex, connect it to as many other vertices as possible without creating a C4.
    
    Let me implement this and see what n we can achieve.
    
    Actually, the key insight for C4-free graphs: if we have a graph G, adding edge (u,v) creates a C4 
    iff u and v have a common neighbor. So we can add (u,v) iff N(u) ∩ N(v) = ∅.
    
    Greedy algorithm:
    1. Start with n vertices, empty graph.
    2. For each vertex v in some order, connect v to all other vertices u such that N(v) ∩ N(u) = ∅.
    3. The order matters. Try different orders.
    
    Let me implement this with a fixed n and see what min degree we get.
    """
    import random
    from itertools import combinations
    
    def is_c4_free(adj, u, v):
        """Check if adding edge (u,v) to graph with adjacency sets adj creates a C4."""
        # Adding (u,v) creates a C4 iff there exists w such that w is adjacent to both u and v.
        # i.e., N(u) ∩ N(v) != empty
        if adj[u] & adj[v]:
            return False
        return True
    
    def greedy_c4_free(n, order):
        """Build a C4-free graph on n vertices using greedy edge addition in given order."""
        adj = [set() for _ in range(n)]
        # Process vertices in given order
        for v in order:
            # Connect v to all previously processed vertices u where no C4 is created
            for u in order:
                if u == v:
                    continue
                if u not in adj[v]:  # not already connected
                    if not (adj[u] & adj[v]):
                        adj[u].add(v)
                        adj[v].add(u)
        return adj
    
    def min_degree(adj):
        return min(len(s) for s in adj)
    
    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    # Try to find the best n
    best_result = None
    best_n = 0
    
    # Try n from 46 down to 42
    for n in range(46, 41, -1):
        required_deg = n - 39
        found = False
        # Try multiple random orders
        for trial in range(50):
            order = list(range(n))
            random.shuffle(order)
            adj = greedy_c4_free(n, order)
            md = min_degree(adj)
            if md >= required_deg:
                edges = to_edges(adj, n)
                return {'n': n, 'edges': edges}
                found = True
                break
        if found:
            break
    
    # Fallback: use PG(2,4) incidence graph
    q = 4
    n_points = q*q + q + 1
    n_lines = q*q + q + 1
    n = n_points + n_lines
    
    add = [
        [0,1,2,3],
        [1,0,3,2],
        [2,3,0,1],
        [3,2,1,0]
    ]
    mul = [
        [0,0,0,0],
        [0,1,2,3],
        [0,2,3,1],
        [0,3,1,2]
    ]
    
    def add_vec(v1, v2):
        return [add[v1[i]][v2[i]] for i in range(3)]
    
    def mul_vec(v, c):
        return [mul[c][v[i]] for i in range(3)]
        
    def is_zero(v):
        return all(x == 0 for x in v)
        
    points = []
    seen_points = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                v = [x,y,z]
                if is_zero(v): continue
                idx = -1
                for i in range(3):
                    if v[i] != 0:
                        idx = i
                        break
                if idx == -1: continue
                inv = [0,1,3,2]
                c = inv[v[idx]]
                canon = mul_vec(v, c)
                key = tuple(canon)
                if key not in seen_points:
                    seen_points.add(key)
                    points.append(canon)
                    
    lines = points
    
    edges = []
    for i, p in enumerate(points):
        for j, l in enumerate(lines):
            dot = 0
            for k in range(3):
                dot = add[dot][mul[p[k]][l[k]]]
            if dot == 0:
                edges.append([i, n_points + j])
                
    return {'n': n, 'edges': edges}
