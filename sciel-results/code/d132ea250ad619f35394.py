import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe

    def check_triangle_free(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def max_independent_set_size(adj, n):
        # Branch and bound for max independent set
        # For small n (13 or 19), this is fast.
        max_size = 0
        used = [False] * n
        
        def backtrack(current_size, candidates):
            nonlocal max_size
            if not candidates:
                if current_size > max_size:
                    max_size = current_size
            else:
                # Pruning: if current_size + len(candidates) <= max_size, return
                if current_size + len(candidates) <= max_size:
                    return
                
                # Pick a vertex
                v = candidates[0]
                rest = candidates[1:]
                
                # Option 1: Include v
                new_candidates = [u for u in rest if not adj[v][u]]
                backtrack(current_size + 1, new_candidates)
                
                # Option 2: Exclude v
                backtrack(current_size, rest)

        backtrack(0, list(range(n)))
        return max_size

    def try_find_base_graph(n_vertices, max_alpha_allowed):
        """
        Find a triangle-free graph K on n_vertices with alpha(K) <= max_alpha_allowed.
        """
        # Randomized search
        for _ in range(1000):
            if time.time() > deadline:
                return None
            
            # Generate a random triangle-free graph
            # Start with empty graph, add edges if they don't create a triangle
            adj = [[False] * n_vertices for _ in range(n_vertices)]
            
            # Randomly shuffle edge order
            edges = [(i, j) for i in range(n_vertices) for j in range(i + 1, n_vertices)]
            random.shuffle(edges)
            
            for u, v in edges:
                # Check if adding (u, v) creates a triangle
                # Triangle exists if there is w such that (u,w) and (v,w) are edges
                creates_triangle = False
                for w in range(n_vertices):
                    if w != u and w != v:
                        if adj[u][w] and adj[v][w]:
                            creates_triangle = True
                            break
                            
                if not creates_triangle:
                    # Add edge with some probability to keep it sparse enough for low alpha
                    # Actually, we want to maximize edges to minimize alpha?
                    # No, adding edges reduces alpha.
                    # So we should try to add as many edges as possible while staying triangle-free.
                    # This is the maximum triangle-free graph (Mantel).
                    # But we need alpha <= limit.
                    # Let's just add all possible edges (Maximal Triangle-Free).
                    adj[u][v] = True
                    adj[v][u] = True
            
            # Check alpha
            alpha = max_independent_set_size(adj, n_vertices)
            if alpha <= max_alpha_allowed:
                return adj
        
        return None

    def build_graph_from_base(K_adj, m):
        """
        Construct G from base graph K.
        G = complement of BlowUp(complement(K), m)
        Which is equivalent to:
        Vertices are (k, i) where k in K, i in 0..m-1.
        (k1, i1) ~ (k2, i2) in G iff:
          if k1 == k2: i1 != i2 (Clique in complement? No.)
          
        Let's re-derive the adjacency of G.
        H = complement(K).
        \bar{G} = BlowUp(H, m).
        G = complement(\bar{G}).
        
        In \bar{G}:
        (h1, i1) ~ (h2, i2) iff h1 ~ h2 in H OR (h1 == h2 AND i1 == i2? No, simple graph).
        BlowUp definition:
        Vertices V' = V(H) x {0..m-1}.
        (h1, i1) ~ (h2, i2) in \bar{G} iff:
          1. h1 ~ h2 in H, OR
          2. h1 == h2 (and i1 != i2? No, cliques are complete subgraphs).
             Actually, standard blow-up replaces each vertex h with a clique of size m.
             So (h, i) ~ (h, j) for all i != j.
             And (h1, i) ~ (h2, j) if h1 ~ h2 in H.
             
        So in G = complement(\bar{G}):
        (k1, i1) ~ (k2, i2) in G iff NOT ((k1, i1) ~ (k2, i2) in \bar{G}).
        
        Condition for NO edge in G:
        1. k1 == k2 AND i1 != i2 (Edge in \bar{G} clique part)
           Wait, if k1 == k2, then h1 == h2.
           In \bar{G}, (h, i) ~ (h, j) for i != j.
           So in G, (k, i) NOT~ (k, j) for i != j.
           And (k, i) ~ (k, i) is not an edge (simple graph).
           
        2. k1 != k2 AND k1 ~ k2 in H.
           Since H = complement(K), k1 ~ k2 in H iff NOT (k1 ~ k2 in K).
           So in \bar{G}, (k1, i) ~ (k2, j) if k1 not ~ k2 in K.
           So in G, (k1, i) NOT~ (k2, j) if k1 not ~ k2 in K.
           
        So, (k1, i1) ~ (k2, i2) in G iff:
        - k1 != k2 AND k1 ~ k2 in K.
        
        Wait, what if k1 == k2?
        If k1 == k2, then in \bar{G}, they are connected (clique).
        So in G, they are NOT connected.
        
        So G is a "blow-up" of K where each vertex is replaced by an INDEPENDENT set of size m?
        Let's check.
        If we replace each vertex k in K with an independent set of size m, and connect all vertices in set k1 to all vertices in set k2 if k1 ~ k2 in K.
        This is the standard blow-up of K with independent sets.
        
        Let's verify properties of this G:
        1. Triangle-free?
           If G has a triangle, it must be either:
           a) 3 vertices from same set: No, independent.
           b) 2 from set k1, 1 from set k2: No, k1 is independent.
           c) 1 from k1, 1 from k2, 1 from k3:
              Requires k1~k2, k2~k3, k3~k1 in K.
              So G is triangle-free iff K is triangle-free.
              
        2. Alpha(G)?
           An independent set in G can pick vertices from sets k1, k2, ...
           If we pick from set k1 and set k2, they must be non-adjacent in G.
           (k1, i) NOT~ (k2, j) in G iff NOT (k1 ~ k2 in K).
           So k1 and k2 must be non-adjacent in K.
           So an independent set in G corresponds to an independent set in K.
           For each k in the independent set of K, we can pick AT MOST 1 vertex from the set of size m?
           No, we can pick ALL m vertices from a set k, because they are independent in G.
           So if S is an independent set in K, then we can pick all m vertices from each k in S.
           Size = m * |S|.
           So alpha(G) = m * alpha(K).
           
        This matches my previous derivation:
        alpha(G) = m * omega(H) where H=comp(K).
        omega(H) = alpha(K).
        So alpha(G) = m * alpha(K).
        
        So we need:
        1. K is triangle-free.
        2. m * alpha(K) <= 9.
        
        This confirms the logic.
        
        So, for n=39:
        m=3, k=13.
        Need K on 13 vertices, triangle-free, alpha(K) <= 3.
        
        For n=38:
        m=2, k=19.
        Need K on 19 vertices, triangle-free, alpha(K) <= 4.
        """
        n_k = len(K_adj[0])
        n_total = n_k * m
        edges = []
        # Map (k, i) to vertex id
        # id = k * m + i
        for k1 in range(n_k):
            for k2 in range(k1 + 1, n_k):
                if K_adj[k1][k2]:
                    # Connect all i in k1 to all j in k2
                    for i in range(m):
                        for j in range(m):
                            u = k1 * m + i
                            v = k2 * m + j
                            edges.append([u, v])
        return {'n': n_total, 'edges': edges}

    # 1. Try n=39
    K13 = try_find_base_graph(13, 3)
    if K13:
        return build_graph_from_base(K13, 3)

    # 2. Try n=38
    K19 = try_find_base_graph(19, 4)
    if K19:
        return build_graph_from_base(K19, 2)

    # 3. Fallback to n=26 (Andrasfai)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
