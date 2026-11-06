import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def verify(n, edges):
        """
        Verify the graph is C4-free and has min degree >= n-39.
        """
        if n < 1 or n > 47:
            return False
        target_deg = max(0, n - 39)
        
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n or u == v:
                return False
            adj[u, v] = 1
            adj[v, u] = 1
        
        degrees = adj.sum(axis=1)
        if np.min(degrees) < target_deg:
            return False
            
        # Check C4-free: A2[i,j] >= 2 implies C4
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
            
        return True

    def generate_paley_graph(n):
        """
        Generate a Paley graph on n vertices if n is a prime power q ≡ 1 (mod 4).
        For n=49, q=49 is a prime power.
        Vertices are elements of GF(q). Edge (u,v) exists if u-v is a quadratic residue.
        Paley graphs are C4-free? 
        Actually, Paley graphs often contain C4s. 
        Wait, the task says "C4-free". 
        Let's check: A Paley graph P(q) has a C4 if there exist distinct a,b,c,d such that a-b, b-c, c-d, d-a are all residues.
        It is known that Paley graphs contain C4s for most q.
        However, the *complement* of a Paley graph might be useful, or we just use it as a strong seed.
        
        Alternative deterministic seed: 
        For n=46 or 47, we can use a random seed but bias it towards a regular structure, 
        or just use the Paley graph on the largest prime power <= n and then prune/repair.
        
        Let's implement a proper GF(49) construction for the seed.
        GF(49) = GF(7^2). Elements are pairs (a, b) with a,b in GF(7).
        Multiplication: (a,b)*(c,d) = (ac - bd, ad + bc) if we use x^2+1=0 (since 1 is not a square mod 7? 
        Squares mod 7: 1, 4, 2, 2, 4, 1. 0,1,2,4. Non-squares: 3,5,6.
        x^2+1=0 => x^2=6. 6 is non-square. So yes, GF(49) is defined by x^2+1=0.
        
        Quadratic Residues in GF(49):
        An element z is a QR if z = w^2 for some w.
        In GF(p^2), the group of non-zero elements is cyclic of order p^2-1 = 48.
        The squares form a subgroup of order 24.
        
        Let's just build the adjacency matrix for GF(49).
        """
        if n != 49:
            return None
            
        # Represent GF(49) elements as 0..48
        # 0 is (0,0)
        # 1..48 are non-zero elements.
        # We need a generator of the multiplicative group.
        # Let alpha = (0, 1). alpha^2 = -1 = (6, 0).
        # alpha has order 4? alpha^2 = -1, alpha^4 = 1. So alpha is order 4.
        # We need an element of order 48.
        # Let's find a primitive element.
        # Try (1, 1). (1,1)^2 = (0, 2). (1,1)^3 = (1,1)(0,2) = (-2, 1) = (5, 1).
        # This is tedious to hardcode. 
        # Since n=46 or 47 is the target, and 49 is slightly larger, 
        # we can generate the Paley graph on 49 vertices, remove 2 or 3 vertices, 
        # and then repair the degree of the remaining vertices.
        
        # Let's implement the GF(49) arithmetic properly.
        def gf49_add(a, b):
            # a, b are (x, y)
            return ((a[0] + b[0]) % 7, (a[1] + b[1]) % 7)
        
        def gf49_mul(a, b):
            # (a1 + a2 * alpha) * (b1 + b2 * alpha)
            # = a1b1 + a1b2*alpha + a2b1*alpha + a2b2*alpha^2
            # alpha^2 = -1 = 6
            # = (a1b1 + 6*a2b2) + (a1b2 + a2b1)*alpha
            return ((a[0]*b[0] + 6*a[1]*b[1]) % 7, (a[0]*b[1] + a[1]*b[0]) % 7)
        
        def gf49_pow(a, p):
            res = (1, 0)
            base = a
            while p:
                if p & 1:
                    res = gf49_mul(res, base)
                base = gf49_mul(base, base)
                p >>= 1
            return res
        
        # Find primitive element
        # The multiplicative group has order 48.
        # Factors of 48: 2, 3, 4, 6, 8, 12, 16, 24, 48.
        # A generator g must satisfy g^(48/p) != 1 for all prime factors p.
        # Prime factors of 48 are 2 and 3.
        # So g^24 != 1 and g^16 != 1.
        
        # Let's try (1, 1)
        g = (1, 1)
        if gf49_pow(g, 24) != (1, 0) and gf49_pow(g, 16) != (1, 0):
            primitive = g
        else:
            # Try (1, 2)
            g = (1, 2)
            if gf49_pow(g, 24) != (1, 0) and gf49_pow(g, 16) != (1, 0):
                primitive = g
            else:
                # Fallback: just pick any non-1, non-0 and hope, or scan
                for i in range(1, 7):
                    for j in range(1, 7):
                        g = (i, j)
                        if gf49_pow(g, 24) != (1, 0) and gf49_pow(g, 16) != (1, 0):
                            primitive = g
                            break
                    else:
                        continue
                    break
        
        # Map elements to 0..48
        # 0 is (0,0) -> index 0
        # Non-zero elements are powers of primitive: primitive^k for k=1..48
        # Let's list them
        elements = [(0,0)]
        curr = (1, 0)
        for _ in range(48):
            curr = gf49_mul(curr, primitive)
            elements.append(curr)
            
        # Now determine which are quadratic residues.
        # In a finite field of odd characteristic, x is a QR if x^((q-1)/2) = 1.
        # (q-1)/2 = 24.
        is_qr = [False] * 49
        for i, elem in enumerate(elements):
            if elem == (0,0):
                continue
            if gf49_pow(elem, 24) == (1, 0):
                is_qr[i] = True
                
        # Build adjacency
        adj = np.zeros((49, 49), dtype=np.int8)
        for i in range(49):
            for j in range(i+1, 49):
                # Check if elements[i] - elements[j] is a QR
                diff = gf49_add(elements[i], (7 - elements[j][0]) % 7, (7 - elements[j][1]) % 7)
                # Find index of diff
                # Since elements is a list, we can make a dict
                pass # This is slow. Let's precompute a lookup.
        
        # Precompute lookup
        elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        
        for i in range(49):
            for j in range(i+1, 49):
                diff = gf49_add(elements[i], (7 - elements[j][0]) % 7, (7 - elements[j][1]) % 7)
                idx = elem_to_idx[diff]
                if is_qr[idx]:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def local_search(n, adj, target_deg, time_limit):
        """
        Incremental local search to satisfy min degree >= target_deg while maintaining C4-free.
        State: adj matrix, neighbor sets.
        Move: Add an edge (u, v) if it doesn't create a C4.
        Priority: Add edges to vertices with lowest degree.
        """
        start_ls = time.time()
        
        # Initialize neighbor sets for O(1) common neighbor check
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    
        degrees = np.array([len(neigh) for neigh in neighbors])
        
        # Check initial C4-freeness (should be true if from Paley, but verify)
        # Actually, Paley graphs on q=49 might have C4s? 
        # It is a known result that Paley graphs are C4-free iff q is a prime power such that... 
        # Wait, Paley graphs are strongly regular. 
        # Parameters: v=q, k=(q-1)/2, lambda=(q-5)/4, mu=(q-1)/4.
        # If lambda >= 2, there are C4s? 
        # A C4 exists if there are two vertices with 2 common neighbors.
        # In a strongly regular graph, any two adjacent vertices have lambda common neighbors.
        # Any two non-adjacent have mu common neighbors.
        # If lambda >= 2 or mu >= 2, there is a C4.
        # For q=49, lambda = (49-5)/4 = 11. So Paley(49) has MANY C4s.
        # So we CANNOT use Paley(49) directly as a C4-free graph.
        
        # STRATEGY PIVOT:
        # The Paley graph is a good *seed* for the structure, but it violates C4-free.
        # We must REMOVE edges to break all C4s.
        # This is a "C4-hitting set" problem.
        # Given a graph with many C4s, remove minimum edges to make it C4-free.
        # This is NP-hard, but for n=46, we can use a greedy heuristic:
        # 1. While C4 exists: find a C4, remove an edge from it.
        # 2. Then repair degrees by adding edges that don't create C4s.
        
        # However, removing edges from Paley(49) to make it C4-free might leave degrees too low.
        # Paley(49) degree is 24. Target is 7 (for n=46).
        # We need to keep ~7 edges per vertex.
        # So we need to remove ~17 edges per vertex.
        # This is a massive pruning.
        
        # Alternative: 
        # Use a random graph G(n, p) with p ~ 7/46 ~ 0.15.
        # This graph will have some C4s.
        # Greedily remove edges from C4s.
        # Then add edges to boost degrees.
        
        # This is essentially what the current program does, but "incremental" and "greedy repair" 
        # is more efficient than full SA if implemented with sets.
        
        # Let's implement the "Greedy Repair" with Sets.
        
        def find_c4():
            """Find a C4 in the current graph. Returns (u, v, w, x) or None."""
            # Efficient check: 
            # For each pair (u, v), count common neighbors.
            # If count >= 2, we have a C4 (or a triangle+chord, but simple graph so just C4 or K4).
            # Actually, if u and v share 2 neighbors a, b, then u-a-v-b-u is a C4.
            # We can just return u, v, a, b.
            
            # To be fast, iterate over pairs.
            # n=46, pairs ~ 1000.
            # Common neighbors check: intersection of two sets.
            # 1000 * (set intersection) is fast.
            
            for u in range(n):
                for v in range(u+1, n):
                    common = neighbors[u].intersection(neighbors[v])
                    if len(common) >= 2:
                        # Pick two
                        common_list = list(common)
                        a = common_list[0]
                        b = common_list[1]
                        return (u, v, a, b)
            return None

        def is_safe_add(u, v):
            """Check if adding (u, v) creates a C4."""
            # If u and v share 2 or more neighbors, adding (u,v) creates a C4?
            # Wait. If u and v already share 2 neighbors, the graph ALREADY has a C4 (u-a-v-b-u).
            # So we assume the current graph is C4-free.
            # If the graph is C4-free, no pair of vertices shares 2 neighbors.
            # Therefore, for any u, v, |N(u) intersect N(v)| <= 1.
            # If we add (u, v), does it create a new C4?
            # A new C4 would involve the new edge (u, v).
            # The cycle must be u - v - x - y - u.
            # This requires x to be a neighbor of v and y to be a neighbor of u, and x-y to be an edge.
            # Also x != y.
            # Essentially, we need to check if there exists x in N(v) and y in N(u) such that x-y is an edge.
            # Wait, the cycle is u-v-x-y-u.
            # Edges: (u,v) [new], (v,x) [exists], (x,y) [exists], (y,u) [exists].
            # So we need to check if there is an edge between N(v) and N(u) that is not just the shared neighbor (if any).
            # Actually, if N(u) and N(v) share a neighbor w, then u-w-v is a path of length 2.
            # Adding (u,v) creates a triangle u-v-w-u. Not a C4.
            # A C4 requires 4 distinct vertices.
            # So we check: Is there an edge (x, y) where x in N(v), y in N(u), x != y, and x != u, y != v (trivial)?
            # And x, y are distinct from u, v.
            # And x != y.
            
            # Optimization:
            # If N(u) and N(v) are disjoint, we check all edges between N(u) and N(v).
            # If they share one vertex w, we exclude w.
            
            # Since n is small, just check:
            # For each x in N(v):
            #   For each y in N(u):
            #     if x == y: continue
            #     if adj[x, y]: return False (unsafe)
            
            # This is O(deg(u) * deg(v)). Max 46*46 ~ 2000 ops. Very fast.
            
            for x in neighbors[v]:
                if x == u: continue
                for y in neighbors[u]:
                    if x == y: continue
                    if adj[x, y]:
                        return False
            return True

        # Main Loop
        # Phase 1: Remove C4s
        # We start with a random graph or a modified Paley?
        # Let's start with a random graph G(n, p) where p is tuned.
        # If we start with Paley(49) pruned to n, it's too complex.
        # Let's start with a random graph of density ~ target_deg / (n-1).
        
        # Actually, the "Inspiration 1" mentioned "polarity graphs".
        # Let's stick to Random + Repair as it's robust.
        
        # Initialize
        p = target_deg / (n - 1)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # Phase 1: Break C4s
        steps = 0
        max_steps = 5000
        while steps < max_steps:
            if time.time() - start_ls > time_limit:
                break
            c4 = find_c4()
            if not c4:
                break
            u, v, a, b = c4
            # Remove one of the edges: (u,a), (u,b), (v,a), (v,b)
            # Choose the edge whose removal minimizes degree drop?
            # Or just random.
            # Let's remove (u, a)
            adj[u, a] = 0
            adj[a, u] = 0
            neighbors[u].remove(a)
            neighbors[a].remove(u)
            steps += 1
            
        # Phase 2: Boost Degrees
        # Now graph is C4-free.
        # Check degrees.
        for _ in range(10000):
            if time.time() - start_ls > time_limit:
                break
            
            degrees = np.array([len(neigh) for neigh in neighbors])
            if np.min(degrees) >= target_deg:
                break
                
            # Find vertex with min degree
            min_idx = np.argmin(degrees)
            u = min_idx
            
            # Try to add an edge to u
            # Candidates: v not in N(u) and v != u
            candidates = [v for v in range(n) if v != u and v not in neighbors[u]]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if is_safe_add(u, v):
                    # Add edge
                    adj[u, v] = 1
                    adj[v, u] = 1
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    added = True
                    break
            
            if not added:
                # Stuck? 
                # Try to remove an edge incident to a high-degree vertex to free up space?
                # Or just restart? 
                # For now, break and let verification fail, triggering a new random seed.
                break
                
    # Main Logic
    # Try n=47, 46, etc.
    # For n=47, target_deg = 8.
    # For n=46, target_deg = 7.
    
    # We will try multiple random restarts for each n.
    
    for n in [47, 46, 45, 44]:
        if time.time() - start_time > deadline - 5:
            break
            
        target_deg = max(0, n - 39)
        
        # Try a few times
        for attempt in range(20):
            if time.time() - start_time > deadline - 2:
                break
                
            adj = np.zeros((n, n), dtype=np.int8)
            
            # Run local search
            # Note: local_search modifies adj in place (conceptually, we pass a copy or let it init)
            # My local_search above initializes adj itself. Let's refactor to pass adj.
            
            # Re-implementing local_search to take adj as input for flexibility
            def run_ls(adj_in):
                neighbors = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i+1, n):
                        if adj_in[i, j]:
                            neighbors[i].add(j)
                            neighbors[j].add(i)
                            
                def find_c4():
                    for u in range(n):
                        for v in range(u+1, n):
                            common = neighbors[u].intersection(neighbors[v])
                            if len(common) >= 2:
                                common_list = list(common)
                                return (u, v, common_list[0], common_list[1])
                    return None

                def is_safe_add(u, v):
                    for x in neighbors[v]:
                        if x == u: continue
                        for y in neighbors[u]:
                            if x == y: continue
                            if adj_in[x, y]:
                                return False
                    return True

                # Phase 1: Remove C4s
                for _ in range(5000):
                    if time.time() - start_time > deadline - 2:
                        return False
                    c4 = find_c4()
                    if not c4:
                        break
                    u, v, a, b = c4
                    # Remove (u, a)
                    adj_in[u, a] = 0
                    adj_in[a, u] = 0
                    neighbors[u].remove(a)
                    neighbors[a].remove(u)
                    
                # Phase 2: Boost
                for _ in range(5000):
                    if time.time() - start_time > deadline - 2:
                        return False
                    degrees = np.array([len(neigh) for neigh in neighbors])
                    if np.min(degrees) >= target_deg:
                        return True
                    
                    min_idx = np.argmin(degrees)
                    u = min_idx
                    candidates = [v for v in range(n) if v != u and v not in neighbors[u]]
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        if is_safe_add(u, v):
                            adj_in[u, v] = 1
                            adj_in[v, u] = 1
                            neighbors[u].add(v)
                            neighbors[v].add(u)
                            break
                            
                return np.min([len(neigh) for neigh in neighbors]) >= target_deg

            # Initialize random adj
            p = target_deg / (n - 1)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        
            if run_ls(adj):
                # Verify
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
                if verify(n, edges):
                    return {'n': n, 'edges': edges}
                    
    # Fallback
    return {'n': 39, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
