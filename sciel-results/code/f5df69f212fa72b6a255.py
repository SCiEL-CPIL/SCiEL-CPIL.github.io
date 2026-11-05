import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety
    
    n = 18
    
    # Helper: Get edges from adjacency list (list of bitmasks)
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # Helper: Check if adding edge (u, v) creates a K4
    # A K4 is created if there exists a pair (w, x) such that:
    # w, x are common neighbors of u and v
    # and w, x are adjacent
    def is_k4_free_after_add(adj, u, v):
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        # Check if the subgraph induced by 'common' contains an edge
        # If it does, adding (u,v) creates a K4 with that edge's endpoints
        w = common
        while w:
            lsb = w & -w
            w ^= lsb
            wi = lsb.bit_length() - 1
            if adj[wi] & (common & ~(1 << wi)):
                return False
        return True

    # Helper: Find maximum triangle-free induced subgraph size
    # Uses bitmask Branch and Bound
    def max_triangle_free_size(adj):
        best = 0
        
        def dfs(current_set, candidates):
            nonlocal best
            if not candidates:
                if current_set.bit_count() > best:
                    best = current_set.bit_count()
                return
            
            # Upper bound
            if current_set.bit_count() + candidates.bit_count() <= best:
                return
            
            # Pick a vertex from candidates to branch on
            # Heuristic: pick the vertex with the lowest index for determinism, 
            # or potentially one with higher degree in candidates for better pruning.
            # Let's stick to lowest index for speed in Python loop overhead.
            v = (candidates & -candidates).bit_length() - 1
            v_mask = 1 << v
            
            # Branch 1: Exclude v
            dfs(current_set, candidates & ~v_mask)
            
            # Branch 2: Include v
            # Check if v can be added to current_set without creating a triangle.
            neighbors_in_current = adj[v] & current_set
            
            valid = True
            check = neighbors_in_current
            while check:
                lsb = check & -check
                check ^= lsb
                ui = lsb.bit_length() - 1
                if adj[ui] & (neighbors_in_current & ~lsb):
                    valid = False
                    break
            
            if valid:
                new_current = current_set | v_mask
                # Update candidates: remove v and any w that forms a triangle with v and some u in current_set
                # w is forbidden if w is neighbor of v AND w is neighbor of some u in neighbors_in_current
                forbidden = 0
                check_u = neighbors_in_current
                while check_u:
                    lsb_u = check_u & -check_u
                    check_u ^= lsb_u
                    ui = lsb_u.bit_length() - 1
                    forbidden |= adj[ui]
                    
                forbidden &= adj[v]
                
                new_candidates = (candidates & ~v_mask) & ~forbidden
                dfs(new_current, new_candidates)
                
        dfs(0, (1 << n) - 1)
        return best

    # Initial Construction: C9 U C9
    # Two disjoint cycles of length 9.
    # Vertices 0-8 form a cycle, 9-17 form a cycle.
    adj = [0] * n
    
    # Cycle 0-8
    for i in range(9):
        u = i
        v = (i + 1) % 9
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Cycle 9-17
    for i in range(9):
        u = 9 + i
        v = 9 + ((i + 1) % 9)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # This graph is K4-free (girth 9, max clique size 2).
    # Largest triangle-free induced subgraph is the whole graph (size 18) because it's triangle-free itself.
    # Wait, the task asks to MINIMIZE the largest triangle-free induced subgraph.
    # C9 U C9 is triangle-free, so the largest triangle-free induced subgraph has size 18.
    # This is a bad starting point if we want to minimize the size.
    # We want a graph with many triangles, but no K4s, such that any large set of vertices contains a triangle.
    
    # Better initial construction:
    # Start with a dense K4-free graph.
    # The Turan graph T_3(18) (complete tripartite) is K4-free.
    # Its largest triangle-free induced subgraph:
    # In T_3(18), a triangle-free set can pick vertices from at most 2 parts?
    # No, if you pick vertices from 3 parts, say u in P1, v in P2, w in P3, they form a triangle.
    # So a triangle-free set in T_3(18) can only contain vertices from 2 parts.
    # Max size = max(|P1|+|P2|, |P2|+|P3|, |P1|+|P3|).
    # For balanced T_3(18), parts are size 6. Max size = 12.
    # So T_3(18) gives score 12.
    # Can we do better?
    # If we start with T_3(18) and remove edges, we create triangles? No, removing edges destroys triangles.
    # Adding edges to T_3(18) creates K4s immediately?
    # In T_3(18), any edge added within a part creates a K4 with any edge between the other two parts?
    # Let u, v in P1. Add edge (u, v). Pick w in P2, x in P3.
    # u, v, w, x form a K4 if all edges exist.
    # (u, v) is new. (u, w), (u, x), (v, w), (v, x) exist in T_3. (w, x) exists in T_3.
    # Yes, adding any intra-part edge to T_3 creates a K4.
    # So T_3(18) is a local maximum for edge density in K4-free graphs.
    # To reduce the triangle-free subgraph size, we need to "break up" the large bipartite structures.
    # But we can't add edges. We must remove edges?
    # If we remove edges from T_3(18), we destroy triangles. The graph becomes sparser.
    # The largest triangle-free subgraph might get LARGER (closer to 18) if we remove enough edges to kill all triangles.
    # We want to keep triangles but ensure no large triangle-free set exists.
    
    # Actually, the strategy of "Simulated Annealing on edge flips" starting from T_3(18) is tricky because:
    # 1. Adding edges is forbidden (creates K4).
    # 2. Removing edges destroys triangles, potentially increasing the max triangle-free set.
    
    # Let's rethink.
    # The score is the size of the LARGEST triangle-free induced subgraph.
    # We want this to be small.
    # This implies the graph should be "triangle-dense" in a way that forces any large subset to contain a triangle.
    # This is related to the Erdos-Rogers function f(n, k, l).
    # Here we want to minimize the largest induced subgraph with no triangles (K3-free).
    # Let S be a triangle-free set. We want max |S| to be small.
    
    # If we start with T_3(18), max triangle-free set is 12 (union of two parts).
    # Can we get it lower?
    # If we take a graph that is "closer" to a complete graph but K4-free...
    # The only K4-free graphs that are very dense are Turan graphs.
    # However, we can modify T_3(18).
    # What if we start with T_3(18) and perform "edge removals" that are "strategic"?
    # No, removing edges from T_3(18) only helps if it breaks the "bipartite" nature of the triangle-free sets?
    # Wait. In T_3(18), the triangle-free sets are exactly the subsets of vertices that do not contain a triangle.
    # Since T_3(18) is 3-partite, any triangle uses one vertex from each part.
    # A set S is triangle-free if it does not contain a triple (u, v, w) with u in P1, v in P2, w in P3.
    # This is equivalent to saying that the bipartite graph between any two parts restricted to S is not "complete" in a way that allows a third vertex?
    # Actually, simpler: S is triangle-free in T_3(18) iff S is not "3-partite complete".
    # If S contains vertices from all 3 parts, does it necessarily contain a triangle?
    # Yes, if S contains u in P1, v in P2, w in P3, then (u, v), (v, w), (w, u) are all edges in T_3(18).
    # So ANY set containing vertices from all 3 parts is NOT triangle-free.
    # Therefore, a triangle-free set in T_3(18) can only use vertices from at most 2 parts.
    # Max size = max(|P1 \cap S| + |P2 \cap S|, ...).
    # To minimize the max size of such a set, we want the parts to be as small as possible?
    # But sum of sizes is 18.
    # Minimize max(A+B, B+C, A+C) subject to A+B+C=18.
    # This is minimized when A=B=C=6, giving 12.
    # If we make parts uneven, say 10, 4, 4.
    # Max(10+4, 4+4, 10+4) = 14. Worse.
    # So T_3(18) with balanced parts is optimal among Turan graphs.
    
    # Can we do better than 12?
    # We need a K4-free graph where every set of 13 vertices contains a triangle.
    # This is equivalent to saying the independence number of the "triangle graph" is small?
    # Or, the graph has no large induced K3-free subgraph.
    
    # Let's try a different family.
    # What if we start with a random K4-free graph?
    # Or better, start with T_3(18) and try to "perturb" it?
    # Since we can't add edges, we can only remove.
    # Removing edges from T_3(18) will make the graph sparser.
    # Does removing edges from T_3(18) reduce the max triangle-free set?
    # No. If we remove an edge (u, v) where u in P1, v in P2, then the set S = P1 \cup P2 is still triangle-free?
    # Wait. In T_3(18), P1 \cup P2 is triangle-free because there are no edges within P1 or P2.
    # Removing edges between P1 and P2 does NOT introduce edges within P1 or P2.
    # So P1 \cup P2 remains triangle-free.
    # Thus, removing edges from T_3(18) CANNOT reduce the size of the largest triangle-free induced subgraph below 12.
    # In fact, it might increase it if we remove enough edges to allow mixing of parts?
    # No, the vertices in P1 \cup P2 never form triangles regardless of inter-part edges, because there are no intra-part edges.
    # So any subgraph of T_3(18) has P1 \cup P2 as a triangle-free induced subgraph.
    # So the score is always >= 12 for any subgraph of T_3(18).
    
    # This implies T_3(18) is a "dead end" if we only remove edges.
    # We MUST start with a different graph or allow edge additions (which requires careful K4 checking).
    
    # Strategy: Start with a sparser K4-free graph, e.g., a random one or a specific construction, and ADD edges.
    # Adding edges creates triangles, which might shrink the triangle-free sets.
    # We must ensure no K4 is formed.
    
    # Let's start with a sparse K4-free graph.
    # E.g., a random graph with p=0.3, then remove K4s?
    # Or start with C9 U C9 (score 18) and add edges.
    # Adding edges to C9 U C9:
    # We can add edges within the cycles or between cycles.
    # Adding an edge within C9 creates a triangle?
    # C9 has no triangles. Adding a chord creates triangles.
    # This increases triangle density.
    # We want to maximize triangle density while keeping K4-free.
    
    # Let's use Simulated Annealing on Edge Additions/Removals starting from a medium density K4-free graph.
    # Initial: Random K4-free graph.
    # Generate random graph, remove K4s iteratively.
    
    def generate_random_k4_free():
        adj = [0]*n
        # Random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Remove K4s
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        continue
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Check for edge in common
                    w = common
                    found_edge = False
                    while w:
                        lsb = w & -w
                        w ^= lsb
                        wi = lsb.bit_length() - 1
                        if adj[wi] & (common & ~(1 << wi)):
                            found_edge = True
                            break
                    if found_edge:
                        # Remove edge (u, v) to break this K4
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        changed = True
        return adj
        
    # Start with T_3(18) is bad for search because we can't add edges.
    # Start with a random K4-free graph.
    adj = generate_random_k4_free()
    
    current_score = max_triangle_free_size(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # Simulated Annealing
    # Moves: Add edge (u, v) if K4-free, or Remove edge (u, v).
    # Prefer adding edges to increase triangle density.
    
    temp = 1.0
    min_temp = 0.01
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        
        is_edge = (adj[u] >> v) & 1
        
        if is_edge:
            # Try removing
            new_adj = adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = max_triangle_free_size(new_adj)
        else:
            # Try adding
            if not is_k4_free_after_add(adj, u, v):
                continue
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_score = max_triangle_free_size(new_adj)
            
        if new_score is None:
            continue
            
        delta = new_score - current_score
        
        if new_score < best_score:
            best_score = new_score
            best_adj = new_adj
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Acceptance
        if delta < 0:
            accept = True
        else:
            if delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < (2.71828 ** (-delta / temp))
                
        if accept:
            adj = new_adj
            current_score = new_score
            
        temp = max(min_temp, temp * 0.99)
        
        # Restart if stuck
        if no_improve_count > 500:
            # Random restart or perturb
            adj = generate_random_k4_free()
            current_score = max_triangle_free_size(adj)
            temp = 1.0
            no_improve_count = 0
            
    return get_edges_from_adj(best_adj)

# Fallback if construct fails or returns empty
if __name__ == "__main__":
    pass
