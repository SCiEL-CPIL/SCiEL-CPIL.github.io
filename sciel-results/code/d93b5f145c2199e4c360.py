import random
import time
import sys

N = 18
DEADLINE = 115.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    def bit_count(x):
        return bin(x).count('1')
    
    def get_vertices(mask):
        verts = []
        for i in range(N):
            if mask & (1 << i):
                verts.append(i)
        return verts

    def build_triangles(adj):
        """Return list of triangle bitmasks."""
        triangles = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    k_mask = common & ~((1 << i) | (1 << j))
                    while k_mask:
                        lsb = k_mask & -k_mask
                        k = lsb.bit_length() - 1
                        if k > j:
                            triangles.append((1 << i) | (1 << j) | (1 << k))
                        k_mask ^= lsb
        return triangles

    def largest_triangle_free(adj):
        """Exact B&B for largest triangle-free induced subgraph size."""
        triangles = build_triangles(adj)
        if not triangles:
            return N
        
        best = 0
        # Precompute for each vertex: which triangles it belongs to
        vert_tris = [[] for _ in range(N)]
        for ti, tri in enumerate(triangles):
            for v in range(N):
                if tri & (1 << v):
                    vert_tris[v].append(ti)
        
        # Sort vertices: put high-degree-in-triangles first for better pruning
        tri_deg = [len(vert_tris[v]) for v in range(N)]
        # Also consider graph degree
        graph_deg = [bit_count(adj[v]) for v in range(N)]
        
        def bnb(remaining, current_mask, current_size):
            nonlocal best
            if current_size + bit_count(remaining) <= best:
                return
            if current_size >= 3:
                for tri in triangles:
                    if (current_mask & tri) == tri:
                        return
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick vertex with max (tri_deg + graph_deg) in remaining
            v = -1
            max_score = -1
            temp_rem = remaining
            while temp_rem:
                lsb = temp_rem & -temp_rem
                u = lsb.bit_length() - 1
                score = tri_deg[u] * 3 + graph_deg[u]
                if score > max_score:
                    max_score = score
                    v = u
                temp_rem ^= lsb
            
            if v == -1:
                return

            # Try include first
            neigh_in_curr = adj[v] & current_mask
            valid_include = True
            temp_neigh = neigh_in_curr
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                u = lsb.bit_length() - 1
                if adj[u] & neigh_in_curr:
                    valid_include = False
                    break
                temp_neigh ^= lsb
            
            if valid_include:
                bnb(remaining ^ (1 << v), current_mask | (1 << v), current_size + 1)
            
            bnb(remaining ^ (1 << v), current_mask, current_size)

        bnb((1 << N) - 1, 0, 0)
        return best

    def get_largest_tf_set(adj):
        """Returns bitmask of a largest triangle-free induced subgraph."""
        triangles = build_triangles(adj)
        if not triangles:
            return (1 << N) - 1
        
        best_mask = 0
        best_size = 0
        
        tri_deg = [0] * N
        for tri in triangles:
            for v in range(N):
                if tri & (1 << v):
                    tri_deg[v] += 1
        graph_deg = [bit_count(adj[v]) for v in range(N)]
        
        def bnb(remaining, current_mask, current_size):
            nonlocal best_mask, best_size
            if current_size + bit_count(remaining) <= best_size:
                return
            if current_size >= 3:
                for tri in triangles:
                    if (current_mask & tri) == tri:
                        return
            if remaining == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            v = -1
            max_score = -1
            temp_rem = remaining
            while temp_rem:
                lsb = temp_rem & -temp_rem
                u = lsb.bit_length() - 1
                score = tri_deg[u] * 3 + graph_deg[u]
                if score > max_score:
                    max_score = score
                    v = u
                temp_rem ^= lsb
            
            if v == -1:
                return

            neigh_in_curr = adj[v] & current_mask
            valid_include = True
            temp_neigh = neigh_in_curr
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                u = lsb.bit_length() - 1
                if adj[u] & neigh_in_curr:
                    valid_include = False
                    break
                temp_neigh ^= lsb
            
            if valid_include:
                bnb(remaining ^ (1 << v), current_mask | (1 << v), current_size + 1)
            
            bnb(remaining ^ (1 << v), current_mask, current_size)

        bnb((1 << N) - 1, 0, 0)
        return best_mask

    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False
        common = adj[u] & adj[v]
        common &= ~((1 << u) | (1 << v))
        temp = common
        while temp:
            lsb = temp & -temp
            w1 = lsb.bit_length() - 1
            if adj[w1] & common:
                return False
            temp ^= lsb
        return True

    def adj_to_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def k4_free_check(adj):
        """Verify graph is K4-free."""
        triangles = build_triangles(adj)
        for tri in triangles:
            # Get the three vertices
            verts = []
            for v in range(N):
                if tri & (1 << v):
                    verts.append(v)
            i, j, k = verts[0], verts[1], verts[2]
            common = adj[i] & adj[j] & adj[k]
            common &= ~((1 << i) | (1 << j) | (1 << k))
            if common != 0:
                return False
        return True

    def spectral_init():
        """
        Spectral-inspired initialization:
        Use a continuous relaxation to guide edge placement.
        We want to maximize triangle density (overlapping triangles) while keeping K4-free.
        
        Approach: Start with K_{6,6,6}, then use a score-based method to add edges
        that create the most triangles per edge (maximizing triangle overlap).
        """
        # Start with complete tripartite K_{6,6,6}
        adj = [0] * N
        parts = [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17]]
        
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Now try to add intra-part edges to create more triangles
        # Score: for each potential edge (u,v), count how many triangles it creates
        # A triangle is created if there exists w adjacent to both u and v
        potential_edges = []
        for i in range(3):
            for a in range(len(parts[i])):
                for b in range(a+1, len(parts[i])):
                    u, v = parts[i][a], parts[i][b]
                    if not (adj[u] & (1 << v)):
                        # Count common neighbors
                        common = adj[u] & adj[v]
                        common &= ~((1 << u) | (1 << v))
                        score = bit_count(common)
                        potential_edges.append((score, u, v))
        
        # Sort by score descending
        potential_edges.sort(key=lambda x: -x[0])
        
        # Greedily add edges that maintain K4-free
        added = 0
        for score, u, v in potential_edges:
            if score <= 0:
                break
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
                if added >= 12:  # Don't add too many
                    break
        
        return adj

    def random_k4_free_dense():
        """Generate a random dense K4-free graph."""
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, current_score):
        """
        Local search: try edge additions within TF set, edge swaps, and perturbations.
        Returns (best_adj, best_score, improved).
        """
        best_adj = adj[:]
        best_score = current_score
        improved = True
        
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            
            # Find largest TF set
            tf_mask = get_largest_tf_set(adj)
            tf_vertices = get_vertices(tf_mask)
            
            if len(tf_vertices) < 2:
                break
            
            # Strategy 1: Add edges within TF set
            random.shuffle(tf_vertices)
            for i in range(len(tf_vertices)):
                if time.time() - start_time > DEADLINE:
                    break
                for j in range(i + 1, len(tf_vertices)):
                    if time.time() - start_time > DEADLINE:
                        break
                    u = tf_vertices[i]
                    v = tf_vertices[j]
                    
                    if adj[u] & (1 << v):
                        continue
                    if not can_add_edge(adj, u, v):
                        continue
                    
                    # Tentatively add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    new_score = largest_triangle_free(adj)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj[:]
                        improved = True
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Strategy 2: Edge swap - add an edge, remove another
            # Try adding an edge that creates triangles, and removing an edge that "wastes" structure
            all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)
                          if not (adj[i] & (1 << j)) and can_add_edge(adj, i, j)]
            all_existing = [(i, j) for i in range(N) for j in range(i + 1, N)
                          if (adj[i] & (1 << j))]
            
            random.shuffle(all_possible)
            random.shuffle(all_existing)
            
            for add_u, add_v in all_possible[:50]:
                if time.time() - start_time > DEADLINE:
                    break
                for rem_u, rem_v in all_existing[:30]:
                    if time.time() - start_time > DEADLINE:
                        break
                    if add_u == rem_u or add_u == rem_v or add_v == rem_u or add_v == rem_v:
                        continue
                    
                    # Try: add (add_u, add_v), remove (rem_u, rem_v)
                    adj[add_u] |= (1 << add_v)
                    adj[add_v] |= (1 << add_u)
                    adj[rem_u] ^= (1 << rem_v)
                    adj[rem_v] ^= (1 << rem_u)
                    
                    new_score = largest_triangle_free(adj)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj[:]
                        improved = True
                    else:
                        # Revert
                        adj[add_u] ^= (1 << add_v)
                        adj[add_v] ^= (1 << add_u)
                        adj[rem_u] |= (1 << rem_v)
                        adj[rem_v] |= (1 << rem_u)
                    
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Strategy 3: Random perturbation
            all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.shuffle(all_edges)
            perturbed = False
            for u, v in all_edges:
                if time.time() - start_time > DEADLINE:
                    break
                if not (adj[u] & (1 << v)) and can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_score = largest_triangle_free(adj)
                    if new_score <= best_score:
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = adj[:]
                            improved = True
                        perturbed = True
                        break
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
            
            if not perturbed:
                break
        
        return best_adj, best_score, improved

    best_score = N
    best_adj = [0] * N
    
    # Initialize with spectral approach
    adj = spectral_init()
    current_score = largest_triangle_free(adj)
    if current_score < best_score:
        best_score = current_score
        best_adj = adj[:]
    
    # Local search from spectral init
    adj, current_score, _ = local_search(adj, current_score)
    if current_score < best_score:
        best_score = current_score
        best_adj = adj[:]
    
    # Multiple random restarts
    while time.time() - start_time < DEADLINE:
        adj = random_k4_free_dense()
        current_score = largest_triangle_free(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
        
        adj, current_score, _ = local_search(adj, current_score)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
    
    # Final: try to improve best_adj further
    if time.time() - start_time < DEADLINE - 5:
        adj = best_adj[:]
        current_score = best_score
        adj, current_score, _ = local_search(adj, current_score)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
    
    return adj_to_edges(best_adj)

if __name__ == "__main__":
    construct()
