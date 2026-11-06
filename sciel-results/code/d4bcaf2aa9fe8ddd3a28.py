"""Combined SA with fast bitmask K4-free check and triangle-free B&B via hitting set."""
import random
import time
import itertools

N = 18
ALL_MASK = (1 << N) - 1

def _build_adj(adj_list):
    """Build adjacency bitmask list from edge list or return existing."""
    adj = [0] * N
    for u, v in adj_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free(adj):
    """Check K4-free using adjacency bitmasks. For each triangle (i,j,k),
    check if N(i)&N(j)&N(k) has any vertex (which would form K4)."""
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj[i] >> j) & 1:
                continue
            common = adj[i] & adj[j]
            # For each k in common (k > j to avoid duplicates), check triangle i,j,k
            # then check if that triangle extends to K4
            k_mask = common
            while k_mask:
                lsb = k_mask & (-k_mask)
                k = lsb.bit_length() - 1
                if k > j:
                    # i, j, k form a triangle (i-j, i-k, j-k all edges)
                    # Check if there's a 4th vertex connected to all three
                    common_ijk = adj[i] & adj[j] & adj[k]
                    if common_ijk:
                        return False
                k_mask ^= lsb
    return True

def _max_triangle_free_size(adj):
    """Exact B&B: largest triangle-free induced subgraph.
    Equivalent to n - min_vertex_cover_of_triangles."""
    # Find all triangles
    triangles = []
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj[i] >> j) & 1:
                continue
            common = adj[i] & adj[j]
            k_mask = common
            while k_mask:
                lsb = k_mask & (-k_mask)
                k = lsb.bit_length() - 1
                if k > j:
                    triangles.append((i, j, k))
                k_mask ^= lsb

    if not triangles:
        return N

    # B&B for minimum vertex cover of triangles (hitting set)
    best_cover = [N]
    
    # Precompute for each triangle its bitmask
    tri_masks = []
    for t in triangles:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        tri_masks.append(m)
    
    n_tris = len(triangles)
    
    # Sort triangles by frequency of vertices (most constrained first)
    vert_freq = [0] * N
    for t in triangles:
        for v in t:
            vert_freq[v] += 1
    
    def _branch(tri_idx, covered_mask, cover_size):
        if cover_size >= best_cover[0]:
            return
        if tri_idx >= n_tris:
            if cover_size < best_cover[0]:
                best_cover[0] = cover_size
            return
        
        # Check if this triangle is already hit
        tm = tri_masks[tri_idx]
        if covered_mask & tm:
            _branch(tri_idx + 1, covered_mask, cover_size)
            return
        
        # Pruning: count remaining unhit triangles
        remaining = 0
        for ti in range(tri_idx, n_tris):
            if not (covered_mask & tri_masks[ti]):
                remaining += 1
        if cover_size + remaining >= best_cover[0]:
            return
        
        # Branch on each vertex in this triangle
        i, j, k = triangles[tri_idx]
        # Order by frequency (most frequent first for better pruning)
        verts = sorted([i, j, k], key=lambda v: -vert_freq[v])
        for v in verts:
            _branch(tri_idx + 1, covered_mask | (1 << v), cover_size + 1)
    
    _branch(0, 0, 0)
    return N - best_cover[0]

def _greedy_triangle_free(adj, trials=20):
    """Quick greedy estimate of max triangle-free subgraph size."""
    best = 0
    for _ in range(trials):
        order = list(range(N))
        random.shuffle(order)
        selected = 0
        for v in order:
            # Check if adding v creates a triangle with any two already-selected vertices
            common = selected & adj[v]
            ok = True
            cm = common
            while cm:
                lsb = cm & (-cm)
                a = lsb.bit_length() - 1
                rest = common & ~(1 << a)
                if rest & adj[a]:
                    ok = False
                    break
                cm ^= lsb
            if ok:
                selected |= (1 << v)
        sz = bin(selected).count('1')
        if sz > best:
            best = sz
    return best

def _generate_random_k4_free(density=0.5):
    """Generate a random K4-free graph by adding edges one at a time."""
    adj = [0] * N
    edges = list(itertools.combinations(range(N), 2))
    random.shuffle(edges)
    for u, v in edges:
        if random.random() >= density:
            continue
        # Check if adding (u,v) creates a K4
        common = adj[u] & adj[v]
        has_k4 = False
        cm = common
        verts = []
        while cm:
            lsb = cm & (-cm)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            cm ^= lsb
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if adj[verts[i]] >> verts[j] & 1:
                    has_k4 = True
                    break
            if has_k4:
                break
        if not has_k4:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _generate_structured(density=0.5):
    """Generate a structured K4-free graph (e.g., based on modular arithmetic)."""
    adj = [0] * N
    # Use a random permutation and a structured pattern
    perm = list(range(N))
    random.shuffle(perm)
    for i in range(N):
        for j in range(i + 1, N):
            a, b = perm[i], perm[j]
            # Structured: connect if (a-b) mod N is in some set
            diff = abs(a - b)
            if random.random() < density:
                # Check K4-free
                common = adj[a] & adj[b]
                has_k4 = False
                cm = common
                verts = []
                while cm:
                    lsb = cm & (-cm)
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    cm ^= lsb
                for x in range(len(verts)):
                    for y in range(x + 1, len(verts)):
                        if adj[verts[x]] >> verts[y] & 1:
                            has_k4 = True
                            break
                    if has_k4:
                        break
                if not has_k4:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    return adj

def _adj_to_edges(adj):
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u] >> v & 1:
                edges.append([u, v])
    return edges

def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # Multi-start: try several initial graphs
    best_adj = None
    best_score = N + 1
    
    # Try different densities
    for density in [0.4, 0.5, 0.6, 0.55]:
        if time.time() > deadline:
            break
        adj = _generate_random_k4_free(density)
        score = _max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    # Also try structured
    if time.time() < deadline:
        adj = _generate_structured(0.5)
        score = _max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    if best_adj is None:
        best_adj = _generate_random_k4_free(0.5)
        best_score = _max_triangle_free_size(best_adj)
    
    # Simulated Annealing
    current_adj = best_adj[:]
    current_score = best_score
    T = 3.0
    T_min = 0.005
    cooling = 0.9998
    
    all_pairs = list(itertools.combinations(range(N), 2))
    
    # Track recent scores for adaptive cooling
    no_improve = 0
    
    while time.time() < deadline and T > T_min:
        u, v = random.choice(all_pairs)
        
        # Toggle edge
        current_adj[u] ^= (1 << v)
        current_adj[v] ^= (1 << u)
        
        if not _is_k4_free(current_adj):
            # Revert
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            T *= cooling
            continue
        
        # Quick greedy filter
        greedy = _greedy_triangle_free(current_adj, trials=15)
        if greedy >= best_score:
            # Revert
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            T *= cooling
            continue
        
        # Exact evaluation
        new_score = _max_triangle_free_size(current_adj)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                current_score = new_score
                no_improve += 1
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                no_improve += 1
        
        T *= cooling
        
        # Adaptive: if stuck for a while, cool faster or restart
        if no_improve > 5000:
            T *= 0.95
            no_improve = 0
    
    return _adj_to_edges(best_adj)
