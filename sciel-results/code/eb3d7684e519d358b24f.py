import time
import random
import math

N = 18
FULL = (1 << N) - 1
DEADLINE_MARGIN = 10  # safety margin before the 150 s wall-clock kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random(42)

    def is_k4_free(adj_mask):
        # Optimized K4 check
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        m &= m - 1
                        if (adj_mask[c] & m) != 0:
                            return False
        return True

    def max_tri_free(adj_mask):
        """Exact size of the largest triangle-free induced subgraph (branch & bound)."""
        # Find triangles
        tris = []
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        if c > b:
                            tris.append((a, b, c))
                        m &= m - 1
        
        if not tris:
            return N
            
        best = [0]
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in tris]
        
        # Sort triangles: prioritize triangles with vertices that have high degree in the graph
        # This helps find large independent sets faster
        tri_masks.sort(key=lambda tm: -((tm & -tm).bit_count() * 10 + (tm >> 1 & -1).bit_count() * 5 + (tm >> 2 & -1).bit_count() * 2))
        
        # Precompute for each vertex, which triangles it belongs to (for faster lookup)
        vertex_to_tris = {v: [] for v in range(N)}
        for i, tm in enumerate(tri_masks):
            m = tm
            while m:
                bit = m & -m
                v = bit.bit_length() - 1
                vertex_to_tris[v].append(i)
                m ^= bit
        
        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            
            # Find a triangle in avail - check triangles in sorted order
            tri = None
            for tm in tri_masks:
                if (tm & avail) == tm:
                    tri = tm
                    break
                    
            if tri is None:
                best[0] = cnt
                return
                
            a_bit = tri & -tri
            a = a_bit.bit_length() - 1
            rest = tri ^ a_bit
            b_bit = rest & -rest
            b = b_bit.bit_length() - 1
            c_bit = tri ^ a_bit ^ b_bit
            c = c_bit.bit_length() - 1
            
            # Heuristic: remove vertex with highest degree in avail
            da = (avail & adj_mask[a]).bit_count()
            db = (avail & adj_mask[b]).bit_count()
            dc = (avail & adj_mask[c]).bit_count()
            
            verts = sorted([(da, a), (db, b), (dc, c)], reverse=True)
            
            for _, v in verts:
                bnb(avail & ~(1 << v))
                if cnt - 1 <= best[0]:
                    return

        bnb(FULL)
        return best[0]

    def make_turan_3():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def make_paley_17_plus_isolated():
        """Paley(17) graph on vertices 0-16, vertex 17 isolated."""
        adj_mask = [0] * N
        # Paley(17): connect i,j if (j-i) mod 17 is a quadratic residue mod 17
        # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
        residues = set([1, 2, 4, 8, 9, 13, 15, 16])
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        # Vertex 17 is isolated
        return adj_mask

    def make_structured_cyclic():
        adj_mask = [0] * N
        offsets = [2, 3, 5, 7, 11, 13]
        for i in range(N):
            for off in offsets:
                j = (i + off) % N
                if j > i:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def flip(adj_mask, u, v):
        adj_mask[u] ^= (1 << v)
        adj_mask[v] ^= (1 << u)

    best_score = N + 1
    best_mask = None

    # Add Paley(17) + isolated as a strong seed
    seeds = []
    seeds.append(make_paley_17_plus_isolated())
    seeds.append(make_turan_3())
    seeds.append(make_structured_cyclic())
    
    for d in [0.4, 0.5, 0.6, 0.7, 0.8]:
        adj = [0]*N
        for i in range(N):
            for j in range(i+1, N):
                if rng.random() < d:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if is_k4_free(adj):
            seeds.append(adj)

    # Evaluate seeds first
    for seed in seeds:
        if time.time() >= deadline:
            break
        if not is_k4_free(seed):
            continue
        cur = max_tri_free(seed)
        if cur < best_score:
            best_score = cur
            best_mask = seed[:]

    # Local search from best seed
    if best_mask is not None:
        adj = best_mask[:]
        cur = best_score
        
        # Simulated Annealing
        temperature = 2.0
        cooling = 0.99
        steps = 0
        max_steps = 5000
        no_improve = 0
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            u, v = rng.sample(range(N), 2)
            flip(adj, u, v)
            
            if is_k4_free(adj):
                new_score = max_tri_free(adj)
                delta = new_score - cur
                
                if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 0.01)):
                    cur = new_score
                    if cur < best_score:
                        best_score = cur
                        best_mask = adj[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    flip(adj, u, v)
                    no_improve += 1
            else:
                flip(adj, u, v)
                no_improve += 1
                
            temperature *= cooling
            if temperature < 0.1:
                break
                
        if cur < best_score:
            best_score = cur
            best_mask = adj[:]

    # Final polish: try all single-edge flips from best solution
    if best_mask is not None and time.time() < deadline:
        adj = best_mask[:]
        cur = best_score
        for i in range(N):
            for j in range(i+1, N):
                if time.time() >= deadline:
                    break
                flip(adj, i, j)
                if is_k4_free(adj):
                    new_score = max_tri_free(adj)
                    if new_score < cur:
                        cur = new_score
                        best_score = cur
                        best_mask = adj[:]
                flip(adj, i, j)

    if best_mask is None:
        best_mask = make_paley_17_plus_isolated()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
