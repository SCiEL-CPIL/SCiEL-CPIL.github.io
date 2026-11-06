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
        
        # Sort triangles by number of common vertices? Or just use as is.
        # To speed up B&B, we can precompute for each vertex the triangles it belongs to?
        # The current B&B scans all triangles to find one in 'avail'.
        # This is O(T) per node.
        
        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            
            # Find a triangle in avail
            tri = None
            # Optimization: Check triangles in a specific order?
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

    def random_k4_free_graph(density=0.5):
        adj_mask = [0] * N
        order = list(range(N))
        rng.shuffle(order)
        for i in range(N):
            for j in range(i + 1, N):
                u, v = order[i], order[j]
                if rng.random() < density:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~((1 << u) | (1 << v))
                    m = common
                    ok = True
                    while m:
                        w = (m & -m).bit_length() - 1
                        m &= m - 1
                        if (adj_mask[w] & m) != 0:
                            ok = False
                            break
                    if ok:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        return adj_mask

    def repair_k4(adj_mask):
        """Remove edges to make graph K4-free. Returns True if successful."""
        changed = True
        while changed:
            changed = False
            for a in range(N):
                for b in range(a + 1, N):
                    if (adj_mask[a] >> b) & 1:
                        common = adj_mask[a] & adj_mask[b]
                        common &= ~((1 << a) | (1 << b))
                        m = common
                        while m:
                            c = (m & -m).bit_length() - 1
                            m &= m - 1
                            common_c = adj_mask[c] & common
                            common_c &= ~((1 << c) | (1 << a) | (1 << b))
                            if common_c:
                                # Found K4 a,b,c,d
                                d_bit = common_c & -common_c
                                d = d_bit.bit_length() - 1
                                # Remove random edge from {a,b,c,d}
                                edges_to_remove = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                u, v = rng.choice(edges_to_remove)
                                adj_mask[u] &= ~(1 << v)
                                adj_mask[v] &= ~(1 << u)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return is_k4_free(adj_mask)

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

    seeds = []
    seeds.append(make_turan_3())
    seeds.append(make_structured_cyclic())
    for d in [0.4, 0.5, 0.6, 0.7, 0.8]:
        adj = [0]*N
        for i in range(N):
            for j in range(i+1, N):
                if rng.random() < d:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if repair_k4(adj):
            seeds.append(adj)

    for r in range(50): # More restarts
        if time.time() >= deadline:
            break
        if r < len(seeds):
            adj = seeds[r][:]
        else:
            # Generate random dense graph and repair
            adj = [0]*N
            density = 0.5 + rng.random() * 0.4
            for i in range(N):
                for j in range(i+1, N):
                    if rng.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if not repair_k4(adj):
                continue

        if not is_k4_free(adj):
            continue
            
        cur = max_tri_free(adj)
        
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

    if best_mask is None:
        best_mask = make_turan_3()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
