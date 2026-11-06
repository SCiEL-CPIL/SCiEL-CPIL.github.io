import time
import random
import math

N = 18
FULL = (1 << N) - 1
DEADLINE_MARGIN = 5  # safety margin before 150s

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random(42)

    def is_k4_free(adj_mask):
        for a in range(N):
            for b in range(a + 1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    c_list = []
                    while m:
                        c = (m & -m).bit_length() - 1
                        c_list.append(c)
                        m &= m - 1
                    for i in range(len(c_list)):
                        for j in range(i + 1, len(c_list)):
                            if (adj_mask[c_list[i]] >> c_list[j]) & 1:
                                return False
        return True

    def find_all_triangles(adj_mask):
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
        return tris

    def max_tri_free(adj_mask):
        """Exact size of largest triangle-free induced subgraph via branch and bound."""
        tris = find_all_triangles(adj_mask)
        if not tris:
            return N
        best = [0]
        # Precompute for each triangle its vertices as a tuple for fast access
        tri_list = list(tris)
        
        def bnb(avail):
            cnt = avail.bit_count()
            if cnt <= best[0]:
                return
            # Find any triangle fully contained in avail
            tri = None
            for (a, b, c) in tri_list:
                if (avail >> a) & 1 and (avail >> b) & 1 and (avail >> c) & 1:
                    tri = (a, b, c)
                    break
            if tri is None:
                best[0] = cnt
                return
            a, b, c = tri
            # Branch: remove one of the three vertices
            bnb(avail & ~(1 << a))
            bnb(avail & ~(1 << b))
            bnb(avail & ~(1 << c))
        
        bnb(FULL)
        return best[0]

    def make_k666():
        adj_mask = [0] * N
        for i in range(N):
            for j in range(N):
                if i != j and i // 6 != j // 6:
                    adj_mask[i] |= (1 << j)
        return adj_mask

    def random_k4_free_graph():
        """Generate a random K4-free graph by adding edges one at a time."""
        adj_mask = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.5:
                    # Check if adding (u,v) creates a K4
                    creates_k4 = False
                    for w in range(N):
                        if w == u or w == v:
                            continue
                        if not ((adj_mask[u] >> w) & 1) or not ((adj_mask[v] >> w) & 1):
                            continue
                        for x in range(N):
                            if x in (u, v, w):
                                continue
                            if ((adj_mask[u] >> x) & 1 and (adj_mask[v] >> x) & 1
                                    and (adj_mask[w] >> x) & 1
                                    and (adj_mask[x] >> w) & 1):
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    if not creates_k4:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        return adj_mask

    def flip(adj_mask, u, v):
        if (adj_mask[u] >> v) & 1:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

    def anneal(adj_mask, current_score, deadline):
        """Simulated annealing to improve the graph."""
        T = 1.0
        T_min = 0.01
        cooling = 0.997
        local_best = current_score
        best_local_mask = adj_mask[:]
        
        while time.time() < deadline and T > T_min:
            u, v = rng.sample(range(N), 2)
            old_val = (adj_mask[u] >> v) & 1
            flip(adj_mask, u, v)
            
            if not is_k4_free(adj_mask):
                flip(adj_mask, u, v)  # revert
                T *= cooling
                continue
            
            new_score = max_tri_free(adj_mask)
            
            if new_score < current_score or (
                    T > T_min and rng.random() < math.exp((current_score - new_score) / T)):
                current_score = new_score
                if new_score < local_best:
                    local_best = new_score
                    best_local_mask = adj_mask[:]
            else:
                flip(adj_mask, u, v)  # revert
            
            T *= cooling
        
        return local_best, best_local_mask

    best_score = N + 1
    best_mask = None

    # Initial structures to seed the search
    seeds = [make_k666()]
    seeds.append(random_k4_free_graph())
    # Add a few more random seeds for diversity
    for _ in range(3):
        seeds.append(random_k4_free_graph())

    # Run multiple independent restarts
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() >= deadline:
            break
        
        if r < len(seeds):
            adj = seeds[r][:]
        else:
            adj = random_k4_free_graph()
        
        if not is_k4_free(adj):
            adj = make_k666()
        
        cur = max_tri_free(adj)
        
        # Give each restart a portion of the remaining time
        remaining = deadline - time.time()
        per_restart_time = remaining / max(1, n_restarts - r)
        cur_deadline = time.time() + min(per_restart_time, 10)
        
        local_best, local_mask = anneal(adj, cur, cur_deadline)
        
        if local_best < best_score:
            best_score = local_best
            best_mask = local_mask[:]

    # If we still have time, do a final polish on the best found
    if best_mask is not None and time.time() < deadline - 1:
        final_adj = best_mask[:]
        final_score = max_tri_free(final_adj)
        final_best, final_mask = anneal(final_adj, final_score, deadline)
        if final_best < best_score:
            best_score = final_best
            best_mask = final_mask[:]

    if best_mask is None:
        best_mask = make_k666()

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
