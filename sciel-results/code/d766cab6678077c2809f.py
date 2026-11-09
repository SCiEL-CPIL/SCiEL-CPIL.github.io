import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 108.0  # seconds, safety margin before 120s

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    # Precompute common neighbor masks
    common = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE:
            return
        
        # Upper bound: current size + number of available vertices from idx onwards
        remaining_count = 0
        for i in range(idx, n):
            if not (forbidden_mask >> i) & 1:
                remaining_count += 1
        
        cur_sz = popcount(current_mask)
        if cur_sz + remaining_count <= best[0]:
            return
        
        if idx == n:
            if cur_sz > best[0]:
                best[0] = cur_sz
            return
        
        v = idx
        bit = 1 << v
        
        # Option 1: Include v (if not forbidden)
        if not (forbidden_mask & bit):
            new_forbidden = forbidden_mask
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    new_forbidden |= (adj[v] & adj[u])
            
            bb(idx + 1, current_mask | bit, new_forbidden)
        
        # Option 2: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)
    
    # Greedy initial lower bound
    greedy_set = 0
    greedy_forbidden = 0
    remaining = (1 << n) - 1
    while remaining:
        candidates = remaining & ~greedy_forbidden
        if not candidates:
            break
        best_v = -1
        best_score = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c -= lsb
            new_forb = 0
            gm = greedy_set
            while gm:
                lsb2 = gm & -gm
                u = lsb2.bit_length() - 1
                gm -= lsb2
                if (adj[v] >> u) & 1:
                    new_forb |= (adj[v] & adj[u])
            cnt = popcount(new_forb & remaining)
            if cnt < best_score:
                best_score = cnt
                best_v = v
        if best_v == -1:
            break
        greedy_set |= (1 << best_v)
        remaining &= ~(1 << best_v)
        gm2 = greedy_set & ~(1 << best_v)
        while gm2:
            lsb2 = gm2 & -gm2
            u = lsb2.bit_length() - 1
            gm2 -= lsb2
            if (adj[best_v] >> u) & 1:
                greedy_forbidden |= (adj[best_v] & adj[u])
    
    best[0] = popcount(greedy_set)
    
    # Run B&B
    bb(0, 0, 0)
    return best[0]

def _random_3_regular(n=18):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    # Each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    adj = [0] * n
    for i in range(0, n * 3, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self-loop, retry by swapping
            # For simplicity, just skip and try another pairing
            # In a proper implementation we'd re-shuffle
            pass
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Ensure it's actually 3-regular (might have multi-edges or missing degree)
    # For our purposes, we just need a sparse graph to complement
    # Let's use a simpler approach: random sparse graph
    
    # Alternative: generate random graph with each vertex having degree ~3
    adj = [0] * n
    for v in range(n):
        # Choose 3 random neighbors
        neighbors = random.sample([x for x in range(n) if x != v], 3)
        for u in neighbors:
            adj[v] |= (1 << u)
    
    # Make symmetric
    adj_sym = [0] * n
    for v in range(n):
        for u in range(n):
            if (adj[v] >> u) & 1:
                adj_sym[v] |= (1 << u)
                adj_sym[u] |= (1 << v)
    
    return adj_sym

def _complement(adj, n=18):
    """Return the complement of the graph."""
    full = (1 << n) - 1
    comp = [0] * n
    for v in range(n):
        comp[v] = full & ~(adj[v] | (1 << v))
    return comp

def _is_k4_free_complement(adj, n=18):
    """Check if complement of adj is K4-free, i.e., adj has no independent set of size 4."""
    # K4 in complement means 4 vertices with no edges in original
    # So we check if there's an independent set of size 4 in adj
    # This is equivalent to: does adj contain an independent set of size 4?
    
    # For n=18, we can check all C(18,4) = 3060 quadruples
    for a in range(n):
        for b in range(a + 1, n):
            if (adj[a] >> b) & 1:
                continue
            for c in range(b + 1, n):
                if (adj[a] >> c) & 1 or (adj[b] >> c) & 1:
                    continue
                for d in range(c + 1, n):
                    if (adj[a] >> d) & 1 or (adj[b] >> d) & 1 or (adj[c] >> d) & 1:
                        continue
                    # Found independent set of size 4
                    return False
    return True

def _improve_adj(adj, n=18):
    """Local search to improve a K4-free graph (minimize max triangle-free induced subgraph)."""
    current = adj[:]
    current_score = _max_triangle_free_induced(current, n)
    best = adj[:]
    best_score = current_score
    
    steps_since_improve = 0
    max_steps = 5000
    step = 0
    
    while step < max_steps and _time_left() > 15:
        step += 1
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        new_adj = current[:]
        if (new_adj[u] >> v) & 1:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        if not _is_k4_free(new_adj, n):
            continue
        
        new_score = _max_triangle_free_induced(new_adj, n)
        
        if new_score < best_score:
            best_score = new_score
            best = new_adj
            current = new_adj
            steps_since_improve = 0
        elif new_score == best_score and random.random() < 0.3:
            current = new_adj
        else:
            steps_since_improve += 1
        
        if steps_since_improve > 200:
            # Try a random restart from a perturbed version
            current = current[:]
            for _ in range(random.randint(1, 3)):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru != rv:
                    current[ru] ^= (1 << rv)
                    current[rv] ^= (1 << ru)
            if not _is_k4_free(current, n):
                current = best[:]
            steps_since_improve = 0
    
    return best, best_score

def construct():
    n = 18
    random.seed(42)
    
    candidates = []
    
    # Strategy 1: Complement of sparse graphs (dense graphs)
    # A dense graph is more likely to have small triangle-free induced subgraphs
    # because triangles are abundant
    
    # 1a. Complement of random sparse graphs with degree ~3
    for i in range(8):
        if _time_left() < 20:
            break
        sparse = _random_3_regular(n)
        dense = _complement(sparse, n)
        if _is_k4_free(dense, n):
            candidates.append(dense)
    
    # 1b. Complement of random bipartite graphs
    for i in range(4):
        if _time_left() < 20:
            break
        sparse = [0] * n
        # Random bipartition
        parts = [set(), set()]
        for v in range(n):
            parts[random.randint(0, 1)].add(v)
        # Add cross edges to sparse (so they become non-edges in dense)
        for a in parts[0]:
            for b in parts[1]:
                if random.random() < 0.5:
                    sparse[a] |= (1 << b)
                    sparse[b] |= (1 << a)
        dense = _complement(sparse, n)
        if _is_k4_free(dense, n):
            candidates.append(dense)
    
    # 1c. Complement of random matchings + extra edges
    for i in range(4):
        if _time_left() < 20:
            break
        sparse = [0] * n
        # Random matching
        remaining = list(range(n))
        random.shuffle(remaining)
        for i in range(0, n - 1, 2):
            u, v = remaining[i], remaining[i + 1]
            sparse[u] |= (1 << v)
            sparse[v] |= (1 << u)
        # Add some random edges
        for _ in range(random.randint(3, 8)):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not (sparse[u] >> v) & 1:
                sparse[u] |= (1 << v)
                sparse[v] |= (1 << u)
        dense = _complement(sparse, n)
        if _is_k4_free(dense, n):
            candidates.append(dense)
    
    # Strategy 2: Turan-like (complete multipartite)
    adj_turan = [0] * n
    k = 3
    part_sizes = [6, 6, 6]
    start = 0
    parts = []
    for sz in part_sizes:
        parts.append(list(range(start, start + sz)))
        start += sz
    for i in range(k):
        for j in range(i + 1, k):
            for a in parts[i]:
                for b in parts[j]:
                    adj_turan[a] |= (1 << b)
                    adj_turan[b] |= (1 << a)
    candidates.append(adj_turan)
    
    # Strategy 3: Random K4-free graphs
    for i in range(3):
        if _time_left() < 20:
            break
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Remove edges to make K4-free
        changed = True
        while changed and _time_left() > 10:
            changed = False
            for a in range(n):
                if _time_left() < 10:
                    break
                for b in range(a + 1, n):
                    if not (adj[a] >> b) & 1:
                        continue
                    common = adj[a] & adj[b]
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c -= lsb
                        if (adj[x] & common) != 0:
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        candidates.append(adj)
    
    # Evaluate and improve each candidate
    best_overall = None
    best_overall_score = 999
    
    for adj in candidates:
        if _time_left() < 15:
            break
        if not _is_k4_free(adj, n):
            continue
        improved, score = _improve_adj(adj)
        if score < best_overall_score:
            best_overall_score = score
            best_overall = improved
    
    if best_overall is None:
        best_overall = adj_turan
        best_overall_score = _max_triangle_free_induced(adj_turan, n)
    
    # Final polish
    if _time_left() > 15:
        best_overall, best_overall_score = _improve_adj(best_overall)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_overall[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
