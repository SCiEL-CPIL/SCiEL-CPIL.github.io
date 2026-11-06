import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    def bit_count(x):
        return x.bit_count()
        
    def repair(adj):
        # Remove edges to make graph valid
        # Use a queue of violations to be efficient, or just scan
        # Scanning is fine for n=20
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    cnt = bit_count(common)
                    if cnt > 2:
                        # Remove an edge (w, u) for some w in common
                        mask = common
                        ws = []
                        while mask:
                            lsb = mask & -mask
                            ws.append(lsb.bit_length() - 1)
                            mask ^= lsb
                        if ws:
                            w = random.choice(ws)
                            # Remove (w, u)
                            adj[w] &= ~(1 << u)
                            adj[u] &= ~(1 << w)
                            changed = True
                            break
                if time.time() > deadline:
                    return adj
        return adj

    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False
            
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            if bit_count(adj[u] & adj[w]) > 1:
                return False
            mask ^= lsb
            
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            if bit_count(adj[v] & adj[w]) > 1:
                return False
            mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def count_edges(adj):
        return sum(bit_count(adj[u]) for u in range(n)) // 2

    def local_search(adj):
        # Greedy add
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if can_add_edge(adj, u, v):
                            add_edge(adj, u, v)
                            improved = True
            if time.time() > deadline:
                break
                
        # Kick if stuck
        # Try a few random kicks
        for _ in range(10):
            if time.time() > deadline:
                break
            # Remove a random edge
            edges = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append((u, v))
                    mask ^= lsb
            if not edges:
                break
            u, v = random.choice(edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Try to add edges
            improved = True
            while improved:
                improved = False
                for uu in range(n):
                    for vv in range(uu + 1, n):
                        if not (adj[uu] & (1 << vv)):
                            if can_add_edge(adj, uu, vv):
                                add_edge(adj, uu, vv)
                                improved = True
                if time.time() > deadline:
                    break
        return adj

    def run_trial(seed):
        random.seed(seed)
        adj = [0] * n
        # Try different densities
        p = 0.5
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        adj = repair(adj)
        adj = local_search(adj)
        return adj

    trial_num = 0
    while time.time() < deadline:
        trial_num += 1
        # Vary seed
        adj = run_trial(trial_num)
        
        # Final polish
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if can_add_edge(adj, u, v):
                            add_edge(adj, u, v)
                            improved = True
            if time.time() > deadline:
                break
        
        cur_count = count_edges(adj)
        if cur_count > best_count:
            best_count = cur_count
            best_edges = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        best_edges.append([u, v])
                    mask ^= lsb

    return best_edges
# EVOLVE-BLOCK-END
