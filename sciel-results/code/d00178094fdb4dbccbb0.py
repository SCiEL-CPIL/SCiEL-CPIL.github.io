import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 20
    best_edges = []
    best_count = 0

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)

    def is_valid_add(adj, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free.
        # After adding (u,v):
        # 1. u and v become common neighbors of pairs (v,y) for y in N(u)\{v} and (u,x) for x in N(v)\{u}
        # 2. Pair (u,v) itself must have < 3 common neighbors
        
        # Common neighbors of u and v (excluding u and v themselves, which aren't in their own adj)
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() >= 3:
            return False
        
        # For each neighbor y of u (y != v, y != u): u becomes common neighbor of (v, y)
        # Need current common(v, y) < 2
        w_mask = adj[u] & ~(1 << v)  # neighbors of u excluding v
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() >= 2:
                return False
            w_mask ^= lsb
        
        # For each neighbor x of v (x != u, x != v): v becomes common neighbor of (u, x)
        # Need current common(u, x) < 2
        w_mask = adj[v] & ~(1 << u)  # neighbors of v excluding u
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() >= 2:
                return False
            w_mask ^= lsb
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    missing.append((i, j))
        return missing

    def get_present(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(order):
        adj = [0] * n
        count = 0
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def local_search(adj, count):
        """Try to add any missing edge that is valid. Repeat until no improvement."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing(adj)
            if not missing:
                break
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
        return count

    def two_opt_swap(adj, count):
        """Remove one edge, try to add two others. Keep if net gain >= 1."""
        if time.time() > deadline:
            return count
        
        present = get_present(adj)
        missing = get_missing(adj)
        
        if not present or len(missing) < 2:
            return count

        random.shuffle(present)
        
        for u, v in present:
            if time.time() > deadline - 5:
                break
            
            remove_edge(adj, u, v)
            count -= 1
            
            added_edges = []
            candidates = missing[:]
            random.shuffle(candidates)
            
            for a, b in candidates:
                if (a, b) == (u, v) or (a, b) == (v, u):
                    continue
                if is_valid_add(adj, a, b):
                    add_edge(adj, a, b)
                    count += 1
                    added_edges.append((a, b))
                    if len(added_edges) >= 2:
                        break
            
            if len(added_edges) < 2:
                # Revert
                for a, b in added_edges:
                    remove_edge(adj, a, b)
                    count -= 1
                add_edge(adj, u, v)
                count += 1
            else:
                break
        
        return count

    def simulated_annealing(adj, count):
        """Simulated annealing with edge removals to escape local optima."""
        if time.time() > deadline:
            return count
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.01
        
        while T > min_T and time.time() < deadline - 2:
            if random.random() < 0.6:
                missing = get_missing(adj)
                if not missing:
                    T *= cooling_rate
                    continue
                u, v = random.choice(missing)
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                T *= cooling_rate
            else:
                present = get_present(adj)
                if not present:
                    T *= cooling_rate
                    continue
                u, v = random.choice(present)
                remove_edge(adj, u, v)
                count -= 1
                
                # Greedily try to add back edges
                for _ in range(5):
                    if time.time() > deadline - 2:
                        break
                    missing = get_missing(adj)
                    if not missing:
                        break
                    random.shuffle(missing)
                    added = False
                    for a, b in missing:
                        if is_valid_add(adj, a, b):
                            add_edge(adj, a, b)
                            count += 1
                            added = True
                            break
                    if not added:
                        break
                T *= cooling_rate
        
        return count

    def final_polish(adj, count):
        """High-precision polish: exhaustively try all missing edges in multiple passes."""
        for _ in range(10):
            if time.time() > deadline - 1:
                break
            improved = False
            missing = get_missing(adj)
            if not missing:
                break
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
            if not improved:
                break
        return count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 15:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        
        count = local_search(adj, count)
        count = two_opt_swap(adj, count)
        count = local_search(adj, count)
        count = simulated_annealing(adj, count)
        count = local_search(adj, count)
        count = final_polish(adj, count)
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb
        
        restarts += 1
        if time.time() > deadline - 15:
            break

    return best_edges
