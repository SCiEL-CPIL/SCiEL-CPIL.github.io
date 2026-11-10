import time
import random
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 112  # Safety margin
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Adjacency as bitmasks
        adj = [0] * N  # adj[i] is bitmask of neighbors of i
        edge_count = 0
        
        def has_edge(u, v):
            return (adj[u] >> v) & 1
        
        def add_edge(u, v):
            nonlocal edge_count
            if not has_edge(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                return True
            return False
        
        def remove_edge(u, v):
            nonlocal edge_count
            if has_edge(u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_count -= 1
                return True
            return False
        
        def get_edges_list():
            edges = []
            for u in range(N):
                m = adj[u]
                while m:
                    low = m & (-m)
                    v = low.bit_length() - 1
                    m ^= low
                    if u < v:
                        edges.append([u, v])
            return edges
        
        def has_c6_with_edge(u, v):
            """
            Check if adding edge (u,v) would create a C6.
            Fast bitset-based check.
            """
            # We need a simple path u-a-b-c-d-v of length 5
            # BFS from u, avoiding v, up to depth 4
            layers = [0] * 5
            layers[0] = 1 << u
            
            for d in range(1, 5):
                mask = 0
                prev = layers[d-1]
                m = prev
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    nm = adj[node] & ~(1 << v)
                    mask |= nm
                # Remove nodes in earlier layers
                for d2 in range(d):
                    mask &= ~layers[d2]
                layers[d] = mask
            
            v_neighbors = adj[v] & ~(1 << u)
            if (layers[4] & v_neighbors) == 0:
                return False
            
            # Verify with DFS
            def dfs(node, depth, path_mask):
                if depth == 4:
                    return (adj[node] >> v) & 1
                nm = adj[node] & ~(1 << v) & ~path_mask
                m = nm
                while m:
                    low = m & (-m)
                    nb = low.bit_length() - 1
                    m ^= low
                    if dfs(nb, depth + 1, path_mask | (1 << nb)):
                        return True
                return False
            
            return dfs(u, 0, 1 << u)
        
        def is_c6_free(u, v):
            if has_edge(u, v):
                return True
            return not has_c6_with_edge(u, v)
        
        # Initial graph: 8 disjoint K5s
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(base + a, base + b)
        
        local_best = edge_count
        local_best_adj = adj[:]
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        # Greedy phase with perturbation
        T = 5.0
        cooling = 0.99
        
        while time.time() < deadline:
            # Try to add edges in random order
            random.shuffle(all_pairs)
            added = 0
            for u, v in all_pairs:
                if time.time() > deadline:
                    break
                if not has_edge(u, v) and is_c6_free(u, v):
                    add_edge(u, v)
                    added += 1
                    if edge_count > local_best:
                        local_best = edge_count
                        local_best_adj = adj[:]
            
            if added == 0:
                # Stuck: perturbation
                # Remove 2-4 random edges
                num_removals = random.randint(2, 4)
                for _ in range(num_removals):
                    if edge_count > 0:
                        u, v = random.choice(all_pairs)
                        if has_edge(u, v):
                            remove_edge(u, v)
            
            T *= cooling
            if T < 0.1:
                break
        
        # Final greedy
        for _ in range(5):
            improved = False
            for u, v in all_pairs:
                if time.time() > deadline:
                    break
                if not has_edge(u, v) and is_c6_free(u, v):
                    add_edge(u, v)
                    improved = True
                    if edge_count > local_best:
                        local_best = edge_count
                        local_best_adj = adj[:]
            if not improved:
                break
        
        if local_best > best_count:
            best_count = local_best
            best_edges_list = get_edges_list()
        
        return local_best
    
    seeds = [42, 123, 7, 999, 31415, 271, 808, 555, 111, 222]
    for i, s in enumerate(seeds):
        if time.time() > deadline - 3:
            break
        run_once(s)
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
