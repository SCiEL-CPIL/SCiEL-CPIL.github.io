import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def check_c4_safe(adj, u, v):
        """Check if adding edge (u,v) to graph with adjacency bitmasks 'adj' creates a C4.
        Returns True if safe (no C4 created), False otherwise.
        Assumes (u,v) is not currently an edge.
        """
        common = adj[u] & adj[v]
        return (common & (common - 1)) == 0

    def get_degree(adj, u):
        return bin(adj[u]).count('1')

    def try_build(n, target_min_deg):
        """
        Attempts to construct a C4-free graph on n vertices with min degree >= target_min_deg.
        Returns list of edges if successful, None otherwise.
        """
        adj = [0] * n
        edge_set = set()
        
        # Pre-compute list of all potential edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # 1. Greedy Initialization
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if check_c4_safe(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                
        # 2. Local Search Improvement
        # Goal: Increase min degree to target_min_deg
        
        # If we already met the target, return
        min_deg = min(get_degree(adj, i) for i in range(n))
        if min_deg >= target_min_deg:
            return list(edge_set)

        max_iterations = 100000
        for _ in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Find vertex with min degree
            min_deg = float('inf')
            u = -1
            for i in range(n):
                d = get_degree(adj, i)
                if d < min_deg:
                    min_deg = d
                    u = i
            
            if min_deg >= target_min_deg:
                return list(edge_set)
            
            # Try to improve u
            # Get list of non-neighbors of u
            # We can iterate through all v != u and check if (u,v) is in edge_set or adj
            # To be faster, we can pick random v's
            
            improved = False
            for _attempt in range(100):
                v = random.randint(0, n - 1)
                if v == u:
                    continue
                if (adj[u] >> v) & 1:
                    continue # Already connected
                
                # Check if (u,v) can be added
                if check_c4_safe(adj, u, v):
                    # Add it
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    improved = True
                    break
                else:
                    # C4 conflict. Try to resolve via swap.
                    common = adj[u] & adj[v]
                    c_count = bin(common).count('1')
                    
                    if c_count <= 2:
                        # Find a w in C to remove (v,w)
                        ws = []
                        temp = common
                        while temp:
                            lsb = temp & -temp
                            w_idx = lsb.bit_length() - 1
                            ws.append(w_idx)
                            temp ^= lsb
                        
                        random.shuffle(ws)
                        
                        for w in ws:
                            # Swap: Remove (v, w), Add (u, v)
                            # Verify (v, w) is an edge
                            if not ((adj[v] >> w) & 1):
                                continue
                                
                            # Remove (v, w)
                            adj[v] &= ~(1 << w)
                            adj[w] &= ~(1 << v)
                            if v < w:
                                edge_set.discard((v, w))
                            else:
                                edge_set.discard((w, v))
                                
                            # Add (u, v)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            if u < v:
                                edge_set.add((u, v))
                            else:
                                edge_set.add((v, u))
                                
                            improved = True
                            break
                        
                        if improved:
                            break
            
            if not improved:
                # If we couldn't improve u, we might be stuck.
                # To escape local minima, we can try a random move that doesn't increase min degree
                # but might open up possibilities.
                # E.g., remove a random edge from a high-degree vertex.
                # This is risky. Let's just continue.
                # The randomization in v and w should help eventually.
                pass

        # Final check
        min_deg = min(get_degree(adj, i) for i in range(n))
        if min_deg >= target_min_deg:
            return list(edge_set)
            
        return None

    # Try n=47 first
    result = None
    for n in [47, 46]:
        target = n - 39
        if target < 0:
            target = 0
            
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            edges = try_build(n, target)
            if edges is not None:
                # Validate
                adj_check = [0] * n
                for u, v in edges:
                    adj_check[u] |= (1 << v)
                    adj_check[v] |= (1 << u)
                
                valid = True
                min_deg = float('inf')
                for i in range(n):
                    d = bin(adj_check[i]).count('1')
                    if d < min_deg:
                        min_deg = d
                    if d < target:
                        valid = False
                        break
                
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj_check[i] >> j) & 1:
                                common = adj_check[i] & adj_check[j]
                                if (common & (common - 1)) != 0:
                                    valid = False
                                    break
                        if not valid:
                            break
                            
                if valid:
                    result = {'n': n, 'edges': edges}
                    break
        
        if result:
            break

    if result:
        return result

    # Fallback
    for n in range(45, 38, -1):
        target = n - 39
        if target < 0:
            target = 0
        for seed in range(10):
            if time.time() > deadline - 2:
                break
            random.seed(seed + 100)
            edges = try_build(n, target)
            if edges is not None:
                return {'n': n, 'edges': edges}
                
    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
