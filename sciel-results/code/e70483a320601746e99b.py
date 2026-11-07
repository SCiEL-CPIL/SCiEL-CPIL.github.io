import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    tree_edges = [trees[i] for i in range(n)]
    tree_adj = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in tree_edges[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    tree_deg = []
    for i in range(n):
        k = i + 1
        deg = [0] * k
        for u, v in tree_edges[i]:
            deg[u] += 1
            deg[v] += 1
        tree_deg.append(deg)

    def find_embedding(t_idx, used_edges, host_map, adj, deadline_check, used_hosts, rng):
        """
        Optimized DFS to find embedding of tree t_idx into available edges.
        Uses MRV on the frontier (unplaced vertices with placed neighbors).
        """
        k = t_idx + 1
        if k == 1:
            return True
        
        if deadline_check():
            return False
        
        # Identify frontier: unplaced vertices that have at least one placed neighbor
        frontier = []
        for v in range(k):
            if v not in host_map:
                has_placed_neighbor = any(nb in host_map for nb in adj[v])
                if has_placed_neighbor:
                    frontier.append(v)
        
        if not frontier:
            # If there are unplaced vertices but no frontier, the tree is disconnected or we are stuck
            # In a tree, if we start from a root and grow, this shouldn't happen unless we picked a bad path
            # But if we have unplaced vertices, they must be connected to the placed component.
            # So if frontier is empty, we are done? No, if unplaced is not empty, we are stuck.
            unplaced = [v for v in range(k) if v not in host_map]
            if not unplaced:
                return True
            else:
                return False

        # MRV: Pick vertex in frontier with fewest valid host candidates
        best_v = None
        min_cands = float('inf')
        best_cands = []
        
        for v in frontier:
            placed_neighbors = [host_map[nb] for nb in adj[v] if nb in host_map]
            cand_set = set()
            for ph in placed_neighbors:
                for hv in range(n):
                    if hv in used_hosts or hv == ph:
                        continue
                    e = (min(ph, hv), max(ph, hv))
                    if e not in used_edges:
                        cand_set.add(hv)
            
            if len(cand_set) < min_cands:
                min_cands = len(cand_set)
                best_cands = list(cand_set)
                best_v = v
                if min_cands == 0:
                    return False
                if min_cands == 1:
                    break

        if best_v is None:
            return False

        # Heuristic for candidate ordering: 
        # Prefer hosts that are "central" or have high degree in the remaining graph?
        # Or simply randomize to avoid bias? 
        # Let's use a deterministic tie-breaker based on host ID for consistency, 
        # but shuffle if we want variety. 
        # For now, sort by host ID to be deterministic.
        best_cands.sort()
        
        for hv in best_cands:
            host_map[best_v] = hv
            used_hosts.add(hv)
            if find_embedding(t_idx, used_edges, host_map, adj, deadline_check, used_hosts, rng):
                return True
            del host_map[best_v]
            used_hosts.discard(hv)
        
        return False

    def try_pack(order, rng):
        used_edges = set()
        solution = [None] * n
        placed_trees = [False] * n
        
        def backtrack(idx):
            if idx == len(order):
                return True
            
            if time.time() > deadline:
                return False
                
            t_idx = order[idx]
            k = t_idx + 1
            
            if k == 1:
                # T1: Any vertex. Pick 0 if available? No, vertices can be reused.
                # But to be safe, just pick 0.
                solution[t_idx] = [0]
                placed_trees[t_idx] = True
                if backtrack(idx + 1):
                    return True
                solution[t_idx] = None
                placed_trees[t_idx] = False
                return False

            adj = tree_adj[t_idx]
            
            # Try different roots for this tree
            # To avoid symmetry, we can fix the root of the first tree to 0?
            # No, the host graph is symmetric, but the tree is not.
            # Let's try all roots for the first tree? That's expensive.
            # Let's try a few random roots.
            
            # Heuristic: Try roots that are "leaves" of the tree first? Or "centers"?
            # Let's just try all k roots in a random order.
            roots = list(range(k))
            rng.shuffle(roots)
            
            for root in roots:
                # Try all host vertices for the root?
                # That's 20 options.
                # To speed up, we can fix the root host to 0 for the first tree?
                # No, that might be too restrictive if the tree is large.
                # But K20 is highly symmetric.
                # Let's try a subset of host vertices for the root, e.g., 0..4, then random.
                # Actually, let's just try all 20 host vertices for the root.
                # 20 * (recursive cost).
                
                for root_host in range(n):
                    host_map = {root: root_host}
                    used_hosts = {root_host}
                    
                    if find_embedding(t_idx, used_edges, host_map, adj, lambda: time.time() > deadline, used_hosts, rng):
                        emb = [host_map[i] for i in range(k)]
                        solution[t_idx] = emb
                        for u, v in tree_edges[t_idx]:
                            hu, hv = emb[u], emb[v]
                            e = (min(hu, hv), max(hu, hv))
                            used_edges.add(e)
                        
                        placed_trees[t_idx] = True
                        if backtrack(idx + 1):
                            return True
                        
                        # Backtrack
                        for u, v in tree_edges[t_idx]:
                            hu, hv = emb[u], emb[v]
                            e = (min(hu, hv), max(hu, hv))
                            used_edges.discard(e)
                        solution[t_idx] = None
                        placed_trees[t_idx] = False
            
            return False

        if backtrack(0):
            return solution
        return None

    def verify_solution(sol):
        used = set()
        total = 0
        for i in range(n):
            if sol[i] is not None:
                k = i + 1
                if len(sol[i]) != k:
                    return None
                if len(set(sol[i])) != k:
                    return None
                for v in sol[i]:
                    if not (0 <= v < n):
                        return None
                for u, v in tree_edges[i]:
                    hu, hv = sol[i][u], sol[i][v]
                    if hu == hv:
                        return None
                    e = (min(hu, hv), max(hu, hv))
                    if e in used:
                        return None
                    used.add(e)
                    total += 1
        return total

    # Strategy:
    # 1. Sort trees by size descending (largest first) to constrain early.
    # 2. Use multiple random restarts if the first attempt fails.
    
    # Pre-compute tree sizes
    sizes = [i+1 for i in range(n)]
    
    # Order 1: Largest first
    order1 = sorted(range(n), key=lambda i: -sizes[i])
    
    # Try with a fixed seed first
    rng = Random(42)
    sol = try_pack(order1, rng)
    if sol is not None:
        val = verify_solution(sol)
        if val is not None:
            return sol

    # If failed, try different orders and seeds
    # We have ~100s left.
    # Let's try a few random orders.
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        rng = Random(100 + restart)
        
        # Randomize the order slightly? Or keep largest first?
        # Largest first is usually good. Let's keep it.
        # But maybe shuffle the roots? The try_pack already shuffles roots.
        
        # Let's try a different order: Random
        order_r = list(range(n))
        rng.shuffle(order_r)
        
        sol = try_pack(order_r, rng)
        if sol is not None:
            val = verify_solution(sol)
            if val is not None:
                return sol

    # Fallback: Greedy
    # If backtracking fails, try to pack as many as possible greedily.
    def try_greedy():
        used_edges = set()
        solution = [None] * n
        rng = Random(999)
        
        # Try to place trees in random order
        order = list(range(n))
        rng.shuffle(order)
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [0]
                continue
            
            adj = tree_adj[t_idx]
            found = False
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                root = rng.randint(0, k - 1)
                root_host = rng.randint(0, n - 1)
                host_map = {root: root_host}
                used_hosts = {root_host}
                
                # Use a simple greedy embedding (no backtracking)
                # This is just a quick check
                # For greedy, we can just pick the first valid candidate
                # This is not robust, but better than nothing.
                
                # Actually, let's use the find_embedding but with a strict time limit per tree
                # Or just skip this and return what we have.
                
                # Let's try to find one embedding using find_embedding with a very short deadline
                # But find_embedding uses the global deadline.
                # Let's just return the solution we have so far.
                
                # To make greedy work, we need a non-backtracking embedder.
                # Let's implement a simple greedy embedder.
                
                def greedy_embed(t_idx, used_edges, host_map, adj, used_hosts):
                    k = t_idx + 1
                    if k == 1:
                        return True
                    
                    # BFS/DFS from root
                    # We need to ensure we don't get stuck.
                    # Since we don't backtrack, we just pick the first valid candidate.
                    
                    # We need to process vertices in a specific order (e.g., BFS from root)
                    # Let's use a queue
                    queue = [root]
                    visited = {root}
                    
                    while queue:
                        v = queue.pop(0)
                        for nb in adj[v]:
                            if nb in visited:
                                continue
                            
                            # Find a valid host for nb
                            placed_neighbors = [host_map[x] for x in adj[nb] if x in host_map]
                            # In a tree, nb has exactly one parent in the tree (if we grow from root)
                            # Wait, the tree is undirected.
                            # If we grow from root, each new vertex has exactly one placed neighbor (its parent).
                            # So placed_neighbors should have size 1.
                            
                            if len(placed_neighbors) != 1:
                                # This can happen if the tree is not rooted properly or we made a mistake.
                                # But we are growing a tree, so it should be 1.
                                return False
                            
                            ph = placed_neighbors[0]
                            found_host = False
                            for hv in range(n):
                                if hv in used_hosts or hv == ph:
                                    continue
                                e = (min(ph, hv), max(ph, hv))
                                if e not in used_edges:
                                    host_map[nb] = hv
                                    used_hosts.add(hv)
                                    visited.add(nb)
                                    queue.append(nb)
                                    found_host = True
                                    break
                            
                            if not found_host:
                                return False
                            
                    return True

                if greedy_embed(t_idx, used_edges, host_map, adj, used_hosts):
                    emb = [host_map[i] for i in range(k)]
                    solution[t_idx] = emb
                    for u, v in tree_edges[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
                    found = True
                    break
            
            if not found:
                solution[t_idx] = None

        return solution

    if time.time() < deadline - 5:
        sol_g = try_greedy()
        val_g = verify_solution(sol_g)
        if val_g is not None and val_g > 0:
            return sol_g

    return [None] * n

if __name__ == "__main__":
    pass
