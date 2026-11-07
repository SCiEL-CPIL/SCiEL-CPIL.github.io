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
    deadline = start_time + 115.0
    
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

    def find_embedding(t_idx, used_edges, host_map, adj, deadline_check, used_hosts):
        """
        DFS to find embedding of tree t_idx into available edges.
        """
        k = t_idx + 1
        if k == 1:
            return True
        
        if deadline_check():
            return False
        
        unplaced = [v for v in range(k) if v not in host_map]
        if not unplaced:
            return True
        
        # MRV: Pick vertex with fewest valid host candidates
        best_v = None
        min_cands = float('inf')
        best_cands = []
        
        for v in unplaced:
            placed_neighbors = [host_map[nb] for nb in adj[v] if nb in host_map]
            if not placed_neighbors:
                # If no placed neighbors, candidates are all unused hosts
                # But we need to be careful: if this is the first vertex, it can be anywhere.
                # However, in our recursion, we always start with a root.
                # So there is always at least one placed neighbor unless k=1 (handled) or we are at root.
                # Actually, if v is not connected to any placed vertex, it's only possible if the tree is disconnected (it's a tree, so no) 
                # or if v is the root and we haven't placed it? No, root is placed initially.
                # Wait, the recursion places vertices. If v has no placed neighbors, it means it's not connected to the current component of placed vertices?
                # Since it's a tree, every unplaced vertex must be connected to a placed vertex? 
                # Not necessarily. If we place a subtree, other parts might be unconnected.
                # But we grow the embedding from the root. So every unplaced vertex has a path to the root.
                # The vertex adjacent to the placed part has a placed neighbor.
                # So `placed_neighbors` will be non-empty for all unplaced vertices in a growing tree embedding?
                # Yes, because we only place a vertex if it has a placed neighbor (except root).
                # Wait, my `find_embedding` doesn't enforce that. It picks ANY unplaced vertex.
                # This can lead to disconnected components if we pick a vertex whose parent is not placed yet.
                # To ensure it's a valid tree embedding, we should only pick vertices that have at least one placed neighbor, 
                # UNLESS it's the first vertex (root).
                # But root is placed initially. So all subsequent vertices must have a placed neighbor.
                
                # Let's check: if placed_neighbors is empty, it means v is not adjacent to any placed vertex.
                # In a tree, if v is not adjacent to any placed vertex, and the placed vertices form a connected component (which they do, starting from root),
                # then v cannot be placed yet because it has no edge to the placed component.
                # So we should skip such vertices.
                
                if not placed_neighbors:
                    continue 
            else:
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
            # If no vertex has placed neighbors, we are stuck or it's the root case (handled)
            # If we have unplaced vertices but none have placed neighbors, the placed set is not connected to them?
            # This shouldn't happen if we start with root and grow.
            # But if we picked a bad vertex earlier? No, we only pick vertices with placed neighbors.
            # So if unplaced is not empty, there must be a vertex with a placed neighbor.
            # Unless... the tree is T1? Handled.
            # So this case implies an error or dead end.
            return False

        # Order candidates: prefer those with fewer future options (constrained)
        def cand_score(hv):
            avail = 0
            for other in range(n):
                if other == hv or other in used_hosts:
                    continue
                e = (min(hv, other), max(hv, other))
                if e not in used_edges:
                    avail += 1
            return avail

        best_cands.sort(key=cand_score)
        
        for hv in best_cands:
            host_map[best_v] = hv
            used_hosts.add(hv)
            if find_embedding(t_idx, used_edges, host_map, adj, deadline_check, used_hosts):
                return True
            del host_map[best_v]
            used_hosts.discard(hv)
        
        return False

    def try_pack(deadline_check):
        used_edges = set()
        solution = [None] * n
        placed_trees = [False] * n
        
        # Precompute a "difficulty" score for each tree initially
        # We will dynamically select the next tree in each step
        
        def get_tree_difficulty(t_idx):
            # Approximate difficulty: sum of degrees squared? Or just size?
            # Larger trees are harder. High degree is harder.
            k = t_idx + 1
            if k == 1: return 0
            return sum(d*d for d in tree_deg[t_idx]) + k

        def backtrack():
            # Find the unplaced tree with highest difficulty (most constrained)
            # Or simply the one with smallest available embedding count? That's expensive.
            # Let's stick to static order but optimized, or dynamic based on remaining edges?
            # Dynamic: Sort unplaced trees by (difficulty, size)
            unplaced = [i for i in range(n) if not placed_trees[i]]
            if not unplaced:
                return True
            
            # Sort by difficulty descending, size descending
            unplaced.sort(key=lambda i: (-get_tree_difficulty(i), -(i+1)))
            
            # Try the most difficult one first
            # To add randomness and escape local minima, we can shuffle if difficulties are similar
            # But let's try the hardest one first.
            
            # Actually, placing small trees first might be easier to pack?
            # Or placing large trees first leaves less space?
            # Usually, most constrained first is good.
            
            t_idx = unplaced[0]
            k = t_idx + 1
            
            if k == 1:
                # T1: assign any available vertex? 
                # Wait, T1 has no edges. It just occupies a vertex?
                # "Entry k-1 is None or a list of k distinct int host vertices... position i is the image of vertex i"
                # T1 has 1 vertex. It consumes no edges.
                # Does it consume a host vertex? "list of k distinct int host vertices".
                # Yes, the vertices must be distinct within the tree.
                # Do host vertices have to be distinct ACROSS trees?
                # "Embedded trees must be edge-disjoint". Vertices can be shared?
                # "list of k distinct int host vertices in 0..19".
                # It doesn't say host vertices must be distinct across trees.
                # So T1 can use any vertex.
                # But if we use a vertex, does it affect other trees?
                # Only edges are disjoint. Vertices can be reused.
                # So T1 is trivial.
                solution[t_idx] = [0] # Any vertex works, let's say 0
                placed_trees[t_idx] = True
                if backtrack():
                    return True
                solution[t_idx] = None
                placed_trees[t_idx] = False
                return False

            adj = tree_adj[t_idx]
            
            # Try to find embedding
            # We need to choose a root.
            # Let's try a few random roots to increase success rate.
            for attempt in range(10):
                if deadline_check():
                    return False
                
                root = random.randint(0, k - 1)
                # Root can be any host vertex
                # But to be efficient, maybe pick a host vertex with high degree in remaining graph?
                # For now, random.
                root_host = random.randint(0, n - 1)
                
                host_map = {root: root_host}
                used_hosts = {root_host}
                
                if find_embedding(t_idx, used_edges, host_map, adj, deadline_check, used_hosts):
                    emb = [host_map[i] for i in range(k)]
                    solution[t_idx] = emb
                    for u, v in tree_edges[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
                    
                    placed_trees[t_idx] = True
                    if backtrack():
                        return True
                    
                    # Backtrack
                    for u, v in tree_edges[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.discard(e)
                    solution[t_idx] = None
                    placed_trees[t_idx] = False
            
            return False

        if backtrack():
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

    best_solution = None
    best_value = 0

    def deadline_check():
        return time.time() > deadline

    # Run main search
    sol = try_pack(deadline_check)
    if sol is not None:
        val = verify_solution(sol)
        if val is not None:
            return sol

    # If failed, try a greedy approach with restarts to get a partial solution
    # The current code had a greedy fallback. Let's keep a simpler one.
    
    # Since the main search is now more robust, we might not need the greedy fallback if we manage time well.
    # But if it times out, we return None. The task says "returning 20 entries... None or list".
    # So returning None for all is valid? "Entry k-1 is None or a list".
    # "The raw value is the total number of embedded tree edges".
    # So we want to maximize edges.
    
    # Let's add a quick greedy pass if backtracking fails or times out.
    # But backtracking runs until deadline.
    
    # If we reach here, we didn't find a full solution.
    # We can try to place trees one by one greedily to get some score.
    
    def try_greedy():
        used_edges = set()
        solution = [None] * n
        # Place trees in order of size? Or random?
        # Let's try size ascending (easier to place small ones first? or descending?)
        # Usually, small trees are easier to fit.
        order = sorted(range(n), key=lambda i: i+1)
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [0]
                continue
            
            adj = tree_adj[t_idx]
            found = False
            for attempt in range(50):
                if deadline_check():
                    break
                root = random.randint(0, k - 1)
                root_host = random.randint(0, n - 1)
                host_map = {root: root_host}
                used_hosts = {root_host}
                if find_embedding(t_idx, used_edges, host_map, adj, deadline_check, used_hosts):
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

    # Run greedy if we have time
    if time.time() < deadline - 5:
        sol_g = try_greedy()
        val_g = verify_solution(sol_g)
        if val_g is not None and val_g > 0:
            return sol_g

    return [None] * n

if __name__ == "__main__":
    pass
