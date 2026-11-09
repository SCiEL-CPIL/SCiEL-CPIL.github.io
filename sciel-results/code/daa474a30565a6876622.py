"""Packing trees T1..T20 into K20 using optimized backtracking with forward checking."""
import time
from random import Random
from collections import deque

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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 110.0
    n = 20
    trees = instance_trees(n, 611)
    
    # Global state
    used_edges = set()
    result = [None] * n
    
    # Precompute adjacency for each tree
    tree_adj = []
    tree_deg = []
    for edges in trees:
        k = len(edges) + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)

    # Order trees by size descending (largest first) to constrain early
    order = list(range(n))
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    def solve_tree(k_idx, timeout):
        """
        Embed tree k_idx into K20 avoiding used_edges.
        Returns list of host vertices (length k) or None.
        """
        k = k_idx + 1
        adj = tree_adj[k_idx]
        deg = tree_deg[k_idx]
        
        # Variable ordering: Highest degree first
        # We will assign hosts to vertices in this order
        verts = list(range(k))
        verts.sort(key=lambda v: -deg[v])
        
        # Map from tree vertex index to position in assignment array
        # We maintain:
        # host[v] : the host vertex assigned to tree vertex v, or -1
        # assigned[v] : bool
        host = [-1] * k
        assigned = [False] * k
        
        # For backtracking, we need to undo.
        # We'll use recursion.
        
        # Heuristic: For a vertex v, valid hosts are those h in 0..19
        # such that for all assigned neighbors u of v, (host[u], h) not in used_edges
        # AND (host[u], h) is not in used_edges_global (which is checked dynamically)
        
        # Forward Checking:
        # When assigning host h to vertex v, for every unassigned neighbor w of v,
        # check if there exists at least one host h' for w such that (h, h') is not in used_edges
        # and h' is not already assigned to another vertex.
        
        # This is expensive to check fully every time. 
        # Simplified Forward Checking:
        # Just check if the edge (host[v], host[w]) is free for assigned w.
        # And for unassigned w, just check if the set of available hosts for w is non-empty.
        
        # Let's implement a simple backtracking with strong pruning.
        
        def get_available_hosts(v):
            """Return list of hosts h such that edge to all assigned neighbors is free."""
            avail = []
            for h in range(n):
                ok = True
                for u in adj[v]:
                    if assigned[u]:
                        edge = (min(host[u], h), max(host[u], h))
                        if edge in used_edges:
                            ok = False
                            break
                if ok:
                    avail.append(h)
            return avail
        
        def is_consistent(v, h):
            """Check if assigning h to v is consistent with current assignments."""
            for u in adj[v]:
                if assigned[u]:
                    edge = (min(host[u], h), max(host[u], h))
                    if edge in used_edges:
                        return False
            return True

        def forward_check(v, h):
            """
            Check if assigning h to v leaves any unassigned neighbor w with no available hosts.
            """
            for w in adj[v]:
                if not assigned[w]:
                    # Check if w has any available host
                    # We don't need to check ALL hosts, just if the set is empty.
                    # Optimization: If w has many unassigned neighbors, it's likely fine.
                    # Only check if w has high degree or few options?
                    # For now, check all n hosts.
                    has_option = False
                    for h_w in range(n):
                        # Check consistency with all assigned neighbors of w
                        ok = True
                        for u in adj[w]:
                            if assigned[u]:
                                edge = (min(host[u], h_w), max(host[u], h_w))
                                if edge in used_edges:
                                    ok = False
                                    break
                        if ok:
                            has_option = True
                            break
                    if not has_option:
                        return False
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(verts):
                return True
            
            v = verts[idx]
            # Get available hosts
            avails = get_available_hosts(v)
            
            # Heuristic: Sort hosts by how many options they leave for neighbors?
            # Too expensive. Just shuffle to break symmetry.
            Random(42 + k_idx + idx).shuffle(avails)
            
            for h in avails:
                if not is_consistent(v, h):
                    continue
                if not forward_check(v, h):
                    continue
                
                # Assign
                host[v] = h
                assigned[v] = True
                # Add edges to used_edges temporarily?
                # No, used_edges is global. We must add and remove.
                # But used_edges is a set. Adding/removing is O(1).
                # However, we are in a recursive call.
                # We need to track which edges we added.
                
                # Actually, the check `edge in used_edges` refers to the GLOBAL state.
                # If we assign h to v, we "occupy" edges (host[u], h) for assigned u.
                # But these edges are NOT added to used_edges until the tree is fully embedded?
                # No, they must be considered "used" by this tree to prevent conflicts WITHIN the tree.
                # But the tree edges are distinct.
                # Wait, the constraint is that edges of DIFFERENT trees must be disjoint.
                # Edges of the SAME tree are always disjoint (it's a tree).
                # So we only need to check against `used_edges` (global) and against
                # edges already assigned in THIS tree.
                
                # My `is_consistent` checks against `used_edges`.
                # It does NOT check against edges assigned to other vertices in the same tree!
                # Example: v=0, u=1. Assign h0 to 0, h1 to 1.
                # If we assign h2 to 2, and 2 is connected to 0 and 1.
                # We must ensure (h0, h2) and (h1, h2) are not in used_edges.
                # AND (h0, h2) != (h1, h2) is trivially true if h0 != h1.
                # But what if (h0, h2) was already used by another tree? `used_edges` catches that.
                # What if (h0, h2) is the same as (h1, h2)? Impossible if h0!=h1.
                # So we just need to check against `used_edges` and ensure we don't reuse a host vertex.
                
                # CRITICAL MISSING CONSTRAINT: Host vertices must be distinct!
                # `host` array must contain distinct values.
                
                # Let's add a check: h not in [host[u] for u in assigned vertices]
                # We can maintain a `used_hosts` set.
                
                # Let's restructure to track used_hosts locally.
                pass # Placeholder, will rewrite in final code

        # Rewrite backtrack logic with used_hosts
        used_hosts = set()
        
        def bt(idx):
            if time.time() > deadline:
                return False
            if idx == len(verts):
                return True
            
            v = verts[idx]
            
            # Available hosts: not in used_hosts, and edges to assigned neighbors are in used_edges
            cands = []
            for h in range(n):
                if h in used_hosts:
                    continue
                # Check edges to assigned neighbors
                ok = True
                for u in adj[v]:
                    if assigned[u]:
                        e = (min(host[u], h), max(host[u], h))
                        if e in used_edges:
                            ok = False
                            break
                if ok:
                    cands.append(h)
                    
            if not cands:
                return False
            
            # Heuristic: Try hosts that minimize future conflicts?
            # Just random shuffle
            Random(time.time_ns()).shuffle(cands)
            
            for h in cands:
                # Forward check: For each unassigned neighbor w, is there any host?
                fc_ok = True
                for w in adj[v]:
                    if not assigned[w]:
                        # Check if w has any valid host
                        # A valid host h_w must not be in used_hosts U {h}
                        # and edges to assigned neighbors of w must be free
                        # Note: v is now "assigned" (temporarily) with h
                        # So for w, v is an assigned neighbor with host h.
                        
                        has_opt = False
                        for h_w in range(n):
                            if h_w in used_hosts or h_w == h:
                                continue
                            ok_w = True
                            for u in adj[w]:
                                if assigned[u] or u == v:
                                    # If u is assigned, host[u] is set.
                                    # If u is v, host is h.
                                    hu = host[u] if assigned[u] else h
                                    e = (min(hu, h_w), max(hu, h_w))
                                    if e in used_edges:
                                        ok_w = False
                                        break
                            if ok_w:
                                has_opt = True
                                break
                        if not has_opt:
                            fc_ok = False
                            break
                
                if not fc_ok:
                    continue
                
                # Assign
                host[v] = h
                assigned[v] = True
                used_hosts.add(h)
                
                if bt(idx + 1):
                    return True
                
                # Unassign
                host[v] = -1
                assigned[v] = False
                used_hosts.remove(h)
                
            return False
        
        return bt(0)

    for k_idx in order:
        if time.time() > deadline:
            break
        k = k_idx + 1
        if k == 1:
            # Single vertex, no edges. Any host works.
            # But we need to pick a host.
            # Let's pick 0 if available, else 1, etc.
            for h in range(n):
                # No edges, so no conflict with used_edges.
                # But we should keep track of used hosts?
                # The problem statement says "list of k distinct int host vertices".
                # It does NOT say host sets of different trees must be disjoint.
                # "Embedded trees must be edge-disjoint".
                # So vertices can be shared.
                # So for k=1, we can just pick [0].
                result[k_idx] = [0]
                break
            continue
            
        # Try to solve
        # We need to modify used_edges? No, used_edges is global.
        # The solver checks against used_edges.
        # But it doesn't ADD to used_edges during the search.
        # It only returns a valid mapping.
        # Then we add the edges to used_edges.
        
        # Wait, the solver `bt` checks `if e in used_edges`.
        # This is correct. It ensures the new tree doesn't use edges already taken by previous trees.
        
        # However, `used_hosts` is local to the tree. This is correct because vertices can be reused.
        
        # Let's run the solver.
        # We need to pass `used_edges` to the solver? It's in the closure.
        
        # The `solve_tree` function defined above is a bit messy.
        # Let's just inline a robust solver here.
        
        success = False
        # Try multiple random seeds for tie-breaking
        for seed in range(50):
            if time.time() > deadline:
                break
            # Re-run bt with different shuffle seed?
            # The bt function uses Random(time.time_ns()).
            # Let's just call it.
            
            # We need to reset host/assigned/used_hosts for each attempt?
            # Yes.
            
            # Let's define a fresh solver instance
            host = [-1] * k
            assigned = [False] * k
            used_hosts = set()
            
            def bt(idx, seed_val):
                if time.time() > deadline:
                    return False
                if idx == len(verts):
                    return True
                
                v = verts[idx]
                
                cands = []
                for h in range(n):
                    if h in used_hosts:
                        continue
                    ok = True
                    for u in adj[v]:
                        if assigned[u]:
                            e = (min(host[u], h), max(host[u], h))
                            if e in used_edges:
                                ok = False
                                break
                    if ok:
                        cands.append(h)
                        
                if not cands:
                    return False
                
                rng = Random(seed_val + idx)
                rng.shuffle(cands)
                
                for h in cands:
                    fc_ok = True
                    for w in adj[v]:
                        if not assigned[w]:
                            has_opt = False
                            for h_w in range(n):
                                if h_w in used_hosts or h_w == h:
                                    continue
                                ok_w = True
                                for u in adj[w]:
                                    if assigned[u] or u == v:
                                        hu = host[u] if assigned[u] else h
                                        e = (min(hu, h_w), max(hu, h_w))
                                        if e in used_edges:
                                            ok_w = False
                                            break
                                if ok_w:
                                    has_opt = True
                                    break
                            if not has_opt:
                                fc_ok = False
                                break
                    if not fc_ok:
                        continue
                    
                    host[v] = h
                    assigned[v] = True
                    used_hosts.add(h)
                    
                    if bt(idx + 1, seed_val):
                        return True
                    
                    host[v] = -1
                    assigned[v] = False
                    used_hosts.remove(h)
                    
                return False

            if bt(0, seed):
                # Success
                mapping = host[:]
                # Add edges to used_edges
                for u, v in trees[k_idx]:
                    a, b = mapping[u], mapping[v]
                    used_edges.add((min(a, b), max(a, b)))
                result[k_idx] = [int(x) for x in mapping]
                success = True
                break
        
        if not success:
            result[k_idx] = None

    return result

if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used = set()
    ok = True
    total = 0
    for k in range(1, 21):
        r = res[k - 1]
        if r is None:
            continue
        if len(r) != k or len(set(r)) != k:
            ok = False
        for v in r:
            if not isinstance(v, int) or v < 0 or v > 19:
                ok = False
        for (u, v) in trees[k - 1]:
            key = (min(r[u], r[v]), max(r[u], r[v]))
            if key in used:
                ok = False
            used.add(key)
        total += len(trees[k - 1])
    print(f"Valid: {ok}, Total edges: {total} / 190")
