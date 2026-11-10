import time
import random
from itertools import combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_bitmask(adj, n=16):
    colors = [-1] * n
    memo = {}
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        
        # Memoization key: uncolored mask + colors of colored vertices
        # We only need to track which colors are used by neighbors of uncolored vertices
        # Actually, let's use a simpler approach: track (uncolored_mask, tuple of colors for colored vertices)
        # But that's too expensive. Let's use a different memo strategy.
        
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        
        return True
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask)

def is_not_3_colorable_fast(adj, n=16):
    # Quick check: if any vertex has degree < 2, it's 3-colorable (remove it, color rest, add it back)
    # More generally, if graph has a vertex of degree <= 2, it's 3-colorable
    for v in range(n):
        deg = bin(adj[v]).count('1')
        if deg <= 2:
            return False
    
    # Try greedy coloring with multiple orderings
    for trial in range(30):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        success = True
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                success = False
                break
        if success:
            return False
    
    # Exact check
    return is_not_3_colorable_bitmask(adj, n)

def build_groetzsch_edges():
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    for u, v in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
        edges.append([5+u, v])
        edges.append([5+v, u])
    for u in range(5):
        edges.append([10, 5+u])
    return edges

def get_groetzsch_adj(n=11):
    edges = build_groetzsch_edges()
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def generate_blowup_distributions(num_extra, num_original):
    dists = []
    def rec(rem, idx, current):
        if idx == num_original - 1:
            current.append(rem)
            dists.append(tuple(current))
            current.pop()
            return
        for i in range(rem + 1):
            current.append(i)
            rec(rem - i, idx + 1, current)
            current.pop()
    rec(num_extra, 0, [])
    return dists

def local_search(adj, current_edges, n=16, deadline=None):
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        
        # Try adding edges
        candidates_add = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates_add.append((u, v))
        
        random.shuffle(candidates_add)
        
        for u, v in candidates_add:
            if deadline and time.time() > deadline:
                break
            
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            current_edges.append([u, v])
            
            if is_not_3_colorable_fast(adj, n):
                improved = True
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                current_edges.pop()
        
        if improved:
            continue
        
        # Try edge swaps: remove one edge, add another
        candidates_swap = []
        edge_set = set()
        for e in current_edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        # This edge can be added
                        for e1, e2 in edge_set:
                            # Check if removing (e1,e2) and adding (u,v) keeps triangle-free
                            # Removing an edge always preserves triangle-free
                            # Adding (u,v) requires adj[u] & adj[v] == 0 after removal
                            # But we already checked adj[u] & adj[v] == 0 in current graph
                            # After removing (e1,e2), adj[u] and adj[v] can only lose neighbors
                            # So the condition still holds
                            # However, we need to ensure we don't create a triangle
                            # The only way adding (u,v) creates a triangle is if there's w with (u,w) and (v,w) in graph
                            # We already checked that, and removing edges doesn't add neighbors
                            # So any swap that adds a valid edge is valid
                            candidates_swap.append(((e1, e2), (u, v)))
        
        if candidates_swap:
            random.shuffle(candidates_swap)
            for (rem_u, rem_v), (add_u, add_v) in candidates_swap[:200]:  # Limit swaps per iteration
                if deadline and time.time() > deadline:
                    break
                
                # Remove edge (rem_u, rem_v)
                adj[rem_u] ^= 1 << rem_v
                adj[rem_v] ^= 1 << rem_u
                # Find and remove from current_edges
                for i, e in enumerate(current_edges):
                    if (min(e[0], e[1]), max(e[0], e[1])) == (rem_u, rem_v):
                        current_edges.pop(i)
                        break
                
                # Add edge (add_u, add_v)
                adj[add_u] |= 1 << add_v
                adj[add_v] |= 1 << add_u
                current_edges.append([add_u, add_v])
                
                if is_not_3_colorable_fast(adj, n):
                    improved = True
                    break
                else:
                    # Revert
                    adj[add_u] ^= 1 << add_v
                    adj[add_v] ^= 1 << add_u
                    current_edges.pop()
                    adj[rem_u] |= 1 << rem_v
                    adj[rem_v] |= 1 << rem_u
                    current_edges.append([rem_u, rem_v])
    
    return current_edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    n = 16
    num_original = 11
    num_extra = n - num_original
    
    g_adj = get_groetzsch_adj(num_original)
    
    best_edges = []
    best_count = 0
    
    can_saturate = [[False]*num_original for _ in range(num_original)]
    for i in range(num_original):
        for j in range(i+1, num_original):
            if not (g_adj[i] & (1 << j)):
                if (g_adj[i] & g_adj[j]) == 0:
                    can_saturate[i][j] = True
                    can_saturate[j][i] = True
    
    dists = generate_blowup_distributions(num_extra, num_original)
    
    deadline = START_TIME + TIME_LIMIT
    
    for dist in dists:
        if time.time() > deadline:
            break
            
        sizes = [1 + d for d in dist]
        
        orig_map = [0] * n
        current_idx = 0
        for i in range(num_original):
            for _ in range(sizes[i]):
                orig_map[current_idx] = i
                current_idx += 1
        
        edges = []
        adj = [0] * n
        
        # Build blow-up edges
        for i in range(num_original):
            for j in range(i+1, num_original):
                if g_adj[i] & (1 << j):
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            edges.append([u, v])
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
        
        # Add saturating edges
        for i in range(num_original):
            for j in range(i+1, num_original):
                if can_saturate[i][j]:
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            if not (adj[u] & (1 << v)):
                                edges.append([u, v])
                                adj[u] |= 1 << v
                                adj[v] |= 1 << u
        
        if not is_triangle_free_bitmask(adj, n):
            continue
            
        if is_not_3_colorable_fast(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
                
    if best_edges:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        
        # Run local search multiple times with different shuffles
        for trial in range(10):
            if time.time() > deadline:
                break
            random.seed(42 + trial * 137)
            local_search(adj, current_edges, n, deadline)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                adj = build_adj(best_edges, n)
            else:
                adj = build_adj(best_edges, n)
                
    if not best_edges:
        base_edges = build_groetzsch_edges()
        adj = build_adj(base_edges, n)
        current_edges = list(base_edges)
        for v in range(11, 16):
            candidates = list(range(11))
            random.shuffle(candidates)
            for c in candidates:
                if time.time() - START_TIME > TIME_LIMIT:
                    return current_edges
                if adj[v] & adj[c] == 0:
                    current_edges.append([v, c])
                    adj[v] |= 1 << c
                    adj[c] |= 1 << v
        best_edges = current_edges
        
    return best_edges
