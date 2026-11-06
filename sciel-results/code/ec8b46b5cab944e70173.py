"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Fast bitmask helpers
    def popcount(x):
        return bin(x).count('1')

    def build_polarity_graph_q5():
        """Construct the polarity graph of PG(2,5).
        Points: 31 vertices (projective points of F_5^3).
        Adjacency: (x,y,z) ~ (x',y',z') iff xx' + yy' + zz' = 0 in F_5.
        This graph is C4-free.
        """
        # Enumerate projective points: representatives with first non-zero coordinate = 1
        points = []
        # z=1, x,y in 0..4 -> 25 points
        for x in range(5):
            for y in range(5):
                points.append((x, y, 1))
        # z=0, x=1, y in 0..4 -> 5 points
        for y in range(5):
            points.append((1, y, 0))
        # z=0, x=0, y=1 -> 1 point
        points.append((0, 1, 0))
        
        n = len(points)  # 31
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            xi, yi, zi = points[i]
            for j in range(i+1, n):
                xj, yj, zj = points[j]
                if (xi*xj + yi*yj + zi*zj) % 5 == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj, points

    def check_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u] & adj[v]
                    if len(common) > 1:
                        return False
        return True

    def has_independent_set(adj, k, time_limit=None):
        """Returns True if there is an independent set of size k."""
        n = len(adj)
        if k <= 0:
            return True
        if n < k:
            return False
        
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            masks[i] = mask
        
        sys.setrecursionlimit(10000)
        
        def search(v, remaining, size):
            if time_limit is not None and time.time() > time_limit:
                return False
            if size == 0:
                return True
            if v == n:
                return False
            if popcount(remaining) < size:
                return False
            
            # Try to include v
            if (remaining & (1 << v)) != 0:
                new_remaining = remaining & ~(masks[v] | (1 << v))
                if search(v + 1, new_remaining, size - 1):
                    return True
            
            # Try to exclude v
            if search(v + 1, remaining, size):
                return True
            
            return False
        
        all_mask = (1 << n) - 1
        return search(0, all_mask, k)

    def try_construct_random(n, seed, deadline):
        """Try to construct a C4-free graph on n vertices with alpha < 11 using random greedy."""
        random.seed(seed)
        adj = [set() for _ in range(n)]
        
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline - 5:
                break
            
            # Check if adding (u,v) creates a C4
            # Condition: for all a in N(v), N(u) & N(a) must be empty (since v becomes a common neighbor of u and a)
            # and for all b in N(u), N(v) & N(b) must be empty
            # Also N(u) & N(v) must have size <= 1 (but actually if we're adding edge, they become adjacent,
            # and common neighbors of adjacent vertices don't form C4 by themselves, but the ripple matters)
            
            # Simple sufficient check:
            # 1. |N(u) & N(v)| <= 1
            common_uv = adj[u] & adj[v]
            if len(common_uv) > 1:
                continue
            
            # 2. For all a in N(v): |N(u) & N(a)| == 0
            valid = True
            for a in adj[v]:
                if adj[u] & adj[a]:
                    valid = False
                    break
            if not valid:
                continue
            
            # 3. For all b in N(u): |N(v) & N(b)| == 0
            for b in adj[u]:
                if adj[v] & adj[b]:
                    valid = False
                    break
            if not valid:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
        
        # Check C4-free
        if not check_c4_free(adj):
            return None
        
        # Check alpha
        if has_independent_set(adj, 11, time_limit=deadline - 10):
            return None
        
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append([u, v])
        return {'n': n, 'edges': edge_list}

    def try_extend_base(base_adj, target_n, seed, deadline):
        """Try to extend the base graph to target_n vertices."""
        random.seed(seed)
        n_base = len(base_adj)
        if target_n <= n_base:
            return None
        
        adj = [set(base_adj[i]) for i in range(n_base)]
        # Add new vertices
        for i in range(n_base, target_n):
            adj.append(set())
        
        # Try to connect new vertices to existing ones
        # New vertices start with no edges
        # We want to add edges from new vertices to old vertices
        # such that C4-freeness is preserved and alpha < 11
        
        # For each new vertex, try to connect to a random subset of old vertices
        # such that the C4 condition holds
        
        for new_v in range(n_base, target_n):
            if time.time() > deadline - 5:
                break
            
            # Get all old vertices
            old_verts = list(range(n_base))
            random.shuffle(old_verts)
            
            # Try to add edges to old vertices one by one
            for old_v in old_verts:
                if time.time() > deadline - 5:
                    break
                
                # Check if adding edge (new_v, old_v) is valid
                # Condition: for all a in N(old_v), |N(new_v) & N(a)| == 0
                # (since new_v is not adjacent to any old vertex yet, N(new_v) is empty or has other new vertices)
                # Actually, new_v might already be connected to other new vertices? No, we're only connecting to old.
                # Wait, we might have connected new_v to other old vertices already.
                
                # Let's check properly:
                # Adding edge (new_v, old_v):
                # 1. For all a in N(old_v): |N(new_v) & N(a)| == 0
                #    But N(new_v) currently contains some old vertices (those already connected).
                #    Actually, N(new_v) contains vertices already connected to new_v.
                #    Let's denote S = N(new_v) (currently).
                #    We need: for all a in N(old_v), S & N(a) == empty.
                #    But S consists of old vertices. So we need: for all a in N(old_v), for all s in S, s not in N(a).
                #    i.e., no two vertices in S ∪ {new_v} have a common neighbor in N(old_v) ∩ N(s) ... 
                #    This is getting complex. Let's use the general C4 check.
                
                # Simple approach: tentatively add the edge, check C4-free, if not, remove.
                adj[new_v].add(old_v)
                adj[old_v].add(new_v)
                
                if not check_c4_free(adj):
                    adj[new_v].remove(old_v)
                    adj[old_v].remove(new_v)
        
        # Check alpha
        if has_independent_set(adj, 11, time_limit=deadline - 10):
            return None
        
        edge_list = []
        for u in range(target_n):
            for v in adj[u]:
                if u < v:
                    edge_list.append([u, v])
        return {'n': target_n, 'edges': edge_list}

    # Build the base graph (q=5 polarity graph, n=31)
    base_adj, _ = build_polarity_graph_q5()
    
    # Verify C4-free
    if not check_c4_free(base_adj):
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Check alpha of base graph
    base_alpha_ok = not has_independent_set(base_adj, 11, time_limit=deadline - 10)
    
    # Try to extend the base graph
    best_result = None
    
    # First, try random construction for n=43 down to 39
    for n in range(43, 38, -1):
        if time.time() > deadline - 15:
            break
        for seed in range(30):
            if time.time() > deadline - 10:
                break
            res = try_construct_random(n, seed, deadline)
            if res is not None:
                if best_result is None or res['n'] > best_result['n']:
                    best_result = res
                if res['n'] >= n:
                    return res
    
    # Try to extend the base graph
    if base_alpha_ok:
        for target_n in range(43, 38, -1):
            if time.time() > deadline - 10:
                break
            for seed in range(20):
                if time.time() > deadline - 5:
                    break
                res = try_extend_base(base_adj, target_n, seed, deadline)
                if res is not None:
                    if best_result is None or res['n'] > best_result['n']:
                        best_result = res
                    if res['n'] >= target_n:
                        return res
    
    # If we found a result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return the base graph if it's valid
    if base_alpha_ok:
        edge_list = []
        for u in range(len(base_adj)):
            for v in base_adj[u]:
                if u < v:
                    edge_list.append([u, v])
        return {'n': len(base_adj), 'edges': edge_list}
    
    # Final fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
