"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def fast_is_valid(edge_set, n):
        """Faster validity check using degree sets for each edge's endpoint neighborhoods."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            # Get degree sets of neighbors (excluding each other)
            degs_u = set()
            for s in adj[u]:
                if s != v:
                    degs_u.add(deg[s])
            for t in adj[v]:
                if t != u:
                    if deg[t] in degs_u:
                        # Need to check if there exists s in adj[u]\{v} with deg[s]==deg[t] and s != t
                        # If t is not in adj[u], then any s with deg[s]==deg[t] works
                        if t not in adj[u]:
                            return False
                        else:
                            # t is in adj[u], so we need another s with same degree
                            # Check if there's an s in adj[u]\{v,t} with deg[s]==deg[t]
                            for s in adj[u]:
                                if s != v and s != t and deg[s] == deg[t]:
                                    return False
            # Also need symmetric check: for s in adj[u]\{v}, check if deg[s] matches any t in adj[v]\{u}
            # The above loop covers this partially. Let's do it more carefully.
            pass
        return True

    # Actually, let's use a cleaner fast validity check
    def is_valid_fast(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        for (u, v) in edge_set:
            # For edge (u,v), check all pairs (s,t) where s in N(u)\{v}, t in N(v)\{u}, s!=t
            # Constraint: deg[s] != deg[t]
            # Equivalent: the multiset of degrees in N(u)\{v} and N(v)\{u} must be disjoint
            # except that if a vertex x is in both N(u)\{v} and N(v)\{u}, it can match itself.
            # More precisely: for each degree value d, if there's an s in N(u)\{v} with deg[s]=d
            # and a t in N(v)\{u} with deg[t]=d, then we need s==t (i.e., the vertex is common).
            
            # Build mapping: for N(u)\{v}, which vertices have which degrees
            # For N(v)\{u}, which vertices have which degrees
            
            # Simple approach: collect (deg, vertex) for each side
            side_u = [(deg[s], s) for s in adj[u] if s != v]
            side_v = [(deg[t], t) for t in adj[v] if t != u]
            
            # For each pair, check constraint
            for ds, s in side_u:
                for dt, t in side_v:
                    if s == t:
                        continue
                    if ds == dt:
                        return False
        return True

    # Start with K_{7,8} as baseline (56 edges, proven valid)
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    
    best_edges = set(base_edges)
    best_count = len(base_edges)

    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Greedy edge addition
    current = set(base_edges)
    missing = [e for e in all_possible if e not in current]
    
    # Try adding edges in a smart order: prefer edges that create distinct degrees
    # First, try intra-A edges, then intra-B edges, then re-add removed bipartite edges
    random.seed(42)
    random.shuffle(missing)
    
    # Try each missing edge
    for e in missing:
        if time.time() > deadline:
            break
        if e in current:
            continue
        trial = current | {e}
        if is_valid_fast(trial, n):
            current = trial
            if len(current) > best_count:
                best_edges = set(current)
                best_count = len(current)
    
    # Phase 2: Local search with swaps
    # Try to improve by swapping edges
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        current_list = list(current)
        random.shuffle(current_list)
        
        for rem in current_list:
            if time.time() > deadline:
                break
            
            trial = current - {rem}
            
            # Try adding each missing edge
            for add in all_possible:
                if add in trial:
                    continue
                if add == rem:
                    continue
                
                trial2 = trial | {add}
                if len(trial2) <= len(current):
                    continue
                
                if is_valid_fast(trial2, n):
                    current = trial2
                    improved = True
                    if len(current) > best_count:
                        best_edges = set(current)
                        best_count = len(current)
                    break
            
            if improved:
                break
        
        if not improved:
            # Try 2-edge swaps: remove 2, add 2
            found = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                rem1, rem2 = random.sample(list(current), 2)
                trial = current - {rem1, rem2}
                
                for add1 in all_possible:
                    if add1 in trial or add1 == rem1 or add1 == rem2:
                        continue
                    trial1 = trial | {add1}
                    for add2 in all_possible:
                        if add2 in trial1 or add2 == rem1 or add2 == rem2 or add2 == add1:
                            continue
                        trial2 = trial1 | {add2}
                        if len(trial2) <= len(current):
                            continue
                        if is_valid_fast(trial2, n):
                            current = trial2
                            found = True
                            if len(current) > best_count:
                                best_edges = set(current)
                                best_count = len(current)
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                break
    
    # Phase 3: Try different initial partitions
    # Maybe K_{6,9} or other splits work better with intra-edges
    for split_a in [5, 6, 7, 8, 9]:
        if time.time() > deadline:
            break
        
        A2 = list(range(split_a))
        B2 = list(range(split_a, n))
        base2 = set()
        for a in A2:
            for b in B2:
                base2.add((a, b))
        
        if len(base2) <= best_count:
            continue
        
        # Verify base is valid
        if not is_valid_fast(base2, n):
            continue
        
        cur = set(base2)
        miss2 = [e for e in all_possible if e not in cur]
        random.shuffle(miss2)
        
        for e in miss2:
            if time.time() > deadline:
                break
            if e in cur:
                continue
            trial = cur | {e}
            if is_valid_fast(trial, n):
                cur = trial
        
        if len(cur) > best_count:
            best_edges = set(cur)
            best_count = len(cur)
    
    # Final verification
    if not is_valid_fast(best_edges, n):
        # Fallback to K_{7,8}
        best_edges = set(base_edges)
        best_count = len(base_edges)
    
    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final

# EVOLVE-BLOCK-END
