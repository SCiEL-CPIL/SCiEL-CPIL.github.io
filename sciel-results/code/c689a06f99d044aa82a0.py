import time
import random

def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_constraint(edge_set, n):
    """Check if no simple 3-edge path has endpoints with equal degree. Returns True if valid."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u, v in edge_set:
        du = adj[u]
        dv = adj[v]
        for s in du:
            if s == v:
                continue
            ds = deg[s]
            for t in dv:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def find_violations(edge_set, n):
    """Return list of violating paths (s, u, v, t)."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    violations = []
    for u, v in edge_set:
        du = adj[u]
        dv = adj[v]
        for s in du:
            if s == v:
                continue
            ds = deg[s]
            for t in dv:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    violations.append((s, u, v, t))
    return violations

def construct():
    start = time.time()
    deadline = start + 85
    n = 15
    rng = random.Random(12345)

    # 1. Start with complete graph
    edge_set = set((i, j) for i in range(n) for j in range(i + 1, n))
    
    # 2. Repair: Remove edges to satisfy constraint
    # We will iteratively find violations and remove edges
    # To be efficient, we can just check validity and if invalid, find one violation and remove one of its edges.
    # Better: Find all violations, count edge frequencies, remove the most frequent edge in violations.
    
    while time.time() < deadline - 10:
        if check_constraint(edge_set, n):
            break
            
        violations = find_violations(edge_set, n)
        if not violations:
            break
            
        # Count edge frequencies in violations
        edge_counts = {}
        for s, u, v, t in violations:
            e1 = (min(s, u), max(s, u))
            e2 = (min(u, v), max(u, v))
            e3 = (min(v, t), max(v, t))
            edge_counts[e1] = edge_counts.get(e1, 0) + 1
            edge_counts[e2] = edge_counts.get(e2, 0) + 1
            edge_counts[e3] = edge_counts.get(e3, 0) + 1
            
        # Remove the edge with highest count
        # If tie, pick random
        max_count = max(edge_counts.values())
        candidates = [e for e, c in edge_counts.items() if c == max_count]
        edge_to_remove = rng.choice(candidates)
        
        edge_set.remove(edge_to_remove)
        
    # Now edge_set is valid.
    # 3. Local Search: Try to add edges back
    adj = _build_adj(edge_set, n)
    
    # Helper to check if adding edge (u,v) keeps validity
    # We can just tentatively add, check, and revert.
    # Since N=15, check is fast.
    
    def try_add(u, v):
        if (u, v) in edge_set:
            return False
        if (v, u) in edge_set:
            return False
        edge_set.add((u, v))
        if check_constraint(edge_set, n):
            return True
        else:
            edge_set.remove((u, v))
            return False

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding all missing edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            if time.time() > deadline - 5:
                break
            if try_add(u, v):
                improved = True
                adj = _build_adj(edge_set, n) # Update adj for potential future optimizations, though check_constraint rebuilds it
        
    # 4. Swap Search: Remove one, add another
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            # Try swapping: remove e1, add e2
            # To make this efficient, we can iterate over missing edges e2, and for each, try to remove an edge e1 that allows e2 to be added.
            # Or just random swaps.
            
            # Let's try random swaps
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                if not edge_set:
                    break
                e1 = rng.choice(list(edge_set))
                # Pick a missing edge
                missing = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edge_set and (j, i) not in edge_set]
                if not missing:
                    break
                e2 = rng.choice(missing)
                
                # Try swap
                edge_set.remove(e1)
                edge_set.add(e2)
                if check_constraint(edge_set, n):
                    improved = True
                else:
                    # Revert
                    edge_set.remove(e2)
                    edge_set.add(e1)
                    
    # Final result
    result = []
    seen = set()
    for u, v in edge_set:
        u = int(u); v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
            
    return result
