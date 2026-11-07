def construct():
    n = 15
    import time
    import random
    import itertools
    
    deadline = time.time() + 75
    
    def check_validity(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge (u,v), check all paths s-u-v-t
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def get_degrees(edge_set, n):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def get_adj(edge_set, n):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Strategy: Build graph by assigning distinct degree targets
    # and constructing edges to match them while maintaining validity
    
    # Approach: Try to build a graph where all degrees are distinct
    # If all degrees are distinct, the constraint is automatically satisfied
    # (no two endpoints can have equal degree)
    
    # For n=15, we need 15 distinct degrees from 0 to 14
    # But a graph with degree sequence (0,1,2,...,14) is not graphical
    # (sum = 105, which is odd - impossible)
    
    # Let's try: make most degrees distinct, with at most a few equal
    # If two vertices have the same degree, they must not be endpoints of any 3-path
    
    # Better approach: Use a near-complete graph and remove edges strategically
    
    # Start with complete graph K15 (105 edges) and remove edges to satisfy constraint
    # This is hard, so let's try a different construction
    
    # Key insight: If we can make all 15 degrees distinct, we're done
    # Max possible distinct degrees: need a graphical sequence
    
    # Let's try degrees: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,14 (one pair equal)
    # Sum = 1+2+...+13 + 14+14 = 91 + 28 = 119 (odd - not graphical)
    
    # Try: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 - sum=105 (odd, not graphical)
    
    # Try: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 - sum=120 (even, but max degree 15 > 14)
    
    # For n=15, max degree is 14. Let's try:
    # 1,2,3,4,5,6,7,8,9,10,11,12,13,14,14 - sum=119 (odd)
    # 2,3,4,5,6,7,8,9,10,11,12,13,14,14,14 - sum=131 (odd)
    
    # It seems hard to get all distinct. Let's use a different strategy:
    # Build greedily, checking validity at each step
    
    # Start with K_{7,8} bipartite (56 edges) - all A vertices have deg 8, all B have deg 7
    # This is valid since in any 3-path s-u-v-t, s and t are in different parts
    # so deg(s)=8 and deg(t)=7 (or vice versa), never equal
    
    # Now try to add edges within A and within B
    
    A = list(range(7))
    B = list(range(7, 15))
    
    edges = set()
    for u in A:
        for v in B:
            edges.add((u, v))
    
    # Generate candidate edges within A and within B
    candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidates.append((i, j))
    
    random.seed(42)
    random.shuffle(candidates)
    
    # Greedy addition with validity check
    for e in candidates:
        if time.time() > deadline - 20:
            break
        if e in edges:
            continue
        test = edges | {e}
        if check_validity(test, n):
            edges = test
    
    current_best = len(edges)
    best_edges = set(edges)
    
    # Try multiple random orderings
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        random.shuffle(candidates)
        
        edges_trial = set()
        for u in A:
            for v in B:
                edges_trial.add((u, v))
        
        for e in candidates:
            if time.time() > deadline - 10:
                break
            if e in edges_trial:
                continue
            test = edges_trial | {e}
            if check_validity(test, n):
                edges_trial = test
        
        if len(edges_trial) > current_best:
            current_best = len(edges_trial)
            best_edges = set(edges_trial)
    
    # Try a different approach: start from scratch with greedy
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        
        random.seed(1000 + seed)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        edges_trial = set()
        for e in all_pairs:
            if time.time() > deadline - 5:
                break
            test = edges_trial | {e}
            if check_validity(test, n):
                edges_trial = test
        
        if len(edges_trial) > current_best:
            current_best = len(edges_trial)
            best_edges = set(edges_trial)
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
