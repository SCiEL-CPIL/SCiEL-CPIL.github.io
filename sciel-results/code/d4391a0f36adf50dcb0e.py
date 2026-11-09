"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    n = 15
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    best_adj = None
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
        
    def check_violations(adj):
        """
        Returns a list of edges (u, v) that are involved in at least one violation.
        A violation is a path s-u-v-t with s!=t and deg(s)==deg(t).
        """
        deg = [len(adj[i]) for i in range(n)]
        violating_edges = set()
        
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                
                # Collect degrees of neighbors of u (excluding v)
                # and neighbors of v (excluding u)
                # We need to check if there exist s in N(u)\{v} and t in N(v)\{u} 
                # such that s != t and deg(s) == deg(t).
                
                # Map degree to set of vertices for N(u)\{v} and N(v)\{u}
                deg_map_u = defaultdict(set)
                for s in adj[u]:
                    if s != v:
                        deg_map_u[deg[s]].add(s)
                        
                deg_map_v = defaultdict(set)
                for t in adj[v]:
                    if t != u:
                        deg_map_v[deg[t]].add(t)
                
                # Check intersection of degrees
                common_degrees = set(deg_map_u.keys()) & set(deg_map_v.keys())
                
                is_violating = False
                for d in common_degrees:
                    S_u = deg_map_u[d]
                    S_v = deg_map_v[d]
                    
                    # We need s in S_u, t in S_v such that s != t.
                    # This is impossible ONLY if S_u == S_v == {w} for some w.
                    # If S_u has > 1 element, we can pick s1, s2. If s1 in S_v, pick t=s2 (if s2!=s1). 
                    # Actually, simpler logic:
                    # If S_u and S_v are not identical singletons, there exists s != t.
                    # Case 1: |S_u| > 1 or |S_v| > 1.
                    #   If |S_u| > 1, pick s1, s2 in S_u.
                    #   If s1 in S_v, pick t=s1? No, s!=t.
                    #   Let's just check:
                    #   If S_u == S_v and len(S_u) == 1:
                    #       Let w be the single element. s=w, t=w. s==t. No violation from this degree d.
                    #   Else:
                    #       Violation exists.
                    
                    if S_u == S_v and len(S_u) == 1:
                        continue
                    else:
                        is_violating = True
                        break
                
                if is_violating:
                    violating_edges.add((u, v))
                    
        return list(violating_edges)

    def remove_worst_edges(adj, max_iter=5000):
        """
        Repeatedly remove edges involved in violations.
        Prioritizes edges involved in more violations if possible, 
        but for simplicity, we just pick one from the violating set.
        To be smarter, we could count how many times an edge appears in violations,
        but check_violations currently just returns a set.
        Let's stick to random selection from violating edges for diversity, 
        or deterministic if we want reproducibility.
        """
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            viol_edges = check_violations(adj)
            if not viol_edges:
                break
            # Pick a random violating edge to remove
            u, v = random.choice(viol_edges)
            adj[u].discard(v)
            adj[v].discard(u)
        return adj

    def greedy_add_edges(adj, max_time=5.0):
        """
        Try to add missing edges greedily.
        """
        t_start = time.time()
        improved = True
        while improved:
            if time.time() - t_start > max_time:
                break
            improved = False
            # Shuffle missing edges to avoid bias
            missing = [(u, v) for u, v in all_edges if v not in adj[u]]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() - t_start > max_time:
                    break
                # Try adding
                adj[u].add(v)
                adj[v].add(u)
                if not check_violations(adj):
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        return adj

    def local_search(adj, max_time=10.0):
        """
        Combine removal and addition.
        """
        t_start = time.time()
        while time.time() - t_start < max_time:
            # Fix violations
            adj = remove_worst_edges(adj, max_iter=1000)
            
            # Add edges
            adj = greedy_add_edges(adj, max_time=2.0)
            
            # Check if we can do better by removing a random edge and re-optimizing
            # This helps escape local optima
            if random.random() < 0.3:
                present = [(u, v) for u, v in all_edges if v in adj[u]]
                if present:
                    u, v = random.choice(present)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    # Try to add back or add others
                    adj = greedy_add_edges(adj, max_time=1.0)
        
        return adj

    # Strategy 1: Multiple random restarts
    for seed in range(100):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(seed)
        
        # Random initialization
        density = random.uniform(0.4, 0.8)
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            if random.random() < density:
                adj[u].add(v)
                adj[v].add(u)
        
        # Optimize
        adj = local_search(adj, max_time=5.0)
        
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = [set(s) for s in adj]

    # Strategy 2: Start with complete graph and remove
    if time.time() < deadline - 15.0:
        random.seed(999)
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        adj = remove_worst_edges(adj, max_iter=5000)
        adj = greedy_add_edges(adj, max_time=5.0)
        
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = [set(s) for s in adj]

    # Strategy 3: Structured initialization (distinct degrees)
    # Try to build a graph with degrees 1, 2, ..., 14 (skip 0)
    # This is a heuristic.
    if time.time() < deadline - 15.0:
        random.seed(888)
        adj = [set() for _ in range(n)]
        # Sort vertices by index, try to connect i to j if j > i and i < j - i? No.
        # Just try a "threshold" graph
        for i in range(n):
            for j in range(i+1, n):
                if j - i <= 5: # Connect close vertices
                    adj[i].add(j)
                    adj[j].add(i)
        
        adj = local_search(adj, max_time=5.0)
        
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = [set(s) for s in adj]

    # Fallback
    if best_adj is None:
        best_adj = [set() for _ in range(n)]
        for v in range(1, n):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = n - 1

    # Final polish on the best found
    if time.time() < deadline - 2.0:
        adj = [set(s) for s in best_adj]
        adj = greedy_add_edges(adj, max_time=2.0)
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = adj

    # Convert to list of lists
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
                
    return edges

# EVOLVE-BLOCK-END
