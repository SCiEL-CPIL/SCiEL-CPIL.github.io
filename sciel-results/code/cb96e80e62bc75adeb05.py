import time
import random

def _is_valid(adj, n):
    """Check if graph satisfies the constraint."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
    return True

def _count_edges(adj):
    return sum(len(adj[i]) for i in range(len(adj))) // 2

def construct():
    n = 15
    deadline = time.time() + 80  # 80s out of 90s budget
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Start with K_{7,8} (56 edges) and try to improve
    # K_{7,8} is valid.
    adj = [set() for _ in range(n)]
    # Part A: 0..6, Part B: 7..14
    for i in range(7):
        for j in range(7, 15):
            adj[i].add(j)
            adj[j].add(i)
    
    if _is_valid(adj, n):
        count = _count_edges(adj)
        if count > best_count:
            best_count = count
            best_edges = [[i, j] for i in range(n) for j in adj[i] if i < j]
    
    # Strategy 2: Local search from K_{7,8}
    # Try adding random edges, if invalid, remove random edge from the violation set or the added edge.
    current_adj = [set(x) for x in adj]
    current_count = _count_edges(current_adj)
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Try to add an edge
        # Pick a random non-edge
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if j not in current_adj[i]:
                    non_edges.append((i, j))
        
        if not non_edges:
            break
            
        u, v = random.choice(non_edges)
        current_adj[u].add(v)
        current_adj[v].add(u)
        
        if _is_valid(current_adj, n):
            current_count += 1
            if current_count > best_count:
                best_count = current_count
                best_edges = [[i, j] for i in range(n) for j in current_adj[i] if i < j]
        else:
            # Invalid, try to fix by removing an edge
            # For simplicity, just remove the added edge
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            # Sometimes, try removing a different edge to allow this one?
            # This is complex. Stick to simple hill climbing.
    
    # Strategy 3: Greedy removal from K_15 (fallback or alternative)
    # This might find a different local maximum
    adj2 = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj2[i].add(j)
            adj2[j].add(i)
            
    # Greedy removal
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        deg = [len(adj2[i]) for i in range(n)]
        
        # Find violations
        viol_edges = set()
        for u in range(n):
            for v in adj2[u]:
                if u >= v:
                    continue
                for s in adj2[u]:
                    if s == v:
                        continue
                    ds = deg[s]
                    for t in adj2[v]:
                        if t == u or t == s:
                            continue
                        if deg[t] == ds:
                            viol_edges.add((u, v))
                            break
                    if (u, v) in viol_edges:
                        break
                if (u, v) in viol_edges:
                    break
        
        if not viol_edges:
            break
            
        # Remove one violating edge
        u, v = random.choice(list(viol_edges))
        adj2[u].discard(v)
        adj2[v].discard(u)
        
        if _is_valid(adj2, n):
            count = _count_edges(adj2)
            if count > best_count:
                best_count = count
                best_edges = [[i, j] for i in range(n) for j in adj2[i] if i < j]

    # Ensure return type
    if not best_edges:
        best_edges = [[0, v] for v in range(1, n)]
        
    return best_edges
