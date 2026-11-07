import time
from collections import deque

P = {'n': 16}
_start = time.time()
_DEADLINE = 75.0  # safety margin before 90s kill

def _bfs_dist(adj, src):
    n = len(adj)
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _diameter(adj):
    n = len(adj)
    diam = 0
    for i in range(n):
        d = _bfs_dist(adj, i)
        if -1 in d:
            return float('inf')
        diam = max(diam, max(d))
    return diam

def _is_valid(adj):
    n = len(adj)
    # Check diameter exactly 2
    if _diameter(adj) != 2:
        return False
    # Check every edge is essential
    for i in range(n):
        for j in range(i+1, n):
            if j in adj[i]:
                # Remove edge (i,j)
                adj2 = [lst[:] for lst in adj]
                adj2[i].remove(j)
                adj2[j].remove(i)
                d = _diameter(adj2)
                if d == 2:
                    return False
    return True

def construct():
    n = P['n']
    
    # C5 blow-up construction:
    # Partition 16 vertices into 5 groups along a cycle.
    # Connect all vertices in group i to all in group (i+1)%5.
    # This gives diameter 2 and every edge is critical.
    
    # Try multiple group size assignments to maximize edges
    # Edges = s0*s1 + s1*s2 + s2*s3 + s3*s4 + s4*s0
    # with s0+s1+s2+s3+s4 = 16, si >= 1
    
    import itertools
    
    best_edges = None
    best_count = 0
    
    # Generate all compositions of 16 into 5 positive parts
    # Use stars and bars: 15 dividers, choose 4
    for comp in itertools.combinations_with_replacement(range(1, 16), 4):
        parts = [comp[0]]
        for i in range(1, 4):
            parts.append(comp[i] - comp[i-1])
        parts.append(16 - comp[3])
        if any(p < 1 for p in parts):
            continue
        # Check it's a valid composition
        if sum(parts) != 16:
            continue
        # Compute edge count
        s = parts
        ec = s[0]*s[1] + s[1]*s[2] + s[2]*s[3] + s[3]*s[4] + s[4]*s[0]
        if ec > best_count:
            best_count = ec
            best_parts = s[:]
    
    if best_parts is None:
        # Fallback: star
        return [[0, v] for v in range(1, n)]
    
    # Assign vertices to groups
    groups = []
    idx = 0
    for size in best_parts:
        groups.append(list(range(idx, idx + size)))
        idx += size
    
    # Build edges: connect all in group i to all in group (i+1)%5
    edges = []
    for i in range(5):
        j = (i + 1) % 5
        for u in groups[i]:
            for v in groups[j]:
                if u < v:
                    edges.append([u, v])
                else:
                    edges.append([v, u])
    
    # Validate
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj = [sorted(lst) for lst in adj]
    
    if _is_valid(adj):
        return edges
    
    # Fallback: star
    return [[0, v] for v in range(1, n)]
