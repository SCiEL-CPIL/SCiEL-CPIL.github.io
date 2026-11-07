import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    def build_A(top_signs, bot_signs, match_signs):
        """Build n×n signed adjacency matrix using prism structure."""
        A = np.zeros((n, n))
        for i in range(m):
            j = (i + 1) % m
            A[i, j] = top_signs[i]
            A[j, i] = top_signs[i]
        for i in range(m):
            j = (i + 1) % m
            A[m + i, m + j] = bot_signs[i]
            A[m + j, m + i] = bot_signs[i]
        for i in range(m):
            A[i, m + i] = match_signs[i]
            A[m + i, i] = match_signs[i]
        return A
    
    def spectral_radius(top_signs, bot_signs, match_signs):
        A = build_A(top_signs, bot_signs, match_signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(top_signs, bot_signs, match_signs):
        """Check if 2-lift is connected via BFS."""
        lift_n = 2 * n
        visited = np.zeros(lift_n, dtype=bool)
        adj = [[] for _ in range(lift_n)]
        for i in range(m):
            j = (i + 1) % m
            s = top_signs[i]
            if s == 1:
                adj[i*2].append(j*2); adj[j*2].append(i*2)
                adj[i*2+1].append(j*2+1); adj[j*2+1].append(i*2+1)
            else:
                adj[i*2].append(j*2+1); adj[j*2+1].append(i*2)
                adj[i*2+1].append(j*2); adj[j*2].append(i*2+1)
        for i in range(m):
            j = (i + 1) % m
            s = bot_signs[i]
            if s == 1:
                adj[(m+i)*2].append((m+j)*2); adj[(m+j)*2].append((m+i)*2)
                adj[(m+i)*2+1].append((m+j)*2+1); adj[(m+j)*2+1].append((m+i)*2+1)
            else:
                adj[(m+i)*2].append((m+j)*2+1); adj[(m+j)*2+1].append((m+i)*2)
                adj[(m+i)*2+1].append((m+j)*2); adj[(m+j)*2].append((m+i)*2+1)
        for i in range(m):
            s = match_signs[i]
            if s == 1:
                adj[i*2].append((m+i)*2); adj[(m+i)*2].append(i*2)
                adj[i*2+1].append((m+i)*2+1); adj[(m+i)*2+1].append(i*2+1)
            else:
                adj[i*2].append((m+i)*2+1); adj[(m+i)*2+1].append(i*2)
                adj[i*2+1].append((m+i)*2); adj[(m+i)*2].append(i*2+1)
        
        q = deque([0])
        visited[0] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        return visited.all()
    
    def make_signs_array(top_signs, bot_signs, match_signs):
        signs = np.zeros(num_edges, dtype=int)
        for i in range(m):
            signs[i] = top_signs[i]
        for i in range(m):
            signs[m + i] = bot_signs[i]
        for i in range(m):
            signs[2*m + i] = match_signs[i]
        return signs
    
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate all 2^(m+1) switching classes
    # Gauge: matching = +1, top[0..m-2] = +1
    # Free: top[m-1] (top_prod), ladder_prods[0..m-1]
    for pattern in range(2 ** (m + 1)):
        if time.time() > deadline - 2:
            break
        
        top_prod = 1 if (pattern >> m) & 1 else -1
        ladder_prods = [1 if (pattern >> i) & 1 else -1 for i in range(m)]
        
        top_signs = [1] * m
        top_signs[m - 1] = top_prod
        
        bot_signs = [ladder_prods[i] for i in range(m - 1)]
        bot_signs.append(ladder_prods[m - 1] * top_prod)
        
        match_signs = [1] * m
        
        if not is_connected_lift(top_signs, bot_signs, match_signs):
            continue
        
        r = spectral_radius(top_signs, bot_signs, match_signs)
        
        if r < best_radius:
            best_radius = r
            best_signs = make_signs_array(top_signs, bot_signs, match_signs).copy()
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
