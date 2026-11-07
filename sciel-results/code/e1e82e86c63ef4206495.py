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
    
    def is_connected_lift_fast(top_signs, bot_signs, match_signs):
        """Fast connectivity check using cycle products."""
        # The 2-lift is connected iff the product of signs around every 
        # independent cycle is consistent with a single component.
        # For a prism graph, we can check via BFS on the lift directly,
        # but optimized: use bitmasks for small m.
        lift_n = 2 * n
        # Build adjacency as bitmasks for speed
        adj_bits = [0] * lift_n
        for i in range(m):
            j = (i + 1) % m
            s = top_signs[i]
            if s == 1:
                adj_bits[i*2] |= (1 << (j*2)) | (1 << (i*2+1 - (j*2+1) + j*2+1))  # placeholder
                # Let's just do BFS but with list-based adj for clarity
                pass
        # Fall back to BFS but with pre-built adjacency
        visited = [False] * lift_n
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
        return all(visited)
    
    def make_signs_array(top_signs, bot_signs, match_signs):
        signs = np.zeros(num_edges, dtype=int)
        for i in range(m):
            signs[i] = top_signs[i]
        for i in range(m):
            signs[m + i] = bot_signs[i]
        for i in range(m):
            signs[2*m + i] = match_signs[i]
        return signs
    
    def local_search_polish(best_top, best_bot, best_match, best_r):
        """Try flipping individual signs to improve spectral radius."""
        improved = True
        cur_top = list(best_top)
        cur_bot = list(best_bot)
        cur_match = list(best_match)
        cur_r = best_r
        
        while improved and time.time() < deadline - 5:
            improved = False
            # Try flipping each top sign
            for i in range(m):
                old = cur_top[i]
                cur_top[i] = -old
                if is_connected_lift_fast(cur_top, cur_bot, cur_match):
                    r = spectral_radius(cur_top, cur_bot, cur_match)
                    if r < cur_r - 1e-12:
                        cur_r = r
                        improved = True
                    else:
                        cur_top[i] = old
            # Try flipping each bot sign
            for i in range(m):
                old = cur_bot[i]
                cur_bot[i] = -old
                if is_connected_lift_fast(cur_top, cur_bot, cur_match):
                    r = spectral_radius(cur_top, cur_bot, cur_match)
                    if r < cur_r - 1e-12:
                        cur_r = r
                        improved = True
                    else:
                        cur_bot[i] = old
            # Try flipping each match sign
            for i in range(m):
                old = cur_match[i]
                cur_match[i] = -old
                if is_connected_lift_fast(cur_top, cur_bot, cur_match):
                    r = spectral_radius(cur_top, cur_bot, cur_match)
                    if r < cur_r - 1e-12:
                        cur_r = r
                        improved = True
                    else:
                        cur_match[i] = old
        
        return cur_top, cur_bot, cur_match, cur_r
    
    best_signs = None
    best_radius = float('inf')
    best_top = None
    best_bot = None
    best_match = None
    
    # Enumerate all 2^(m+1) switching classes
    # Gauge: matching = +1, top[0..m-2] = +1
    # Free: top[m-1] (top_prod), ladder_prods[0..m-1]
    for pattern in range(2 ** (m + 1)):
        if time.time() > deadline - 10:
            break
        
        top_prod = 1 if (pattern >> m) & 1 else -1
        ladder_prods = [1 if (pattern >> i) & 1 else -1 for i in range(m)]
        
        top_signs = [1] * m
        top_signs[m - 1] = top_prod
        
        bot_signs = [ladder_prods[i] for i in range(m - 1)]
        bot_signs.append(ladder_prods[m - 1] * top_prod)
        
        match_signs = [1] * m
        
        if not is_connected_lift_fast(top_signs, bot_signs, match_signs):
            continue
        
        r = spectral_radius(top_signs, bot_signs, match_signs)
        
        if r < best_radius:
            best_radius = r
            best_top = top_signs[:]
            best_bot = bot_signs[:]
            best_match = match_signs[:]
    
    if best_top is not None:
        # Local search polish
        best_top, best_bot, best_match, best_radius = local_search_polish(
            best_top, best_bot, best_match, best_radius
        )
        best_signs = make_signs_array(best_top, best_bot, best_match)
    else:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
