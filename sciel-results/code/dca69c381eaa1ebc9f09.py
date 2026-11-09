import numpy as np
import time
from collections import deque

def priority(adjacency):
    """
    Calculates priority scores for vertices to maximize the size of the independent set
    selected by a greedy decoder (descending score).
    
    Heuristic: 'Degree-Thresholded Component Size'
    For each vertex v, we consider the subgraph induced by vertices with degree >= degree(v).
    The score is the size of the connected component of v in this subgraph.
    This rewards vertices that are part of large, relatively sparse regions (low degree threshold
    implies a larger subgraph, and being in a large component implies 'uncontested territory').
    Ties are broken by lower degree.
    """
    start_time = time.time()
    # Safety margin: 150s limit. We use 140s.
    deadline = start_time + 140.0
    
    A = np.array(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    degrees = A.sum(axis=1).astype(int)
    
    # Precompute neighbor lists for faster BFS
    # neighbors[i] is a list of indices of neighbors of i
    neighbors = [np.where(A[i])[0] for i in range(n)]
    
    scores = np.zeros(n, dtype=float)
    
    for v in range(n):
        if time.time() > deadline:
            # If we are running out of time, fill remaining scores with a fallback
            # (e.g., simple degree heuristic) to ensure a valid return
            for u in range(v, n):
                scores[u] = -degrees[u]
            return scores
        
        d_v = degrees[v]
        
        # Create mask for nodes with degree >= d_v
        # This defines the "valid" nodes for v's territory exploration
        mask = degrees >= d_v
        
        # BFS from v
        visited = np.zeros(n, dtype=bool)
        queue = deque([v])
        visited[v] = True
        count = 0
        
        while queue:
            u = queue.popleft()
            count += 1
            for w in neighbors[u]:
                if not visited[w] and mask[w]:
                    visited[w] = True
                    queue.append(w)
        
        # Score: Larger component is better.
        # Tie-break: Lower degree is better (consistent with LDF).
        # We scale count to dominate the degree term.
        scores[v] = count * 1000.0 - degrees[v]
        
    return scores
