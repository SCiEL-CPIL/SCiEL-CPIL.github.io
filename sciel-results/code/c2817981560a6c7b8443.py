"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    
    def compute_diameter(adj, n):
        max_dist = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                return -1
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    def check_valid(adj, n):
        diam = compute_diameter(adj, n)
        if diam != 2:
            return False
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    adj2 = [list(neigh) for neigh in adj]
                    adj2[u].remove(v)
                    adj2[v].remove(u)
                    essential = False
                    for s in range(n):
                        dist = [-1] * n
                        dist[s] = 0
                        queue = deque([s])
                        while queue:
                            node = queue.popleft()
                            for w in adj2[node]:
                                if dist[w] == -1:
                                    dist[w] = dist[node] + 1
                                    queue.append(w)
                        if -1 in dist or max(dist) > 2:
                            essential = True
                            break
                    if not essential:
                        return False
        return True
    
    def count_edges(adj, n):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Construct the Clebsch graph explicitly
    # Clebsch graph: vertices are binary strings of length 4 with even parity
    # Two vertices are adjacent if they differ in exactly 2 positions
    clebsch_adj = [[] for _ in range(n)]
    # Map even-parity 4-bit strings to 0..15
    vertices = []
    for i in range(16):
        bits = [(i >> k) & 1 for k in range(4)]
        if sum(bits) % 2 == 0:
            vertices.append(i)
    
    # Actually, let me just construct it directly
    # The Clebsch graph can be constructed as follows:
    # Vertices: all subsets of {1,2,3,4} of even size
    # Adjacent if symmetric difference has size 2
    
    # Even subsets of {1,2,3,4}:
    # Size 0: {} -> 0
    # Size 2: {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4} -> 1,2,3,4,5,6
    # Size 4: {1,2,3,4} -> 7
    # That's only 8 vertices. I need 16.
    
    # Let me use a different construction. The Clebsch graph is the folded 5-cube.
    # Or: it's the complement of the 4x4 lattice graph... 
    # 
    # Actually, the Clebsch graph has 16 vertices and is (16,5,0,2) strongly regular.
    # Let me just use a known adjacency list.
    
    # Known Clebsch graph adjacency (one representation):
    # Vertices 0-15
    clebsch_edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),
        (1,6),(1,7),(1,8),(1,9),(1,10),
        (2,6),(2,7),(2,11),(2,12),(2,13),
        (3,6),(3,8),(3,11),(3,14),(3,15),
        (4,7),(4,9),(4,12),(4,14),(4,15),
        (5,7),(5,8),(5,9),(5,10),(5,11),
        (6,10),(6,13),(6,14),(6,15),
        (7,12),(7,13),(7,15),
        (8,12),(8,13),(8,14),
        (9,10),(9,13),(9,14),
        (10,11),(10,12),
        (11,15),
        (12,14),
        (13,14),
        (14,15),
    ]
    
    # Hmm, I'm not sure this is correct. Let me just use the search.
    
    best_edges = 0
    best_adj = None
    
    # Try to find good triangle-free diameter-2 graphs
    for restart in range(3000):
        if time.time() > deadline:
            break
        
        adj = [[] for _ in range(n)]
        prob = 0.2 + 0.15 * random.random()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < prob:
                    creates_tri = False
                    for k in adj[i]:
                        if k in adj[j]:
                            creates_tri = True
                            break
                    if not creates_tri:
                        adj[i].append(j)
                        adj[j].append(i)
        
        # Greedy add edges
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                if j not in adj[i]:
                    creates_tri = False
                    for k in adj[i]:
                        if k in adj[j]:
                            creates_tri = True
                            break
                    if creates_tri:
                        continue
                    adj[i].append(j)
                    adj[j].append(i)
                    if check_valid(adj, n):
                        ec = count_edges(adj, n)
                        if ec > best_edges:
                            best_edges = ec
                            best_adj = [list(neigh) for neigh in adj]
                    else:
                        adj[i].remove(j)
                        adj[j].remove(i)
    
    if best_adj is not None:
        edges = []
        for u in range(n):
            for v in best_adj[u]:
                if u < v:
                    edges.append([int(u), int(v)])
        return edges
    
    return [[0, int(v)] for v in range(1, n)]

# EVOLVE-BLOCK-END
