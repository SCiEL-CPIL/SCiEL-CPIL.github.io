import time
import random

def is_k4_free(edges, n):
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Check for K4
    # A K4 is a set of 4 vertices all connected to each other.
    # We can iterate over all 4-combinations. n=18, C(18,4) = 3060. Very fast.
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]: continue
            for k in range(j + 1, n):
                if not adj[i][k] or not adj[j][k]: continue
                for l in range(k + 1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return False
    return True

def get_paley_edges(q):
    # q must be a prime power q = 1 mod 4
    # Returns edges for vertices 0..q-1
    # Determine quadratic residues
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    edges = []
    for u in range(q):
        for v in range(u + 1, q):
            diff = (v - u) % q
            if diff in residues:
                edges.append([u, v])
    return edges

def construct():
    # Candidate 1: Paley graph P(17) + isolated vertex
    # P(17) is on vertices 0..16. Vertex 17 is isolated.
    edges_17 = get_paley_edges(17)
    # Check K4-free
    if is_k4_free(edges_17, 18):
        return edges_17

    # Candidate 2: Two disjoint Paley graphs P(9)
    # P(9) on vertices 0..8 and 9..17
    edges_9_1 = get_paley_edges(9)
    edges_9_2 = [[u + 9, v + 9] for u, v in edges_9_1]
    edges_18 = edges_9_1 + edges_9_2
    if is_k4_free(edges_18, 18):
        return edges_18

    # Fallback: Baseline K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
