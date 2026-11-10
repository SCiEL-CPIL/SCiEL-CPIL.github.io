import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    q = 5
    # Build PG(2, q) points: 31 points
    # Points: (1, x, y) for x,y in GF(q); (0, 1, z) for z in GF(q); (0, 0, 1)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    adj = [set() for _ in range(n)]
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                edge_set.add((i, j))
    
    # Check edge count - should be 90 for q=5 polarity graph
    # The polarity graph of PG(2,q) has q^2+q+1 vertices and q(q+1)/2 + (q+1)/2 edges for odd q
    # Actually let me just check: for q=5, we expect about 90 edges
    
    # Find the best non-edge to add (minimizing new 4-cycles)
    best_edge = None
    best_cost = float('inf')
    
    # Precompute for each vertex its neighbors as a sorted list for faster iteration
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    # For counting cost of adding edge (i,j):
    # New 4-cycles are of the form i-a-b-j-i where:
    #   a ~ i, b ~ j, a ~ b, a != b, a != j, b != i
    # Since (i,j) is not an edge, a!=j is automatic (if a~i and a==j, then j~i, contradiction)
    # Similarly b!=i is automatic.
    # So we need: a in N(i), b in N(j), a ~ b, a != b
    # Each such (a,b) pair gives a unique 4-cycle i-a-b-j-i
    
    # Wait, I need to be more careful. A 4-cycle is a subgraph C4, which is a set of 4 distinct vertices 
    # with edges forming a cycle. So i-a-b-j-i means edges (i,a), (a,b), (b,j), (j,i).
    # Since (j,i) is the edge we're adding, we need (i,a), (a,b), (b,j) to already exist.
    # And all four vertices must be distinct: i, a, b, j all different.
    # a != i (a is neighbor of i, simple graph), a != j (since (i,j) not edge and a~i means a!=j... wait no)
    # Actually a can equal j only if j~i, but (i,j) is not an edge, so a != j automatically.
    # Similarly b != i automatically.
    # a != b is needed.
    
    # So cost = number of pairs (a,b) with a in N(i), b in N(j), a~b, a!=b
    
    # To make this faster, I'll precompute the adjacency as a 2D boolean array or use bitsets
    
    # Let's use a 2D list of booleans for fast lookup
    is_adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            is_adj[i][j] = True
            is_adj[j][i] = True
    
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
    
    # Sort non-edges by a heuristic: prefer those with fewer common neighbors
    # First pass: compute common neighbor count for each non-edge
    edge_costs = []
    for (i, j) in non_edges:
        common = len(adj[i] & adj[j])
        edge_costs.append((common, i, j))
    
    # Sort by common neighbors (fewer common neighbors likely means fewer 4-cycles)
    edge_costs.sort()
    
    # Evaluate candidates in order of increasing common neighbors
    evaluated = 0
    for common, i, j in edge_costs:
        if time.time() > deadline:
            break
        
        # Count exact cost: number of pairs (a,b) with a in N(i), b in N(j), a~b, a!=b
        cost = 0
        for a in adj_lists[i]:
            for b in adj_lists[j]:
                if a == b:
                    continue
                if is_adj[a][b]:
                    cost += 1
        
        if cost < best_cost:
            best_cost = cost
            best_edge = [i, j]
        
        evaluated += 1
    
    if best_edge is not None:
        edges.append(best_edge)
    
    # Ensure we have exactly 91 edges
    if len(edges) != 91:
        existing = set()
        for e in edges:
            existing.add(tuple(sorted(e)))
        for i in range(n):
            for j in range(i+1, n):
                if len(edges) >= 91:
                    break
                if (i,j) not in existing:
                    edges.append([i,j])
                    existing.add((i,j))
            if len(edges) >= 91:
                break
    
    return edges[:91]
