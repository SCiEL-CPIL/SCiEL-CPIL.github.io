import time
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def check_dist_sq(p1, p2, D=1):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + D*dy*dy
        return is_square(d2)

    def check_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        a1 = x1 - x4
        b1 = y1 - y4
        c1 = (x1*x1 + y1*y1) - (x4*x4 + y4*y4)
        
        a2 = x2 - x4
        b2 = y2 - y4
        c2 = (x2*x2 + y2*y2) - (x4*x4 + y4*y4)
        
        a3 = x3 - x4
        b3 = y3 - y4
        c3 = (x3*x3 + y3*y3) - (x4*x4 + y4*y4)
        
        det = a1 * (b2*c3 - c2*b3) - b1 * (a2*c3 - c2*a3) + c1 * (a2*b3 - b2*a3)
        return det == 0

    def is_valid_set(points, D=1):
        n = len(points)
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if not check_dist_sq(points[i], points[j], D):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        for combo in combinations(range(n), 4):
            if check_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
                return False
        return True

    # Known valid set of 5
    current_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    if not is_valid_set(current_set, D):
        current_set = [(0,0), (3,0), (0,4)]
        if not is_valid_set(current_set, D):
            return {'D': 1, 'points': [[[0,1],[0,1]]]}

    best_set = current_set[:]
    best_D = D
    best_size = len(best_set)

    # Search for larger sets using clique finding on integer distance graph
    # We search in a box [-R, R] x [-R, R]
    # To speed up, we only consider points that are at integer distance from (0,0)
    # This ensures (0,0) is a candidate, but we can also search generally.
    # Actually, fixing (0,0) is a good heuristic for finding symmetric or structured sets.
    # But let's try to find any 6-clique.
    
    # Strategy: 
    # 1. Generate candidate points P such that dist((0,0), P) is integer.
    # 2. Check pairwise integer distances among P.
    # 3. Find max clique.
    
    # Let's try R=50, 100, 200
    for R in [50, 100, 200]:
        if time.time() > deadline - 10.0:
            break
            
        candidates = []
        for x in range(-R, R+1):
            for y in range(-R, R+1):
                if x == 0 and y == 0:
                    continue
                if is_square(x*x + y*y):
                    candidates.append((x, y))
        
        # Add (0,0) to candidates
        candidates.append((0,0))
        
        n_cand = len(candidates)
        
        # Build adjacency matrix/list
        # adj[i] is a set of indices j such that dist(i,j) is integer
        adj = [set() for _ in range(n_cand)]
        
        # Precompute squared distances
        # This is O(N^2). For R=200, N ~ 200 * (pi/4) ~ 150 points? 
        # Number of integer distance points in radius R is approx R^2 / pi * 1? No.
        # It's roughly R^2 / (pi * sqrt(R))? No.
        # The number of lattice points (x,y) with x^2+y^2 = k^2 is small.
        # Total candidates in box [-R,R] with integer dist to origin is ~ R * (sum of divisors) ~ R log R.
        # So N is small.
        
        for i in range(n_cand):
            xi, yi = candidates[i]
            for j in range(i+1, n_cand):
                xj, yj = candidates[j]
                dx = xi - xj
                dy = yi - yj
                d2 = dx*dx + dy*dy
                if is_square(d2):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find a clique of size 6
        # Simple backtracking search
        # We want a set of indices C such that all pairs in C are connected.
        
        def find_clique(adj, size):
            # Use a simple recursive backtracker
            # Order nodes by degree to prune?
            # Or just iterate.
            # Since N is small, we can use a simple approach.
            
            # Start with all nodes
            # We can use a library or simple algo.
            # Given N is small (< 500), we can just check all combinations? No, C(500,6) is too big.
            # But the graph is sparse.
            
            # Let's try to find a 6-clique containing (0,0) index (which is last)
            # Or just any 6-clique.
            
            # Simple heuristic: 
            # Try to expand a clique starting from each node.
            
            def expand(clique, candidates):
                if len(clique) == size:
                    return clique
                # Next candidates must be connected to all in clique
                new_cands = []
                for c in candidates:
                    if all(c in adj[n] for n in clique):
                        new_cands.append(c)
                
                if not new_cands:
                    return None
                
                # Try each new candidate
                # Order by degree in new_cands?
                for c in new_cands:
                    res = expand(clique + [c], [x for x in new_cands if x > c])
                    if res:
                        return res
                return None
            
            # To make it faster, we can restrict to nodes connected to (0,0) first
            # Or just run it.
            
            # Let's try starting with (0,0)
            idx_0 = n_cand - 1
            # Nodes connected to 0
            cands_0 = list(adj[idx_0])
            
            # We need 5 more nodes from cands_0 that form a clique with each other and with 0.
            # So we need a 5-clique in the subgraph induced by cands_0.
            
            def find_k_clique_in_subgraph(sub_nodes, k):
                if k == 1:
                    return [sub_nodes[0]]
                if k == 2:
                    for i in range(len(sub_nodes)):
                        for j in range(i+1, len(sub_nodes)):
                            if sub_nodes[j] in adj[sub_nodes[i]]:
                                return [sub_nodes[i], sub_nodes[j]]
                    return None
                
                # Recursive
                for i in range(len(sub_nodes)):
                    node = sub_nodes[i]
                    # Common neighbors in sub_nodes[i+1:]
                    common = []
                    for j in range(i+1, len(sub_nodes)):
                        n2 = sub_nodes[j]
                        if n2 in adj[node]:
                            common.append(n2)
                    
                    if len(common) >= k-1:
                        res = find_k_clique_in_subgraph(common, k-1)
                        if res:
                            return [node] + res
                return None
            
            res_5 = find_k_clique_in_subgraph(cands_0, 5)
            if res_5:
                clique_indices = [idx_0] + res_5
                # Verify it's a clique
                is_clique = True
                for i in range(len(clique_indices)):
                    for j in range(i+1, len(clique_indices)):
                        if clique_indices[j] not in adj[clique_indices[i]]:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    points = [candidates[i] for i in clique_indices]
                    if is_valid_set(points, D):
                        if len(points) > best_size:
                            best_set = points
                            best_size = len(points)
                        return best_set, best_D
            
            # If no 6-clique found with (0,0), try without fixing (0,0)
            # This is harder. Let's just return the best found so far.
            # Or try a generic clique finder on the whole graph.
            
            # Generic clique finder for size 6
            # Use Bron-Kerbosch or simple backtracking.
            # Given time, we'll stick to the (0,0) heuristic which is strong for these problems.
            
    # If we didn't find 6, we return 5
    points_out = [[[x, 1], [y, 1]] for x, y in best_set]
    
    return {'D': best_D, 'points': points_out}
