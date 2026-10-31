import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T

    # Conflict if dot > 2 (i.e., distance^2 < 4)
    # dist^2 = ||v||^2 + ||u||^2 - 2*v.u = 4 + 4 - 2*v.u = 8 - 2*v.u
    # We need dist^2 >= 4 => 8 - 2*v.u >= 4 => 4 >= 2*v.u => v.u <= 2
    # So conflict if v.u > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Tomita-style branch-and-bound for Maximum Independent Set
    # We want to find the largest set of vectors with no conflicts (independent set in conflict graph)
    
    # Convert to adjacency lists for faster operations
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    
    best_size = [0]
    best_set = [[]]
    
    # Precompute colors for upper bound (greedy coloring)
    # For branch and bound, we use a greedy coloring of the candidate set to get an upper bound
    
    def color_sort(P, adj):
        """
        Greedily color the induced subgraph on P.
        Returns (coloring, sorted_P) where sorted_P is ordered by color class size.
        coloring[i] = color of vertex P[i]
        """
        n = len(P)
        if n == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        
        color = np.zeros(n, dtype=np.int32)
        # Order by degree in P
        deg_in_P = np.array([np.intersect1d(adj[P[i]], P, assume_unique=False).size for i in range(n)])
        order = np.argsort(deg_in_P)[::-1]  # Highest degree first
        
        # Greedy coloring
        for idx in order:
            v = P[idx]
            # Find colors used by neighbors of v in P that are already colored
            neighbors_in_P = np.intersect1d(adj[v], P, assume_unique=False)
            neighbor_indices = np.where(np.isin(P, neighbors_in_P))[0]
            used_colors = set(color[neighbor_indices].tolist())
            
            c = 0
            while c in used_colors:
                c += 1
            color[idx] = c
        
        # Sort by color
        sorted_indices = np.argsort(color)
        sorted_P = P[sorted_indices]
        sorted_color = color[sorted_indices]
        
        return sorted_color, sorted_P

    def expand(R, P, time_limit):
        """
        Branch and bound expansion.
        R: current independent set (list of vertex indices)
        P: candidate set (list of vertex indices that can be added)
        """
        if time.time() > time_limit:
            return
        
        # Upper bound: |R| + |P|
        if len(R) + len(P) <= best_size[0]:
            return
        
        if len(P) == 0:
            if len(R) > best_size[0]:
                best_size[0] = len(R)
                best_set[0] = list(R)
            return
        
        # Color sort for upper bound
        sorted_color, sorted_P = color_sort(P, adj)
        
        # Process in reverse order (highest color first)
        for i in range(len(sorted_P) - 1, -1, -1):
            if time.time() > time_limit:
                return
            
            v = sorted_P[i]
            
            # Upper bound: |R| + i + 1 (since we have i+1 vertices with colors <= sorted_color[i])
            # Actually, the upper bound is |R| + (number of vertices from i to end)
            # But a tighter bound: |R| + max color in P[i:]
            # Simpler: |R| + (len(sorted_P) - i)
            if len(R) + (len(sorted_P) - i) <= best_size[0]:
                return
            
            # Branch: include v
            new_R = R + [v]
            
            # New P: vertices in P[i:] that are not adjacent to v
            # P[i:] corresponds to sorted_P[i:]
            remaining_P = sorted_P[i+1:]
            # Filter out neighbors of v
            new_P = [u for u in remaining_P if not conflicts[v, u]]
            
            expand(new_R, new_P, time_limit)
            
            # Remove v from P for next iteration
            # Since we're iterating in reverse, we just continue

    # Run Tomita algorithm
    expand([], list(range(N4)), deadline)
    
    # If Tomita didn't find anything (shouldn't happen), fall back to greedy
    if best_size[0] == 0:
        selected = []
        for i in range(N4):
            if all(not conflicts[i, j] for j in selected):
                selected.append(i)
        best_set[0] = selected
    
    result_v4 = vectors_v4[best_set[0]]
    
    # Combine V4 and V1
    # Check V1 compatibility with selected V4
    # V1 dot V4: V1 has one ±2, V4 has four ±1.
    # Dot = ±2 * (V4 component at that position) which is -2, 0, or 2
    # Conflict if dot > 2: only if dot = 3,4,... impossible since max is 2
    # So all V1 are compatible with all V4.
    # V1 among themselves: same position same sign: dot=4 (conflict)
    # Same position opposite sign: dot=-4 (ok)
    # Different positions: dot=0 (ok)
    # So we can take both +2 and -2 for each position: 22 vectors total.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
