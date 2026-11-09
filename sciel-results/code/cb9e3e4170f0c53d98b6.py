import time
import random

def _has_c5(adj_matrix, n):
    """
    Check if graph contains a 5-cycle.
    adj_matrix is a list of lists of 0/1.
    """
    # For n=33, checking all 5-cycles is O(n^5) which is ~ 39M.
    # Might be too slow in Python if done many times.
    # We can use a heuristic or limit the search.
    # For now, use a simple check.
    for a in range(n):
        for b in range(a + 1, n):
            if not adj_matrix[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj_matrix[b][c]:
                    continue
                for d in range(c + 1, n):
                    if not adj_matrix[c][d]:
                        continue
                    for e in range(d + 1, n):
                        if not adj_matrix[d][e]:
                            continue
                        # Check if e-a is an edge and no other chords?
                        # A 5-cycle is a simple cycle.
                        # We need to ensure it's a simple cycle, i.e., no chords.
                        # But the task says "no 5-cycle (as a subgraph)".
                        # This means we just need to find any 5 vertices that form a cycle of length 5.
                        # It does not need to be induced.
                        # So we just need a, b, c, d, e distinct such that
                        # (a,b), (b,c), (c,d), (d,e), (e,a) are edges.
                        if adj_matrix[e][a]:
                            return True
    return False

def _has_independent_9(adj_matrix, n):
    """
    Check if graph has an independent set of size 9.
    Equivalent to complement graph having a clique of size 9.
    """
    # Greedy check or backtracking.
    # For n=33, finding K9 in complement is hard.
    # We can use a simple heuristic: if max degree in complement is low, maybe no K9.
    # But we need exactness for validity.
    # However, for the search phase, we can use a fast approximation.
    # For the final verification, we must be exact.
    
    # Let's use a simple backtracking for K9 in complement.
    # Complement adjacency: ~adj_matrix (with 0 diagonal)
    comp_adj = [[1 - adj_matrix[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        comp_adj[i][i] = 0

    def find_clique(vertices, size):
        if size == 0:
            return True
        if not vertices:
            return False
        if len(vertices) < size:
            return False
        v = vertices[0]
        # Option 1: include v
        new_vertices = [u for u in vertices[1:] if comp_adj[v][u]]
        if find_clique(new_vertices, size - 1):
            return True
        # Option 2: exclude v
        if find_clique(vertices[1:], size):
            return True
        return False

    # To make it faster, order vertices by degree in complement
    # But for now, just try on full set.
    # This might be slow.
    # We'll use it only for final verification of a candidate.
    return find_clique(list(range(n)), 9)

def _is_valid(n, edges):
    adj_matrix = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
    
    if _has_c5(adj_matrix, n):
        return False
    if _has_independent_9(adj_matrix, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Known valid graph: 8 disjoint K4s, n=32
    n_best = 32
    edges_best = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * b + i
                v = 4 * b + j
                edges_best.append([u, v])
    
    # Try to find a graph with n=33
    # This is a heuristic search
    target_n = 33
    found = False
    
    # Random restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random edges
        # To be C5-free, start sparse or structured
        # Let's start with 8 K4s and one extra vertex, and randomize edges
        edges = list(edges_best)
        # Add vertex 32
        # Connect vertex 32 to some vertices
        # Randomly connect to 4 vertices from different K4s?
        
        # Actually, let's just do a random graph with low density
        # P(edge) = 0.5? Too dense.
        # Let's use a random permutation of the 32-vertex graph and add a vertex
        
        # Simpler: Just check if the 32-vertex graph can be extended.
        # This is complex.
        
        # Given time, I will skip the search for 33 and return 32.
        # The probability of finding a 33-vertex graph in 120s with a naive Python script is low.
        # And if R(C5, K9)=33, it's impossible.
        
        pass
        
    return {'n': n_best, 'edges': edges_best}
