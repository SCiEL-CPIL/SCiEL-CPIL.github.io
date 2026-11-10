import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    
    def get_cycle_lengths(edges):
        """
        Returns the set of lengths of all simple cycles in the graph.
        Uses a DFS-based approach to find all simple cycles.
        To avoid counting isomorphic cycles multiple times, we enforce that
        the start node is the smallest vertex in the cycle.
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        # We iterate through all vertices as potential starts.
        # For a cycle to be counted starting at `start`, `start` must be the 
        # smallest vertex in that cycle.
        # This is achieved by restricting the DFS path to only visit vertices 
        # larger than `start`, except when closing the loop back to `start`.
        
        for start in range(n):
            # DFS from start
            # path[0] is start
            # We want to find paths that return to start.
            
            def dfs(node, path, visited):
                # If we can close the cycle
                if start in adj[node]:
                    # We close the cycle if len(path) >= 4 (triangle-free constraint)
                    # Note: The problem requires girth >= 4, so no triangles.
                    # However, if the graph has triangles, we should probably not count them?
                    # The contract says: "Every simple cycle must have length at least four."
                    # So if a triangle exists, the graph is INVALID.
                    # Our search space is restricted to bipartite graphs, which are triangle-free.
                    # So we don't need to explicitly filter out length 3 if we only search bipartite,
                    # but for a general checker, we would.
                    # Since we are searching bipartite, we just add len(path).
                    if len(path) >= 4:
                        cycle_lengths.add(len(path))
                    
                    # Continue searching? 
                    # If we close the cycle, can we continue?
                    # No, because if we continue, we are extending the path.
                    # But wait, a simple cycle is a closed path with no repeated vertices (except start/end).
                    # Once we decide to close the cycle, that specific sequence of vertices forms a cycle.
                    # Can we form another cycle with the same start and a longer path?
                    # Yes, e.g., start->a->b->start is one, start->a->c->b->start is another.
                    # The DFS naturally explores all simple paths starting at `start` using vertices > start.
                    # When we reach a neighbor that is `start`, it forms a cycle.
                    # We do NOT break, because there might be other neighbors to explore that lead to longer cycles.
                    # BUT, we must not include `start` in the `visited` set again or proceed to it.
                    # The loop below handles this: `if neighbor == start` adds the length.
                    # The `elif` ensures we only recurse on neighbors > start.
                
                for neighbor in adj[node]:
                    if neighbor == start:
                        # Cycle found
                        if len(path) >= 4:
                            cycle_lengths.add(len(path))
                    elif neighbor > start and neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.discard(neighbor)
            
            dfs(start, [start], {start})
        
        return cycle_lengths

    def is_valid_bipartite_cubic(adj_matrix):
        """
        adj_matrix is a list of 6 sets, where adj_matrix[i] contains the B-side neighbors (0-5 index, representing 6-11)
        for A-side vertex i (0-5).
        """
        # 1. Check degree 3 on A side
        for i in range(6):
            if len(adj_matrix[i]) != 3:
                return False
        
        # 2. Check degree 3 on B side
        deg_b = [0] * 6
        for i in range(6):
            for j in adj_matrix[i]:
                deg_b[j] += 1
        for d in deg_b:
            if d != 3:
                return False
                
        # 3. Triangle-free is guaranteed for bipartite.
        
        # 4. Has a cycle.
        # A 3-regular bipartite graph on 12 vertices.
        # If it's a tree, it would have 11 edges. It has 18 edges. So it has cycles.
        
        return True

    def adj_matrix_to_edges(adj_matrix):
        edges = []
        for i in range(6):
            for j in adj_matrix[i]:
                edges.append([i, 6 + j])
        return edges

    def generate_random_3reg_bipartite():
        """
        Generates a random 3-regular bipartite graph.
        We represent it by a 6x6 0-1 matrix with row/col sum 3.
        Method: Start with identity (M1). Then find two disjoint perfect matchings M2, M3.
        Actually, it's easier to just construct the matrix directly.
        
        Strategy:
        1. Initialize a 6x6 matrix with 0s.
        2. For each row i, pick 3 distinct columns.
        3. Ensure column sums are 3.
        
        This is a constraint satisfaction problem.
        Simple approach: Randomly shuffle columns for 3 disjoint perfect matchings.
        Let P1, P2, P3 be random permutations of 0..5.
        Set M[i][P1[i]] = 1
        Set M[i][P2[i]] = 1
        Set M[i][P3[i]] = 1
        
        For this to be a valid simple graph, P1[i], P2[i], P3[i] must be distinct for each i.
        And for each column j, it must appear exactly once in P1, once in P2, and once in P3.
        Wait, if P1, P2, P3 are permutations, then column j appears exactly once in each permutation.
        So column sum is always 3.
        The only constraint is that for each row i, P1[i] != P2[i] != P3[i].
        
        So we need to find 3 permutations P1, P2, P3 such that they are "orthogonal" in the sense that
        P1[i], P2[i], P3[i] are distinct for all i.
        
        We can fix P1 = Identity [0, 1, 2, 3, 4, 5].
        Then we need P2, P3 such that P2[i] != i, P3[i] != i, P2[i] != P3[i].
        """
        
        # Fix P1
        P1 = list(range(6))
        
        # Generate random P2 (derangement of 6 elements)
        # Then generate random P3 (permutation of 6 elements) such that P3[i] != i and P3[i] != P2[i].
        
        while True:
            # Generate random derangement P2
            # Simple rejection sampling for small n
            while True:
                P2 = list(range(6))
                random.shuffle(P2)
                if all(P2[i] != i for i in range(6)):
                    break
            
            # Generate random P3
            # P3 must be a permutation, P3[i] != i, P3[i] != P2[i]
            # This is a permanent of a specific matrix.
            # For n=6, we can just try random permutations.
            
            while True:
                P3 = list(range(6))
                random.shuffle(P3)
                if all(P3[i] != i and P3[i] != P2[i] for i in range(6)):
                    break
            
            # Construct adjacency matrix
            adj_matrix = [set() for _ in range(6)]
            for i in range(6):
                adj_matrix[i].add(P1[i])
                adj_matrix[i].add(P2[i])
                adj_matrix[i].add(P3[i])
            
            return adj_matrix

    # Search Strategy:
    # 1. Exhaustive search over all pairs of permutations P2, P3 (given P1=Id).
    #    Number of derangements of 6 is 265.
    #    Number of valid P3 for a given P2 is also small.
    #    Total combinations are manageable.
    
    # Actually, let's just iterate over all valid (P2, P3) pairs.
    # P2 must be a derangement.
    # P3 must be a permutation disjoint from Id and P2 pointwise.
    
    derangements = []
    for perm in itertools.permutations(range(6)):
        if all(perm[i] != i for i in range(6)):
            derangements.append(perm)
    
    # Precompute valid P3s for each P2? Or just on the fly.
    # Let's iterate.
    
    best_edges = None
    best_score = 100
    
    # We will also include the known K3,3 + K3,3 as a fallback
    # K3,3 edges:
    # 0-6, 0-7, 0-8
    # 1-6, 1-7, 1-8
    # 2-6, 2-7, 2-8
    # 3-9, 3-10, 3-11
    # 4-9, 4-10, 4-11
    # 5-9, 5-10, 5-11
    fallback_edges = []
    for i in range(3):
        for j in range(3):
            fallback_edges.append([i, 6+j])
            fallback_edges.append([i+3, 9+j])
    
    # Calculate score for fallback
    cl_fb = get_cycle_lengths(fallback_edges)
    score_fb = len(cl_fb)
    best_score = score_fb
    best_edges = fallback_edges
    
    # Iterate over derangements
    for P2 in derangements:
        if time.time() > deadline:
            break
            
        # Find valid P3s
        # P3[i] != i and P3[i] != P2[i]
        # This is equivalent to finding a perfect matching in a bipartite graph defined by available slots.
        # Or just iterate permutations (720 is small).
        
        for P3 in itertools.permutations(range(6)):
            valid_p3 = True
            for i in range(6):
                if P3[i] == i or P3[i] == P2[i]:
                    valid_p3 = False
                    break
            if not valid_p3:
                continue
                
            # Construct graph
            adj_matrix = [set() for _ in range(6)]
            for i in range(6):
                adj_matrix[i].add(i)      # P1
                adj_matrix[i].add(P2[i])  # P2
                adj_matrix[i].add(P3[i])  # P3
            
            # This graph is always 3-regular bipartite.
            # We don't need to check is_valid_bipartite_cubic as it's structurally guaranteed.
            
            edges = adj_matrix_to_edges(adj_matrix)
            
            # Get cycle lengths
            cl = get_cycle_lengths(edges)
            
            if not cl:
                continue # Should not happen
            
            score = len(cl)
            
            if score < best_score:
                best_score = score
                best_edges = edges
                
                if score == 1:
                    # Found the optimal!
                    # Ensure types are correct
                    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
                    return result
                    
            if time.time() > deadline:
                break
                
    # Return the best found
    # Ensure types are correct: list of lists of ints
    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
    
    return result
# EVOLVE-BLOCK-END
