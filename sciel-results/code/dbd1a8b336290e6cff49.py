import itertools
import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 16
    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges_total = len(all_edges)
    
    # Q3 structure: 8 vertices, 12 edges
    # Vertices 0..7 (binary 000..111)
    # Edges connect vertices at Hamming distance 1
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    
    # Precompute all Q3 embeddings in K16
    # An embedding is a mapping of Q3 vertices {0..7} to distinct vertices in {0..15}
    # We need to check if the image of q3_edges forms a subgraph in our candidate graph.
    # To speed up, we can precompute the list of edge-sets for each embedding?
    # No, that's too much memory.
    # Instead, we can generate embeddings on the fly or use a smarter check.
    
    # A Q3 subgraph exists if there is a set of 8 vertices V' and a bijection f: Q3 -> V'
    # such that for every (u,v) in Q3_edges, (f(u), f(v)) is in G.
    
    # Strategy:
    # 1. Start with a good initial graph (e.g., K_3,13 + K_3(A) which has 42 edges, or better).
    # 2. Use local search (edge addition/removal) to maximize edges while maintaining Q3-free property.
    
    # Efficient Q3 check:
    # Instead of checking all embeddings, we can look for specific structures.
    # Q3 is bipartite with parts of size 4.
    # Let's precompute all 8-vertex subsets? C(16,8) = 12870.
    # For each subset, check if it contains a Q3.
    # A Q3 in a specific 8-vertex set S is determined by the adjacency within S.
    # Actually, for a fixed set of 8 vertices, the number of Q3 subgraphs is fixed if we consider labeled vertices?
    # No, we need to check if the induced subgraph on S contains a Q3 as a subgraph (not necessarily induced).
    # Since we are dealing with simple graphs, "contains Q3 as a subgraph" means there is a bijection.
    
    # Let's precompute the list of all possible Q3 edge sets on 8 specific vertices?
    # No, the vertices are abstract.
    
    # Better approach for checking Q3-free:
    # Iterate over all 8-vertex subsets S.
    # For each S, check if the subgraph G[S] contains a Q3.
    # How to check if G[S] contains a Q3?
    # G[S] has 8 vertices. We can try to find a Q3 in it.
    # A Q3 is 3-regular on 8 vertices.
    # We can check if G[S] has a 3-regular spanning subgraph that is isomorphic to Q3.
    # Or simpler: iterate over all permutations of the 8 vertices? 8! = 40320.
    # For each permutation, check if the edges match Q3 edges.
    # This is 12870 * 40320 * 12 checks ~ 6 billion. Too slow.
    
    # Optimization:
    # Q3 is bipartite. The two parts are determined by parity of binary representation.
    # In any Q3 embedding, the 4 vertices in part 0 and 4 in part 1 must map to independent sets in G?
    # No, they must map to sets where there are no edges between vertices in the same part?
    # Yes, Q3 is bipartite, so each part is an independent set in Q3.
    # Therefore, if G contains a Q3, there must be two disjoint independent sets I1, I2 of size 4
    # such that the bipartite graph between I1 and I2 contains a perfect matching? No, it contains a 3-regular bipartite graph (which is a Q3).
    # Specifically, the subgraph induced by I1 U I2 must contain a 3-regular bipartite graph.
    
    # Let's try a different angle: Local Search with a fast Q3 detector.
    # Fast detector:
    # Precompute for every pair of vertices (u,v) the list of common neighbors?
    # Or, precompute for every 4-tuple of vertices if they form an independent set?
    
    # Let's use the following Q3 check:
    # A graph contains a Q3 iff there exist two disjoint independent sets A, B of size 4
    # such that the bipartite graph between A and B is 3-regular.
    # 
    # Step 1: Find all independent sets of size 4.
    # Step 2: For each pair of disjoint independent sets (A, B), check if the bipartite graph G[A,B] is 3-regular.
    
    # How many independent sets of size 4 in a graph with ~45 edges?
    # If the graph is dense, independent sets are rare.
    # If the graph is sparse, there are many.
    # In our case, we start with ~42 edges, so it's moderately dense.
    # C(16,4) = 1820.
    # Number of independent sets of size 4 could be small.
    
    # Let's implement this check.
    
    def is_independent_set(vertices, adj_matrix):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if adj_matrix[vertices[i]][vertices[j]]:
                    return False
        return True

    def find_q3(adj_matrix):
        # Find all independent sets of size 4
        indep_sets = []
        for combo in itertools.combinations(range(16), 4):
            if is_independent_set(combo, adj_matrix):
                indep_sets.append(combo)
        
        # For each pair of disjoint independent sets, check if bipartite graph is 3-regular
        # To avoid duplicates, we can just check all pairs.
        # Number of indep sets might be manageable.
        # If there are too many, we might need to optimize.
        
        # Optimization: Precompute adjacency lists for fast degree checks?
        # For a pair (A, B), we need to check if every vertex in A has degree 3 in G[A,B]
        # and every vertex in B has degree 3 in G[A,B].
        
        for i in range(len(indep_sets)):
            A = indep_sets[i]
            for j in range(i+1, len(indep_sets)):
                B = indep_sets[j]
                # Check disjointness
                if len(set(A) | set(B)) < 8:
                    continue
                
                # Check if bipartite graph between A and B is 3-regular
                valid = True
                # Check degrees for A
                for u in A:
                    deg = 0
                    for v in B:
                        if adj_matrix[u][v]:
                            deg += 1
                    if deg != 3:
                        valid = False
                        break
                if not valid:
                    continue
                # Check degrees for B
                for v in B:
                    deg = 0
                    for u in A:
                        if adj_matrix[u][v]:
                            deg += 1
                    if deg != 3:
                        valid = False
                        break
                if valid:
                    return True
        return False

    # Initial Graph: K_3,13 + K_3(A)
    # A = {0, 1, 2}, B = {3..15}
    # Edges: All A-B, All A-A.
    # This graph is Q3-free?
    # Any Q3 must have 4 vertices in A and 4 in B?
    # If a Q3 has vertices in A, since A is a clique, no two vertices in A can be in the same part of the Q3 bipartition?
    # Wait, Q3 is bipartite. If A is a clique, then any independent set in the Q3 can contain at most 1 vertex from A.
    # Since Q3 has parts of size 4, each part must contain at most 1 vertex from A.
    # So total vertices from A in a Q3 is at most 2.
    # But Q3 has 8 vertices.
    # This doesn't immediately prove Q3-free, but it restricts the structure.
    # Actually, if A is a clique, any Q3 subgraph can use at most 2 vertices from A (one in each bipartition part).
    # The remaining 6 vertices must be from B.
    # B is an independent set in the initial graph? No, B has no edges initially.
    # So if we start with K_3,13 + K_3(A), B is an independent set.
    # Any Q3 must have 4 vertices in one part and 4 in the other.
    # If a part contains vertices from B, they are independent, which is fine.
    # But if a part contains 2 vertices from A, they are adjacent in G, so they cannot be in the same part of the Q3.
    # So each part of the Q3 can have at most 1 vertex from A.
    # So the Q3 uses at most 2 vertices from A and at least 6 from B.
    # Let the Q3 parts be X and Y.
    # |X ∩ A| <= 1, |Y ∩ A| <= 1.
    # |X ∩ B| >= 3, |Y ∩ B| >= 3.
    # In the initial graph, there are NO edges within B.
    # But Q3 is bipartite, so there are no edges within X or within Y.
    # This is consistent.
    # However, we need edges between X and Y.
    # Edges between X∩B and Y∩B? None in initial graph.
    # Edges between X∩A and Y∩B? Yes, all A-B edges exist.
    # Edges between X∩B and Y∩A? Yes.
    # Edges between X∩A and Y∩A? No, because X and Y are independent sets, so no edges between vertices in the same set?
    # Wait, X and Y are the two parts. There are NO edges within X and NO edges within Y.
    # So if x1, x2 in X ∩ A, they are not adjacent in Q3. But they ARE adjacent in G (clique A).
    # This is allowed for a subgraph (non-induced).
    # But we need the Q3 edges to be present in G.
    # The Q3 edges are between X and Y.
    # So we need edges between X∩A and Y∩B, X∩B and Y∩A, and X∩B and Y∩B?
    # No, Q3 is 3-regular.
    # Let's check if K_3,13 + K_3(A) is Q3-free.
    # It is known that this construction is Q3-free.
    
    # Let's start with this graph and try to add edges in B.
    
    # Build adjacency matrix
    adj_matrix = [[False] * n for _ in range(n)]
    current_edges = set()
    
    # Add A-B edges
    for a in range(3):
        for b in range(3, 16):
            adj_matrix[a][b] = True
            adj_matrix[b][a] = True
            current_edges.add((min(a,b), max(a,b)))
            
    # Add A-A edges
    for i in range(3):
        for j in range(i+1, 3):
            adj_matrix[i][j] = True
            adj_matrix[j][i] = True
            current_edges.add((i, j))
            
    # Now try to add edges within B
    # Randomly shuffle B-B edges and try to add them
    b_vertices = list(range(3, 16))
    b_edges = [(i, j) for i in range(len(b_vertices)) for j in range(i+1, len(b_vertices))]
    b_edges = [(b_vertices[i], b_vertices[j]) for i, j in b_edges]
    random.shuffle(b_edges)
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search: try to add edges, if it creates a Q3, remove it (or try another)
    # We can use a simple greedy approach: iterate through potential edges, try to add, if valid, keep.
    # If invalid, skip.
    # Then repeat with different orders?
    
    # To be more effective, we can use a hill-climbing approach with restarts.
    
    def try_add_edge(adj, edge_set, u, v):
        if u == v:
            return False
        if (min(u,v), max(u,v)) in edge_set:
            return True
        # Temporarily add
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u,v), max(u,v)))
        
        if not find_q3(adj):
            return True
        else:
            # Revert
            adj[u][v] = False
            adj[v][u] = False
            edge_set.remove((min(u,v), max(u,v)))
            return False

    # Greedy addition
    for u, v in b_edges:
        if time.time() > deadline - 5:
            break
        try_add_edge(adj_matrix, current_edges, u, v)
        
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Now try to remove edges from A-B or A-A to allow more B-B edges?
    # Or just do random restarts with different initial B-B edge orders.
    
    # Let's do a few restarts
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Reset to initial state
        adj_matrix = [[False] * n for _ in range(n)]
        current_edges = set()
        
        for a in range(3):
            for b in range(3, 16):
                adj_matrix[a][b] = True
                adj_matrix[b][a] = True
                current_edges.add((min(a,b), max(a,b)))
                
        for i in range(3):
            for j in range(i+1, 3):
                adj_matrix[i][j] = True
                adj_matrix[j][i] = True
                current_edges.add((i, j))
                
        random.shuffle(b_edges)
        
        for u, v in b_edges:
            if time.time() > deadline - 5:
                break
            try_add_edge(adj_matrix, current_edges, u, v)
            
        if len(current_edges) > best_count:
            best_edges = set(current_edges)
            best_count = len(current_edges)
            # Update best adj matrix? We only need the edge list at the end.
            # But for further improvements, we might want to keep the adj matrix.
            # For now, just keep the count and edges.
            
    # Convert best_edges to list of lists
    result = [[u, v] for u, v in best_edges]
    return result

# Test locally
if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
