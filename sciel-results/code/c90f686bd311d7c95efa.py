import time
import random
from itertools import combinations

def construct():
    """
    Construct a dense Q3-free graph on 16 vertices.
    Strategy: Start with a complete graph, then iteratively remove edges to break
    all Q3 subgraphs. Use a greedy heuristic: at each step, find a Q3 and remove
    the edge that appears in the most Q3s (or just one edge from the Q3).
    Also try local search: flip edges (add/remove) to maximize density while staying Q3-free.
    """
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    n = 16
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    # Helper: check if a set of vertices contains a Q3
    # Q3 has 8 vertices and 12 edges. The 8 vertices of Q3 are all even parity or all odd parity in binary?
    # No, Q3 is bipartite. The vertices of Q3 are {0,1,2,3,4,5,6,7} in binary 000..111.
    # The edges are between vertices differing in exactly one bit.
    # To check if a subset of 8 vertices induces a Q3, we need to check if the induced subgraph is isomorphic to Q3.
    # This is expensive. Instead, we can enumerate all embeddings of Q3 in K16.
    
    # Precompute all Q3 embeddings.
    # A Q3 embedding is defined by choosing 8 vertices and a bijection to {0..7} such that edges match.
    # Actually, it's easier to just iterate over all 8-subsets and check if they form a Q3.
    # There are C(16,8) = 12870 subsets. For each, check if induced subgraph is Q3.
    
    def is_q3(vertices):
        """Check if the induced subgraph on 'vertices' is isomorphic to Q3."""
        if len(vertices) != 8:
            return False
        # Map vertices to 0..7
        vmap = {v: i for i, v in enumerate(vertices)}
        # Build adjacency matrix for the induced subgraph
        adj = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                if (min(vertices[i], vertices[j]), max(vertices[i], vertices[j])) in edge_set:
                    adj[i][j] = 1
                    adj[j][i] = 1
        # Q3 has 12 edges
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count != 12:
            return False
        # Q3 is 3-regular
        for i in range(8):
            if sum(adj[i]) != 3:
                return False
        # Q3 is bipartite with parts of size 4
        # Check if graph is bipartite
        color = [-1]*8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if adj[u][v]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        # Count vertices in each part
        part0 = sum(1 for c in color if c == 0)
        part1 = 8 - part0
        if part0 != 4 or part1 != 4:
            return False
        # Additional check: Q3 has girth 4. No triangles.
        for i in range(8):
            for j in range(i+1, 8):
                if adj[i][j]:
                    for k in range(j+1, 8):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    # This is too slow to recompute from scratch every time we change an edge.
    # Let's precompute all 8-subsets and for each, the 12 edges that would form a Q3 if the mapping is correct.
    # Actually, the structure of Q3 is fixed. For any set of 8 vertices, there are at most 8! / (symmetries) ways to map.
    # But we can just check: does there exist a permutation of the 8 vertices such that the induced graph is Q3?
    
    # Alternative approach: Enumerate all possible Q3s by their vertex sets and edge sets.
    # For a given set of 8 vertices, how many Q3 structures can they support?
    # It's complex. Let's use a different strategy.
    
    # Strategy: Simulated Annealing / Local Search on the edge set.
    # Objective: Maximize |E| subject to no Q3.
    # Penalty: -lambda * (number of Q3s present)
    
    # We need a fast way to count Q3s.
    # Let's precompute all 8-subsets.
    all_8_subsets = list(combinations(range(n), 8))
    
    # For each subset, precompute the list of edge sets that would make it a Q3.
    # This is still hard.
    
    # Let's try a different tactic:
    # Start with a known dense Q3-free graph.
    # The seed uses K_{3,13} with 39 edges.
    # Can we do better?
    # Turan graph T(16, 3) has edges: 3 parts of size 5,5,6.
    # Edges = 5*5 + 5*6 + 5*6 = 25 + 30 + 30 = 85.
    # Does T(16,3) contain a Q3?
    # Q3 is bipartite. A bipartite graph can be embedded in a complete bipartite subgraph of T(16,3)?
    # T(16,3) is 3-partite. A Q3 is bipartite, so it must lie within two of the parts?
    # No, a subgraph of a 3-partite graph is not necessarily bipartite in a way that respects the partition.
    # However, Q3 is bipartite. So its vertices can be colored with 2 colors.
    # In T(16,3), edges only exist between different parts.
    # If a Q3 is embedded, its edges must go between different parts of T(16,3).
    # Let the parts be A, B, C.
    # The Q3 vertices are distributed among A, B, C.
    # Since Q3 is bipartite, it has no odd cycles.
    # Actually, it is known that Turan graphs T(n, r) are K_{r+1}-free.
    # Q3 is not a complete graph.
    # Let's check if T(16,3) is Q3-free.
    # If we take T(16,3), can we find a Q3?
    # Let's test T(16,3) quickly.
    
    # Create T(16,3)
    parts = [[i for i in range(0, 6)], [i for i in range(6, 11)], [i for i in range(11, 16)]]
    t_edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    t_edges.add((min(u, v), max(u, v)))
    
    # Check if T(16,3) is Q3-free
    edge_set = t_edges
    q3_count = 0
    for subset in all_8_subsets:
        if is_q3(subset):
            q3_count += 1
            break
    # print(f"T(16,3) Q3 count: {q3_count}")
    
    # If T(16,3) is Q3-free, it has 85 edges. That's a huge jump from 39.
    # Let's assume it is or test it properly in the code.
    
    # Let's implement the check properly and use T(16,3) if it's free.
    # If not, we can try to remove edges from T(16,3) to make it free.
    
    # Let's refine the is_q3 check to be faster by using bitmasks or precomputed adjacency.
    # But for now, let's just use T(16,3) and verify.
    
    # Actually, I recall that Q3 is a subgraph of K_{4,4}.
    # T(16,3) contains K_{5,5}, K_{5,6}, K_{6,5} as subgraphs (between pairs of parts).
    # So it contains K_{4,4}.
    # Does K_{4,4} contain a Q3?
    # Q3 has 8 vertices, 4 in each part.
    # Yes, Q3 is a subgraph of K_{4,4}.
    # So T(16,3) likely contains Q3s.
    
    # So we need to remove edges from T(16,3) to break all Q3s.
    # This is a hitting set problem.
    
    # Let's use a greedy approach:
    # 1. Start with T(16,3).
    # 2. Find a Q3.
    # 3. Remove an edge from it.
    # 4. Repeat until no Q3s remain.
    
    # To speed up, we can cache the Q3s found.
    
    # Let's implement this.
    
    # First, let's create a faster is_q3 check by precomputing the structure of Q3.
    # Q3 vertices: 0..7. Edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    q3_edges_template = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    def check_subset_q3(subset, edge_set):
        """Check if the induced subgraph on 'subset' is isomorphic to Q3."""
        # We need to check if there is a bijection from subset to 0..7 such that edges match.
        # This is an isomorphism problem. For 8 vertices, it's feasible.
        # We can try all 8! permutations? No, 40320 is too many for each subset.
        # We can use backtracking with pruning.
        
        # First, check degree sequence. Q3 is 3-regular.
        degrees = [0]*8
        for i in range(8):
            for j in range(i+1, 8):
                if (min(subset[i], subset[j]), max(subset[i], subset[j])) in edge_set:
                    degrees[i] += 1
                    degrees[j] += 1
        if any(d != 3 for d in degrees):
            return False
        
        # Check bipartiteness and part sizes.
        # Build adjacency list
        adj = [[] for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                if (min(subset[i], subset[j]), max(subset[i], subset[j])) in edge_set:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Bipartite check
        color = [-1]*8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
        
        part0 = [i for i in range(8) if color[i] == 0]
        part1 = [i for i in range(8) if color[i] == 1]
        if len(part0) != 4 or len(part1) != 4:
            return False
        
        # Now, we need to check if the bipartite graph is isomorphic to Q3.
        # Q3 is the 3-cube. It is the unique 3-regular bipartite graph on 8 vertices with girth 4?
        # Actually, there are multiple 3-regular bipartite graphs on 8 vertices.
        # Q3 has girth 4. The other one is K_{3,3} with two extra vertices? No.
        # Let's just check if it contains a 4-cycle and has the right structure.
        # Or simpler: check if the number of 4-cycles is the same as Q3.
        # Q3 has 6 4-cycles (faces of the cube).
        
        # Count 4-cycles
        count_4 = 0
        for i in part0:
            for j in part1:
                if j in adj[i]:
                    # Find common neighbors
                    common = set(adj[i]) & set(adj[j])
                    for k in common:
                        # (i, j, k, ...) no, this counts paths of length 2.
                        pass
        
        # Let's count 4-cycles properly.
        # A 4-cycle is i-j-k-l-i.
        # In a bipartite graph, i,k in part0, j,l in part1.
        # For each pair (i,k) in part0, count common neighbors in part1.
        # Each 4-cycle is counted twice (once for each diagonal pair).
        # So count = sum_{i<k in part0} C(common_neighbors(i,k), 2) * 2?
        # No. For a fixed pair (i,k), if they have m common neighbors, they form m 4-cycles?
        # No, each common neighbor j and l forms a 4-cycle i-j-k-l-i.
        # So for a pair (i,k), the number of 4-cycles is C(m, 2).
        # Total 4-cycles = sum_{i<k in part0} C(m_{ik}, 2).
        
        total_4 = 0
        for idx_i in range(4):
            for idx_k in range(idx_i+1, 4):
                i = part0[idx_i]
                k = part0[idx_k]
                common = set(adj[i]) & set(adj[k])
                m = len(common)
                total_4 += m * (m - 1) // 2
        
        if total_4 != 6:
            return False
            
        return True

    # This is still potentially slow. Let's optimize.
    # Precompute for each 8-subset whether it can be a Q3 based on degree sequence and bipartiteness.
    # Then only check the 4-cycle count for those that pass.
    
    # Let's start with T(16,3) and greedily remove edges.
    
    # Initialize edge set with T(16,3)
    edge_set = set(t_edges)
    
    # Find all Q3s
    def find_q3s(edge_set):
        q3s = []
        for subset in all_8_subsets:
            if check_subset_q3(subset, edge_set):
                q3s.append(subset)
        return q3s
    
    # Greedy removal
    # To speed up, we can just find one Q3 at a time and remove an edge.
    
    # Let's implement a loop that finds a Q3 and removes an edge.
    # We need to be careful with time.
    
    # Let's try to improve the check by using bitmasks for adjacency.
    # But first, let's see if T(16,3) has many Q3s.
    
    # Actually, let's just run the greedy removal and see how many edges we end up with.
    
    # To make it faster, we can precompute the list of edges for each Q3.
    
    # Let's modify the approach:
    # 1. Start with T(16,3).
    # 2. While there exists a Q3:
    #    a. Find a Q3.
    #    b. Remove the edge in the Q3 that is part of the most Q3s (approximate by just removing one).
    #    c. Update edge_set.
    
    # To avoid recomputing all Q3s from scratch, we can just check if the current edge_set has any Q3.
    
    # Let's implement a function that returns one Q3 if found, else None.
    
    def find_one_q3(edge_set):
        for subset in all_8_subsets:
            if check_subset_q3(subset, edge_set):
                # Return the edges of this Q3
                edges = []
                for i in range(8):
                    for j in range(i+1, 8):
                        if (min(subset[i], subset[j]), max(subset[i], subset[j])) in edge_set:
                            edges.append((min(subset[i], subset[j]), max(subset[i], subset[j])))
                return subset, edges
        return None
    
    # Now, the greedy loop
    # We need to track the best graph found.
    best_edges = set(t_edges)
    
    # Let's run the greedy removal
    # To speed up, we can limit the number of iterations or time.
    
    # Let's try to remove edges from T(16,3) to make it Q3-free.
    
    # We'll keep a list of removed edges to potentially add them back later? No, just remove.
    
    # Let's run the loop.
    
    # To make it faster, we can shuffle the subsets to find Q3s more quickly?
    # Or just iterate.
    
    # Let's try it.
    
    # Note: check_subset_q3 is the bottleneck.
    
    # Let's optimize check_subset_q3 by checking the degree sequence first, which is fast.
    
    # Let's rewrite check_subset_q3 to be faster.
    
    def fast_check_q3(subset, edge_set):
        # 1. Check degrees
        degrees = [0]*8
        for i in range(8):
            for j in range(i+1, 8):
                if (min(subset[i], subset[j]), max(subset[i], subset[j])) in edge_set:
                    degrees[i] += 1
                    degrees[j] += 1
        if any(d != 3 for d in degrees):
            return False, None
        
        # 2. Build adjacency
        adj = [[] for _ in range(8)]
        edges_in_subgraph = []
        for i in range(8):
            for j in range(i+1, 8):
                e = (min(subset[i], subset[j]), max(subset[i], subset[j]))
                if e in edge_set:
                    adj[i].append(j)
                    adj[j].append(i)
                    edges_in_subgraph.append(e)
        
        # 3. Bipartite check
        color = [-1]*8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, None
        
        part0 = [i for i in range(8) if color[i] == 0]
        part1 = [i for i in range(8) if color[i] == 1]
        if len(part0) != 4 or len(part1) != 4:
            return False, None
        
        # 4. Count 4-cycles
        total_4 = 0
        for idx_i in range(4):
            for idx_k in range(idx_i+1, 4):
                i = part0[idx_i]
                k = part0[idx_k]
                common = set(adj[i]) & set(adj[k])
                m = len(common)
                total_4 += m * (m - 1) // 2
        
        if total_4 != 6:
            return False, None
            
        return True, edges_in_subgraph

    # Now, the greedy loop
    # We'll use a time limit.
    
    # Let's start with T(16,3)
    current_edges = set(t_edges)
    
    # Greedy removal
    # We'll try to remove edges one by one.
    
    # To make it smarter, we can try to remove the edge that is in the most Q3s.
    # But finding all Q3s is expensive.
    
    # Let's just remove one edge from the first Q3 found.
    
    # Let's run the loop until no Q3s are found or time runs out.
    
    # To speed up, we can cache the result of fast_check_q3 for subsets?
    # But the edge_set changes.
    
    # Let's just run it.
    
    # We'll keep the best graph (most edges) found.
    
    # Since we start with T(16,3) and only remove edges, the best graph is the final one.
    
    # Let's run the greedy removal.
    
    # To avoid infinite loops or slow performance, we'll add a time check.
    
    # Let's try to improve by randomizing the order of subsets.
    
    # Let's shuffle all_8_subsets once.
    random.shuffle(all_8_subsets)
    
    # Now, the loop
    while time.time() < deadline:
        found = False
        for subset in all_8_subsets:
            is_q3, edges = fast_check_q3(subset, current_edges)
            if is_q3:
                # Remove one edge from this Q3
                # To be strategic, remove the edge that appears in the most Q3s?
                # For now, just remove the first edge.
                edge_to_remove = edges[0]
                current_edges.remove(edge_to_remove)
                found = True
                break
        if not found:
            break
    
    # Convert to list of [u, v]
    result = [[u, v] for u, v in current_edges]
    
    # Let's try to add back some edges if possible?
    # No, we started with T(16,3) and removed edges.
    # We can try local search: try to add an edge back and see if it creates a Q3.
    
    # Let's do a local search to add edges back.
    
    # Try to add each missing edge and check if it creates a Q3.
    
    # This might be slow, but let's try.
    
    # First, let's see how many edges we have.
    # print(len(current_edges))
    
    # Let's try to add edges back.
    
    # We'll iterate over all possible edges not in current_edges.
    
    # For each, try to add it and check if it creates a Q3.
    
    # To check if adding an edge creates a Q3, we can just check all Q3s that contain this edge.
    
    # But we don't have a list of Q3s.
    
    # We can just re-run the check for all subsets? That's expensive.
    
    # Let's skip the add-back for now and just return the result of the greedy removal.
    
    return result
