import time
import random
import sys
import itertools

# EVOLVE-BLOCK-START

def construct():
    """
    Strategy: Construct a graph G on n vertices such that:
    1. G has no K4 (clique number <= 3)
    2. G has no independent set of size 6 (independence number <= 5)
    
    This is equivalent to finding a graph G where:
    - omega(G) <= 3
    - alpha(G) <= 5
    
    We use a backtracking search with constraint propagation.
    We build the graph vertex by vertex. For each new vertex, we choose its
    adjacency to previous vertices such that no K4 is formed in G and no 
    independent set of size 6 is formed in G.
    
    To maximize n, we try to extend a valid graph. We start from a small valid
    graph and try to add vertices. If we get stuck, we backtrack.
    To handle the complexity, we use a heuristic: 
    - Maintain the set of current vertices.
    - When adding a new vertex v, we decide for each existing vertex u whether (u,v) is an edge.
    - Constraint 1 (No K4): If u and v are connected, they cannot share 2 common neighbors 
      that are connected to each other? No, that's not quite right.
      Simpler: For any pair of neighbors of v, say u1, u2, if u1 and u2 are connected, 
      and there is a third vertex w connected to both u1 and u2, then {u1, u2, w, v} would be a K4 
      only if v is connected to u1, u2, and w? No.
      
      Let's rephrase:
      G has no K4.
      G has no independent set of 6.
      
      Let's work with the adjacency matrix or bitsets for efficiency.
      n <= 40 is small enough for bitsets (Python integers).
      
      Approach:
      1. Start with a small valid graph (e.g., n=5, C5).
      2. Try to add a new vertex n.
      3. The new vertex's neighbors must be chosen such that:
         a. No K4 is formed: For any two neighbors u, v of the new vertex, if u and v are 
            connected in the existing graph, they cannot have a common neighbor w that is 
            also connected to the new vertex? 
            Actually, a K4 involving the new vertex v_new would be {v_new, u1, u2, u3} where 
            u1, u2, u3 are pairwise connected in the existing graph, and all are connected to v_new.
            So, the set of neighbors of v_new in the existing graph must NOT contain a triangle.
            i.e., The subgraph induced by the neighbors of v_new in G_existing must be triangle-free.
            
         b. No independent set of size 6 is formed:
            An independent set in G is a clique in the complement G_bar.
            So G_bar has no K6.
            If we add v_new, its non-neighbors in G (i.e., neighbors in G_bar) must form a set 
            that does not contain a K5 in G_bar (which would be an independent set of size 5 in G, 
            plus v_new makes 6).
            So, the set of non-neighbors of v_new in G must NOT contain a K5 in G_bar.
            Equivalently, the set of non-neighbors of v_new in G must NOT contain an independent 
            set of size 5 in G.
            
      This is still complex. Let's use a simpler greedy/backtracking approach with strong pruning.
      
      Alternative: Use the known structure. The Ramsey number R(4,6) is at least 36? No, record is 35.
      We want to find a graph on 35 vertices.
      
      Let's try a different angle: Randomized Local Search on the Edge Set.
      Start with a random graph on n=35.
      Objective: Minimize the number of K4s + number of independent 6-sets.
      If we can reach 0, we have a valid graph.
      
      This is a standard Max-Cut-like or Constraint Satisfaction problem.
      Simulated Annealing or WalkSAT might work.
      
      Let's implement a WalkSAT-style solver.
      1. Initialize G randomly on n=35 (or 36).
      2. Calculate score = (#K4s) + (#Indep6s).
      3. If score == 0, return G.
      4. Pick a random violated constraint (a K4 or an Indep6).
      5. Flip a random edge in that constraint.
      6. Accept if score improves, or with probability p if it worsens (annealing).
      
      This is different from the "Paley extension" strategy. It's a direct search in the space of graphs.
    """
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def count_k4s(adj):
        """Count K4s in graph represented by bitsets adj[i]."""
        count = 0
        n = len(adj)
        for i in range(n):
            # Neighbors of i
            ni = adj[i]
            # For each pair of neighbors j, k
            # If j and k are connected, and they have a common neighbor l > k?
            # This is O(n^4) if done naively.
            # Efficient way:
            # For each i, consider neighbors N(i).
            # For each pair (j, k) in N(i) with j < k:
            #   If j and k are connected (adj[j] & (1 << k)):
            #     Then common neighbors of j and k that are also in N(i) and > k?
            #     Actually, we just need to count triangles in the subgraph induced by N(i).
            #     A K4 containing i corresponds to a triangle in N(i).
            #     So count triangles in G[N(i)].
            
            # To count triangles in a subgraph, we can iterate over edges in the subgraph.
            # But doing this for every i is expensive.
            
            # Alternative: Iterate over all triples (i, j, k) with i < j < k.
            # If i-j, j-k, i-k are edges, it's a triangle.
            # If triangle (i,j,k) exists, and there is a vertex l connected to i,j,k, it's a K4.
            # This is O(n^4). n=35 -> 35^4 ~ 1.5M. This is fast in Python if optimized?
            # 1.5M iterations is fine.
            
            pass # We will implement a faster counter below.
            
        return count

    def count_indep6(adj):
        """Count independent sets of size 6. Equivalent to K6 in complement."""
        # Complement adj_bar[i] = ~adj[i] & mask
        count = 0
        n = len(adj)
        mask = (1 << n) - 1
        adj_bar = [~adj[i] & mask for i in range(n)]
        
        # Count K6s in adj_bar
        # Iterate over 5-tuples? No, too slow.
        # Iterate over triangles in adj_bar?
        # A K6 contains 20 triangles.
        # Counting triangles is O(n^3) or O(n^omega).
        # For n=35, O(n^3) is ~42k. Very fast.
        # But we need K6s, not triangles.
        
        # Let's just use a heuristic or a simpler check for validity.
        # For the SA, we need the score.
        # We can sample? No, we need exact.
        
        # Actually, we can just check validity for the final candidate, 
        # and use a weaker proxy for the SA?
        # Proxy: Sum of degrees in complement? No.
        
        # Let's implement a fast K4 counter and Indep6 counter using bit operations.
        
        # K4 count:
        # For each i:
        #   N = neighbors of i
        #   Count triangles in G[N]
        #   This can be done by iterating j in N, k in N with j<k, check if j-k edge.
        #   If j-k edge, then (i,j,k) is a triangle.
        #   Wait, a K4 is {i,j,k,l}. 
        #   If we fix i, we need triangles in N(i) that are also connected to a 4th vertex?
        #   No, a K4 containing i is formed by i and a triangle in N(i).
        #   So #K4s containing i = #triangles in G[N(i)].
        #   Total #K4s = sum_i (#triangles in G[N(i)]) / 4.
        
        # Triangle count in a subgraph induced by a bitset N:
        # Iterate j in N:
        #   Iterate k in N with k > j:
        #     if adj[j] & (1 << k):
        #        common = adj[j] & adj[k] & N
        #        # We need to count edges in common? No, just count the triangle (i,j,k).
        #        # Wait, the triangle is (i,j,k). i is fixed.
        #        # So if j and k are connected, (i,j,k) is a triangle.
        #        # So we just count pairs (j,k) in N that are connected.
        #        # This is just the number of edges in G[N].
        #        # Wait, no. A triangle is 3 vertices.
        #        # If i is fixed, and j,k are neighbors of i, and j-k is an edge, then {i,j,k} is a triangle.
        #        # So #triangles containing i = #edges in G[N(i)].
        #        # Is that right?
        #        # Yes. The vertices are i, j, k. Edges: (i,j), (i,k), (j,k).
        #        # So for each i, count edges between neighbors of i.
        
        # So #K4s = (1/4) * sum_i (edges in G[N(i)])?
        # No. A K4 {i,j,k,l} contains 4 triangles: (i,j,k), (i,j,l), (i,k,l), (j,k,l).
        # Each triangle is counted once for each of its 4 vertices?
        # No.
        # Let's re-evaluate.
        # A K4 is a set of 4 vertices where all 6 edges exist.
        # My formula: For each vertex i, count the number of edges in the subgraph induced by its neighbors.
        # Let E(N(i)) be the number of edges between neighbors of i.
        # Each K4 {a,b,c,d} contributes:
        #   For i=a: neighbors include b,c,d. Edges among them: (b,c), (c,d), (b,d). 3 edges.
        #   For i=b: neighbors include a,c,d. Edges: (a,c), (a,d), (c,d). 3 edges.
        #   ...
        #   So each K4 contributes 4 * 3 = 12 to the sum?
        #   No, the sum is over i of E(N(i)).
        #   For a specific K4, it contributes 3 to the term for each of its 4 vertices.
        #   Total contribution = 12.
        #   So #K4s = sum_i(E(N(i))) / 12?
        #   Let's check with a single K4.
        #   Vertices 0,1,2,3.
        #   N(0) = {1,2,3}. Edges in N(0): (1,2), (1,3), (2,3). Count = 3.
        #   N(1) = {0,2,3}. Edges: (0,2), (0,3), (2,3). Count = 3.
        #   N(2) = {0,1,3}. Edges: (0,1), (0,3), (1,3). Count = 3.
        #   N(3) = {0,1,2}. Edges: (0,1), (0,2), (1,2). Count = 3.
        #   Sum = 12.
        #   #K4s = 1.
        #   So #K4s = Sum / 12.
        
        # Similarly for Indep6 (K6 in complement).
        # #Indep6s = Sum_i(E_bar(N_bar(i))) / (6 * 5)?
        #   For a K6 in complement, each vertex has 5 neighbors in complement.
        #   Edges among 5 neighbors = 10.
        #   Contribution per vertex = 10.
        #   Total contribution = 6 * 10 = 60.
        #   So #Indep6s = Sum_i(E_bar(N_bar(i))) / 60.
        
        # This is O(n^3) to compute if we do it naively, but with bitsets it's very fast.
        # For each i, we need to count edges in G[N(i)].
        # We can precompute edge counts? Or just iterate.
        
        k4_sum = 0
        n = len(adj)
        for i in range(n):
            ni = adj[i]
            # Count edges in subgraph induced by ni
            # Iterate j in ni:
            #   common = ni & adj[j]
            #   popcount(common) / 2
            #   But we need to be careful with j < k.
            #   Actually, summing popcount(ni & adj[j]) for all j in ni counts each edge twice.
            edge_count = 0
            temp_ni = ni
            while temp_ni:
                lsb = temp_ni & -temp_ni
                j = lsb.bit_length() - 1
                edge_count += (ni & adj[j]).bit_count()
                temp_ni ^= lsb
            k4_sum += edge_count
            
        # Indep6 count
        mask = (1 << n) - 1
        adj_bar = [~adj[i] & mask for i in range(n)]
        indep_sum = 0
        for i in range(n):
            ni_bar = adj_bar[i]
            edge_count = 0
            temp_ni = ni_bar
            while temp_ni:
                lsb = temp_ni & -temp_ni
                j = lsb.bit_length() - 1
                edge_count += (ni_bar & adj_bar[j]).bit_count()
                temp_ni ^= lsb
            indep_sum += edge_count
            
        return k4_sum // 12, indep_sum // 60

    def is_valid(adj):
        k4s, indep6s = count_k4s_and_indep6s(adj)
        return k4s == 0 and indep6s == 0

    def count_k4s_and_indep6s(adj):
        n = len(adj)
        k4_sum = 0
        indep_sum = 0
        mask = (1 << n) - 1
        adj_bar = [~adj[i] & mask for i in range(n)]
        
        for i in range(n):
            ni = adj[i]
            ni_bar = adj_bar[i]
            
            edge_count = 0
            temp = ni
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                edge_count += (ni & adj[j]).bit_count()
                temp ^= lsb
            k4_sum += edge_count
            
            edge_count_bar = 0
            temp = ni_bar
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                edge_count_bar += (ni_bar & adj_bar[j]).bit_count()
                temp ^= lsb
            indep_sum += edge_count_bar
            
        return k4_sum // 12, indep_sum // 60

    # WalkSAT / Simulated Annealing
    best_score = float('inf')
    best_adj = None
    
    # Try different n
    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
            
        n = target_n
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        cur_k4, cur_indep = count_k4s_and_indep6s(adj)
        cur_score = cur_k4 + cur_indep
        best_score_local = cur_score
        best_adj_local = [a for a in adj]
        
        temperature = 1.0
        cooling = 0.995
        
        while time.time() < deadline - 10:
            if cur_score == 0:
                break
                
            # Pick a random violation
            # We don't have the specific violations, just the count.
            # This is a problem. We need to pick an edge to flip.
            # Strategy: Flip a random edge.
            # Or: Estimate where the violations are?
            # Let's just flip a random edge. It's a random walk.
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_k4, new_indep = count_k4s_and_indep6s(adj)
            new_score = new_k4 + new_indep
            delta = new_score - cur_score
            
            if delta < 0 or random.random() < pow(2.718, -delta / temperature):
                cur_score = new_score
                if cur_score < best_score_local:
                    best_score_local = cur_score
                    best_adj_local = [a for a in adj]
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            temperature *= cooling
            if temperature < 0.01:
                break
                
        if best_score_local < best_score:
            best_score = best_score_local
            best_adj = best_adj_local
            if best_score == 0:
                # Found valid graph
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if best_adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    # If no valid graph found, return the best valid one we might have found?
    # The task requires a VALID graph.
    # If we didn't find one, we must return a smaller valid graph.
    # Let's fallback to a known valid graph, e.g., Paley 17.
    
    # Actually, we should keep the best valid graph found during the process?
    # We only track best_score. If best_score > 0, it's invalid.
    # We need to track the best VALID graph.
    
    # Let's re-run with a focus on validity.
    # If the SA doesn't find 0, we return a small valid graph.
    
    # To be safe, let's use the Paley graph if we fail.
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

# EVOLVE-BLOCK-END
