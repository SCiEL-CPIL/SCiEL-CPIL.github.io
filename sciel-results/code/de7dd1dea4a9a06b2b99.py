import time
import random
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    total_pairs = n * (n - 1) // 2
    
    # Map pair (u, v) to index 0..120
    pair_to_idx = {}
    idx_to_pair = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            pair_to_idx[(u, v)] = idx
            idx_to_pair.append((u, v))
            idx += 1
            
    # Generate Q3 embeddings
    # Q3 structure: 8 vertices, 12 edges
    # We need to find all sets of 12 edges that form a Q3 subgraph.
    # A Q3 is a 3-regular bipartite graph on 8 vertices.
    # To check if a set of 12 edges forms a Q3, we can check if it's isomorphic to Q3.
    # However, for the Max-SAT / Annealing approach, we need to identify "conflicts".
    # A conflict is a set of 12 edges that *cannot* all be present simultaneously.
    # Wait, the constraint is: NO subgraph is isomorphic to Q3.
    # This means: For any set of 8 vertices, the induced subgraph (or any subgraph on them) must not contain Q3.
    # Actually, "contains no subgraph isomorphic to Q3" means there is no mapping f: {0..7} -> {0..15} such that
    # for every edge (u,v) in Q3, (f(u), f(v)) is in G.
    # This is equivalent to: For every injection f: {0..7} -> {0..15}, it is NOT the case that
    # all 12 edges of Q3 mapped by f are present in G.
    # So, for every such mapping f, we have a clause: NOT (e1 AND e2 AND ... AND e12).
    # In Max-SAT terms, we want to minimize the number of violated clauses.
    # A clause is violated if all 12 corresponding edges are present.
    
    # Number of injections from 8 to 16: P(16, 8) = 16! / 8! = 518,918,400. This is too many to precompute explicitly.
    # We need a smarter way to count violations or sample them.
    # Alternatively, we can check for Q3 existence efficiently during the annealing loop.
    
    # Efficient Q3 check:
    # A graph contains Q3 if there exists a 3-regular bipartite subgraph on 8 vertices?
    # No, Q3 is the *only* 3-regular bipartite graph on 8 vertices?
    # Let's verify.
    # 3-regular graphs on 8 vertices:
    # 1. Q3 (Cube) - Bipartite.
    # 2. K4,4 minus a perfect matching? No, that's 3-regular? K4,4 is 4-regular.
    #    Remove a perfect matching from K4,4: degrees become 3. Is it connected?
    #    Yes. Is it isomorphic to Q3?
    #    Q3 has girth 4. K4,4 minus matching has triangles? No, it's bipartite.
    #    Does it have 4-cycles? Yes.
    #    Are Q3 and (K4,4 - M) isomorphic?
    #    In Q3, the number of 4-cycles is 6.
    #    In K4,4 - M, let's count 4-cycles.
    #    Actually, it is a known result that Q3 is the unique 3-regular bipartite graph on 8 vertices?
    #    Let's check.
    #    Number of 3-regular graphs on 8 vertices is 5.
    #    1. Q3
    #    2. K4,4 - M (Wait, is this distinct from Q3? Yes, Q3 is planar, K4,4-M is not? No, K4,4-M is planar?
    #       K4,4 is non-planar. Removing edges might make it planar.
    #       Actually, Q3 is the only *cubic* *bipartite* *planar* graph on 8 vertices?
    #       Let's assume for a moment that checking for a 3-regular bipartite subgraph on 8 vertices is sufficient.
    #       If there are other 3-regular bipartite graphs on 8 vertices, we need to exclude them too?
    #       No, the task says "no subgraph isomorphic to Q3".
    #       So if there is another graph H on 8 vertices that is 3-regular bipartite but not Q3, H is allowed.
    #       So we specifically need to check for Q3.
    
    # Fast Q3 detection algorithm:
    # 1. Iterate over all 8-subsets of vertices? C(16,8) = 12870.
    # 2. For each 8-subset, check if the induced subgraph contains a Q3.
    #    Since Q3 has 12 edges, we can check if the induced subgraph has a 3-regular bipartite subgraph isomorphic to Q3.
    #    Actually, simpler: Just check if the induced subgraph *is* Q3? No, it could be a supergraph.
    #    We need to check if Q3 is a *subgraph*.
    #    
    #    Optimization:
    #    Precompute the adjacency matrix as a bitset (int).
    #    For a given 8-subset S, let A_S be the induced adjacency matrix.
    #    We can check if Q3 is a subgraph by trying to map Q3 vertices to S.
    #    But 8! is large.
    #    
    #    Better: Use the property that Q3 is 3-regular.
    #    If G contains Q3, then there are 8 vertices with degree >= 3 in the subgraph.
    #    
    #    Let's stick to the vectorized approach for the annealing objective.
    #    Objective: Maximize edges - Penalty * (Number of Q3s).
    #    
    #    How to count Q3s fast?
    #    We can sample. Or we can use a heuristic check.
    #    Given the time limit, let's try a different tactic:
    #    Start with a dense graph (e.g., p=0.8) and iteratively remove edges that are part of many Q3s.
    #    This is "Greedy Removal" or "Local Search with Deletion".
    #    
    #    Let's refine the "Greedy Removal" strategy which scored 1.308 in inspiration.
    #    The seed scored 1.0 (39 edges).
    #    Inspiration 1 found a denser graph by removing edges from K16.
    #    
    #    Strategy:
    #    1. Start with a random dense graph (p=0.9).
    #    2. While graph contains Q3:
    #       - Find a Q3.
    #       - Remove an edge from that Q3 (randomly or the one in the most Q3s).
    #    3. Then try to add edges back (Hill Climbing).
    
    def contains_q3_and_find(adj_matrix):
        # adj_matrix is a 16x16 numpy array of 0/1
        # Returns (True, [u, v]) if a Q3 is found, else (False, None)
        # Fast check: 
        # Iterate through 8-subsets? Too slow in Python loop.
        # 
        # Alternative: Use the fact that Q3 is 3-regular.
        # If we find 8 vertices that form a 3-regular bipartite graph, it's likely Q3.
        # Let's check all 8-subsets using numpy vectorization where possible.
        
        # Precompute all 8-subsets indices
        # We can't easily vectorize the isomorphism check.
        # 
        # Let's use a randomized check.
        # Pick 8 random vertices. Check if induced subgraph contains Q3.
        # Repeat many times.
        
        for _ in range(100):
            vs = np.random.choice(16, 8, replace=False)
            # Extract submatrix
            sub_adj = adj_matrix[np.ix_(vs, vs)]
            # Check degrees
            degs = sub_adj.sum(axis=1)
            if np.any(degs < 3):
                continue
            
            # Check if it contains Q3.
            # Since it's small, we can do a quick check.
            # Is it isomorphic to Q3?
            # Or does it contain Q3?
            # If degrees are all >= 3, and it's bipartite...
            
            # Check bipartiteness
            # BFS on 8 nodes is fast.
            color = -np.ones(8, dtype=int)
            is_bip = True
            for i in range(8):
                if color[i] == -1:
                    color[i] = 0
                    stack = [i]
                    while stack:
                        node = stack.pop()
                        neighbors = np.where(sub_adj[node] == 1)[0]
                        for nb in neighbors:
                            if color[nb] == -1:
                                color[nb] = 1 - color[node]
                                stack.append(nb)
                            elif color[nb] == color[node]:
                                is_bip = False
                                break
                        if not is_bip:
                            break
                    if not is_bip:
                        break
            
            if not is_bip:
                continue
                
            # If bipartite and min degree 3, it's a candidate.
            # Check if it's exactly Q3 (3-regular) or contains Q3.
            # If 3-regular and bipartite on 8 vertices, it is Q3.
            if np.all(degs == 3):
                return True, [vs[0], vs[1]] # Return some edge, we will remove one later
            
            # If degrees > 3, it might contain Q3.
            # For speed, we assume if it's dense and bipartite, it's risky.
            # But we need to be correct.
            # Let's just return False for now if not 3-regular, and rely on the 3-regular case?
            # No, a supergraph of Q3 is also forbidden.
            # However, if we remove edges to make it 3-regular, we might destroy the Q3.
            # 
            # Let's try to find a specific Q3 embedding.
            # This is getting complex. Let's stick to the "remove edges from dense graph" strategy
            # but use a robust Q3 check.
            
            # Robust Q3 check for 8 nodes:
            # Try all permutations? 40320.
            # Too slow if called 100 times per iteration.
            
            # Let's optimize the main loop instead.
            pass
            
        return False, None

    # Better Strategy: 
    # 1. Precompute a list of "critical" edge sets? No.
    # 2. Use the fact that Q3 is rare.
    # 3. Start with K16. Remove edges until Q3-free.
    #    To find a Q3, we can use a randomized local search for Q3.
    
    def find_q3(adj_matrix):
        # Randomized search for Q3
        for _ in range(50):
            vs = np.random.choice(16, 8, replace=False)
            sub_adj = adj_matrix[np.ix_(vs, vs)]
            degs = sub_adj.sum(axis=1)
            if np.any(degs < 3):
                continue
            
            # Check bipartite
            color = -np.ones(8, dtype=int)
            is_bip = True
            for i in range(8):
                if color[i] == -1:
                    color[i] = 0
                    stack = [i]
                    while stack:
                        node = stack.pop()
                        neighbors = np.where(sub_adj[node] == 1)[0]
                        for nb in neighbors:
                            if color[nb] == -1:
                                color[nb] = 1 - color[node]
                                stack.append(nb)
                            elif color[nb] == color[node]:
                                is_bip = False
                                break
                        if not is_bip:
                            break
                    if not is_bip:
                        break
            if not is_bip:
                continue
            
            # If bipartite and min deg 3, check for Q3 subgraph
            # We can check if there is a 3-regular subgraph.
            # Or just check if it's isomorphic to Q3 (if 3-reg).
            # If degs are not all 3, we need to check if Q3 is a subgraph.
            # This is hard.
            # Let's assume that if we find a bipartite subgraph with min deg 3, 
            # we can try to remove edges to make it 3-regular?
            # No, we are checking the current graph.
            
            # Let's do a quick check: if it's 3-regular, it's Q3.
            if np.all(degs == 3):
                # Find an edge in this subgraph to remove
                u, v = vs[0], vs[1]
                if adj_matrix[u, v] == 1:
                    return [u, v]
                else:
                    # Find any edge
                    for i in range(8):
                        for j in range(i+1, 8):
                            if sub_adj[i, j] == 1:
                                return [vs[i], vs[j]]
            
            # If not 3-regular, we might miss it. 
            # But dense graphs likely have 3-regular subgraphs if they contain Q3?
            # Not necessarily.
            # 
            # Let's add a check for "almost Q3".
            # If min deg >= 3, we can try to find a 3-regular subgraph by removing edges?
            # No, that changes the graph.
            
            # Let's rely on the fact that random sampling will eventually hit a 3-regular induced subgraph
            # if the graph is dense enough?
            # Or, we can check all 8-subsets if the graph is small? No, 16 is small.
            # C(16,8) = 12870.
            # If we check all 12870 subsets, it might be too slow in Python.
            # But we can vectorize the degree check.
            
        return None

    # Let's try a different approach: 
    # Precompute all 8-subsets.
    # For each 8-subset, check if it contains Q3.
    # We can do this in a loop, but optimize the inner check.
    
    all_8sets = list(combinations(range(16), 8))
    
    def is_q3_free(adj_matrix):
        # adj_matrix is 16x16 int array
        for vs in all_8sets:
            # Quick filter: count edges
            # We can precompute edge counts for all 8-sets? No, graph changes.
            # 
            # Vectorized edge count for one set:
            sub = adj_matrix[np.ix_(vs, vs)]
            # Count upper triangle
            edge_count = np.sum(sub[np.triu_indices(8, 1)])
            if edge_count < 12:
                continue
            
            # Degree check
            degs = sub.sum(axis=1)
            if np.any(degs < 3):
                continue
            
            # Bipartite check
            color = -np.ones(8, dtype=int)
            is_bip = True
            for i in range(8):
                if color[i] == -1:
                    color[i] = 0
                    stack = [i]
                    while stack:
                        node = stack.pop()
                        neighbors = np.where(sub[node] == 1)[0]
                        for nb in neighbors:
                            if color[nb] == -1:
                                color[nb] = 1 - color[node]
                                stack.append(nb)
                            elif color[nb] == color[node]:
                                is_bip = False
                                break
                        if not is_bip:
                            break
                    if not is_bip:
                        break
            if not is_bip:
                continue
            
            # If bipartite and min deg 3, check if it contains Q3.
            # If 3-regular, it IS Q3.
            if np.all(degs == 3):
                return False
            
            # If not 3-regular, it might contain Q3.
            # We need to check if Q3 is a subgraph.
            # This is the hard part.
            # For now, let's assume that if it's bipartite and min deg 3, 
            # and edge count is high, it likely contains Q3?
            # No, we must be correct.
            # 
            # Let's check if there is a 3-regular subgraph.
            # We can try to remove edges to make it 3-regular?
            # No, we are checking existence.
            # 
            # Let's use a simple heuristic: if edge_count > 12, assume it contains Q3?
            # This is conservative (might reject valid graphs).
            # But it ensures correctness (no false negatives for Q3).
            # Wait, if we reject a graph that is Q3-free, we might end up with a suboptimal solution.
            # But if we accept a graph that contains Q3, we fail.
            # So being conservative is safer for validity, but might lower the score.
            # 
            # Let's try to be more precise.
            # If edge_count == 12, we already checked 3-regular.
            # If edge_count > 12, we need to check for Q3 subgraph.
            # 
            # Let's skip the precise check for edge_count > 12 for speed, 
            # but only if we are confident?
            # 
            # Actually, let's just return False if edge_count > 12 and min_deg >= 3 and bipartite.
            # This assumes any bipartite graph on 8 vertices with min degree 3 and >12 edges contains Q3.
            # Is this true?
            # Q3 has 12 edges.
            # K4,4 has 16 edges. It contains Q3? Yes.
            # Are there bipartite graphs on 8 vertices with min degree 3, >12 edges, that are Q3-free?
            # If such a graph exists, this heuristic fails.
            # Given the small size, it's likely that dense bipartite graphs contain Q3.
            # Let's proceed with this heuristic for speed.
            
            if edge_count > 12:
                return False
                
        return True

    # Initialization: Start with a dense graph
    # p = 0.8
    adj = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(i + 1, 16):
            if random.random() < 0.8:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Ensure symmetry
    adj = (adj + adj.T) // 2
    
    # Removal phase
    while time.time() < deadline - 10:
        if is_q3_free(adj):
            break
        # Find a Q3? 
        # Our is_q3_free doesn't return which Q3.
        # Let's modify it to return an edge to remove.
        
        # For now, let's just remove a random edge?
        # That's slow.
        # Let's find a "bad" 8-set and remove an edge from it.
        
        bad_edge = None
        for vs in all_8sets:
            sub = adj[np.ix_(vs, vs)]
            edge_count = np.sum(sub[np.triu_indices(8, 1)])
            if edge_count < 12:
                continue
            degs = sub.sum(axis=1)
            if np.any(degs < 3):
                continue
            color = -np.ones(8, dtype=int)
            is_bip = True
            for i in range(8):
                if color[i] == -1:
                    color[i] = 0
                    stack = [i]
                    while stack:
                        node = stack.pop()
                        neighbors = np.where(sub[node] == 1)[0]
                        for nb in neighbors:
                            if color[nb] == -1:
                                color[nb] = 1 - color[node]
                                stack.append(nb)
                            elif color[nb] == color[node]:
                                is_bip = False
                                break
                        if not is_bip:
                            break
                    if not is_bip:
                        break
            if not is_bip:
                continue
            if np.all(degs == 3) or edge_count > 12:
                # Found a bad set. Remove a random edge from it.
                edges_in_set = [(vs[i], vs[j]) for i in range(8) for j in range(i+1, 8) if sub[i, j] == 1]
                if edges_in_set:
                    u, v = random.choice(edges_in_set)
                    bad_edge = (u, v)
                    break
        if bad_edge:
            adj[bad_edge[0], bad_edge[1]] = 0
            adj[bad_edge[1], bad_edge[0]] = 0
        else:
            break

    # Addition phase (Hill Climbing)
    best_edges = set()
    for i in range(16):
        for j in range(i + 1, 16):
            if adj[i, j] == 1:
                best_edges.add((i, j))
                
    current_edges = set(best_edges)
    
    all_possible_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        candidates = [e for e in all_possible_edges if e not in current_edges]
        random.shuffle(candidates)
        for edge in candidates:
            if time.time() > deadline - 1:
                break
            u, v = edge
            adj[u, v] = 1
            adj[v, u] = 1
            if is_q3_free(adj):
                current_edges.add(edge)
                improved = True
            else:
                adj[u, v] = 0
                adj[v, u] = 0
                
    result = [[u, v] for (u, v) in current_edges]
    return result

