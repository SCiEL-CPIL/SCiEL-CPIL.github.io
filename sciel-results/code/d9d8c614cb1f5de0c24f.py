import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    # Q3 adjacency structure (vertices 0-7)
    # Edges in Q3: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # Precompute all 8-vertex subsets
    all_8sets = list(combinations(range(16), 8))
    
    # Map each 8-set to the specific edges (indices in the 8-set) that form a Q3
    # For a given 8-set S, we need to check if the induced subgraph is isomorphic to Q3.
    # Instead of full isomorphism check every time, we can precompute the set of edge-pairs (within the 8-set)
    # that constitute a Q3. There are 480 labelings of Q3 on 8 vertices.
    # However, checking isomorphism is expensive.
    
    # Alternative efficient check:
    # A graph is Q3 if and only if it has 8 vertices, 12 edges, is 3-regular, and is bipartite with parts of size 4.
    # Actually, Q3 is the unique 3-regular bipartite graph on 8 vertices? No, there are others (e.g., prism graph? No, prism is 4-regular or different size).
    # Q3 is the unique 3-regular bipartite graph on 8 vertices that is not the complete bipartite K4,4 (which is 4-regular).
    # Wait, the 3-regular bipartite graphs on 8 vertices:
    # 1. Q3 (cube)
    # 2. The prism graph is 3-regular but not bipartite (has triangles? No, prism has 4-cycles. Wait, triangular prism has triangles. C4 prism is 3-regular bipartite? No, C4 x K2 is the cube. C3 x K2 is triangular prism, has triangles. C5 x K2 is 3-regular bipartite on 10 vertices).
    # So on 8 vertices, 3-regular bipartite implies Q3?
    # Let's verify. Bipartite 3-regular on 8 vertices means two sets of 4.
    # Number of edges = 12.
    # Is it unique?
    # The incidence matrix is 4x4 with row/col sums 3.
    # This means the complement is a 1-regular bipartite graph (perfect matching).
    # So Q3 is K4,4 minus a perfect matching.
    # Are there other 3-regular bipartite graphs on 8 vertices?
    # If the complement is a perfect matching, the graph is determined by the matching.
    # All perfect matchings in K4,4 are isomorphic under vertex permutation?
    # Yes, the automorphism group of K4,4 acts transitively on perfect matchings.
    # So YES, any 3-regular bipartite graph on 8 vertices is isomorphic to Q3.
    
    # Therefore, checking if an induced subgraph on 8 vertices is Q3 is equivalent to:
    # 1. Counting edges in the induced subgraph is 12.
    # 2. Checking if it is bipartite.
    # (If it has 12 edges on 8 vertices and is bipartite, it must be 3-regular? 
    #  Sum of degrees = 24. If bipartite, parts must be 4 and 4 (since max degree 3 in bipartite with 4+4? 
    # If parts were 3 and 5, max edges = 3*5=15. But if it's 3-regular, degrees are 3. 
    # If it's just "12 edges and bipartite", does it imply 3-regular?
    # Not necessarily. E.g., a vertex could have degree 2.
    # However, if we check "Is it isomorphic to Q3", we can use the property:
    # Q3 is 3-regular. So we must check degree sequence is all 3s AND it is bipartite.
    
    def is_q3(vs, adj_bits):
        """Check if vertices vs induce a Q3."""
        # Map global vertices to local indices 0..7
        local_indices = {v: i for i, v in enumerate(vs)}
        
        # Build local adjacency list
        local_adj = [[] for _ in range(8)]
        edge_count = 0
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(i+1, 8):
                v = vs[j]
                if (mask >> v) & 1:
                    local_adj[i].append(j)
                    local_adj[j].append(i)
                    edge_count += 1
        
        if edge_count != 12:
            return False
            
        # Check 3-regular
        for i in range(8):
            if len(local_adj[i]) != 3:
                return False
                
        # Check bipartite (BFS)
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in local_adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
        return True

    def count_q3s(adj_bits):
        """Count number of Q3 subgraphs in the graph defined by adj_bits."""
        count = 0
        for vs in all_8sets:
            if is_q3(vs, adj_bits):
                count += 1
        return count

    def get_edges(adj_bits):
        edges = set()
        for i in range(16):
            for j in range(i+1, 16):
                if (adj_bits[i] >> j) & 1:
                    edges.add((i, j))
        return edges

    # Objective function: Maximize (number of edges - penalty * number of Q3s)
    # We want to find a graph with 0 Q3s and max edges.
    # During search, we allow Q3s but penalize them heavily.
    
    # Initialize with K16 (complete graph)
    adj_bits = [0] * 16
    for i in range(16):
        for j in range(i+1, 16):
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
            
    current_edges_count = 120
    current_q3_count = count_q3s(adj_bits)
    
    # We will use Simulated Annealing.
    # State: adj_bits
    # Move: Flip a random edge.
    # Energy: -current_edges_count + penalty * current_q3_count
    
    best_adj = adj_bits[:]
    best_edges_count = 0
    best_q3_count = 1000000
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    penalty = 5.0 # Heuristic: Q3s are bad.
    
    no_improve = 0
    max_no_improve = 500
    
    # Precompute edge list for fast iteration
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    while time.time() < deadline and T > T_min:
        if no_improve > max_no_improve:
            # Restart from best or random
            if random.random() < 0.5:
                adj_bits = best_adj[:]
                current_edges_count = best_edges_count
                current_q3_count = best_q3_count
            else:
                # Random restart
                adj_bits = [0]*16
                current_edges_count = 0
                for i in range(16):
                    for j in range(i+1, 16):
                        if random.random() < 0.5:
                            adj_bits[i] |= (1 << j)
                            adj_bits[j] |= (1 << i)
                            current_edges_count += 1
                current_q3_count = count_q3s(adj_bits)
            T = 5.0
            no_improve = 0
            continue
            
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        # Flip
        adj_bits[u] ^= (1 << v)
        adj_bits[v] ^= (1 << u)
        
        # Calculate delta
        # Delta edges is +1 or -1
        # Delta Q3 count requires re-evaluating Q3s containing u or v.
        # This is expensive if done naively. 
        # Optimization: Only check 8-sets containing both u and v? 
        # No, flipping (u,v) affects any Q3 that contains the edge (u,v).
        # If (u,v) was present, removing it might destroy Q3s.
        # If (u,v) was absent, adding it might create Q3s.
        
        # Let's just recalculate Q3 count for efficiency in coding first, 
        # but optimize if too slow. 11440 subsets * 12 edge checks is ~100k ops.
        # 120s / 100k ops = 1200 ops/sec? No, Python is slower.
        # 100k ops might take 0.1s - 0.5s. 
        # 120s / 0.1s = 1200 iterations. That's not enough for SA.
        
        # We need a faster Q3 count update.
        # Precompute for each edge (u,v), the list of 8-sets containing u and v.
        # If we add (u,v), we only need to check those 8-sets to see if they BECOME Q3.
        # If we remove (u,v), we only need to check those 8-sets to see if they STOP being Q3.
        
        # Let's implement this incremental check.
        
        # Precomputation
        pair_to_8sets = {}
        for vs in all_8sets:
            for i in range(8):
                for j in range(i+1, 8):
                    pair = (vs[i], vs[j])
                    if pair not in pair_to_8sets:
                        pair_to_8sets[pair] = []
                    pair_to_8sets[pair].append(vs)
        
        # Re-initialize loop logic with incremental check
        # We need to know the state of the graph BEFORE the flip to determine delta.
        # Let's revert the flip and do it properly.
        
        adj_bits[u] ^= (1 << v)
        adj_bits[v] ^= (1 << u)
        
        edge_was_present = (adj_bits[u] >> v) & 1
        
        # Apply flip
        adj_bits[u] ^= (1 << v)
        adj_bits[v] ^= (1 << u)
        
        new_edges_count = current_edges_count + (1 if not edge_was_present else -1)
        
        # Calculate delta Q3
        # Only 8-sets containing (u,v) are affected.
        affected_8sets = pair_to_8sets.get((u, v), [])
        
        delta_q3 = 0
        if not edge_was_present:
            # Adding edge. Check if any affected 8-set BECOMES a Q3.
            for vs in affected_8sets:
                if is_q3(vs, adj_bits):
                    delta_q3 += 1
        else:
            # Removing edge. Check if any affected 8-set STOPS being a Q3.
            # We need to check if it WAS a Q3 before removal.
            # But we already removed it. 
            # We can check if it is a Q3 in the graph WITHOUT the edge.
            # Wait, if we remove (u,v), the graph changes.
            # We need to know if it was a Q3 in the OLD graph.
            # Let's temporarily add it back, check, then remove?
            # Or just check: If is_q3(vs, adj_bits) is False now, it might have been True before.
            # If is_q3(vs, adj_bits) is True now, it was definitely True before (since removing edges can't create Q3s? 
            # No, removing edges destroys Q3s. So if it's a Q3 now, it was a Q3 before.
            # If it's NOT a Q3 now, it MIGHT have been a Q3 before.
            # So:
            #   If is_q3(vs, new_graph) == True: It was a Q3 before. Delta = -1 (lost one).
            #   If is_q3(vs, new_graph) == False: We need to check if it was a Q3 before.
            #       To check if it was a Q3 before, we add the edge back temporarily.
            
            for vs in affected_8sets:
                if is_q3(vs, adj_bits):
                    # Still a Q3 after removal? Impossible if (u,v) was required for Q3.
                    # Q3 is 3-regular. Removing an edge makes degree 2 for u and v.
                    # So it CANNOT be a Q3 after removal if (u,v) was in it.
                    # So this case should never happen.
                    pass
                else:
                    # It's not a Q3 now. Was it before?
                    # Temporarily add edge back
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    if is_q3(vs, adj_bits):
                        delta_q3 -= 1 # It was a Q3, now it's not.
                    # Restore removal
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
        
        new_q3_count = current_q3_count + delta_q3
        
        # Energy calculation
        # We want to maximize edges, minimize Q3s.
        # Objective = edges - penalty * q3s
        old_obj = current_edges_count - penalty * current_q3_count
        new_obj = new_edges_count - penalty * new_q3_count
        delta_obj = new_obj - old_obj
        
        # Acceptance
        accept = False
        if delta_obj > 0:
            accept = True
        else:
            # p = exp(delta_obj / T)
            # Since delta_obj is negative, this is exp(-|delta|/T)
            if T > 0:
                p = np.exp(delta_obj / T)
                if random.random() < p:
                    accept = True
        
        if accept:
            current_edges_count = new_edges_count
            current_q3_count = new_q3_count
            no_improve = 0
            
            # Update best if we have a valid Q3-free graph
            if current_q3_count == 0 and current_edges_count > best_edges_count:
                best_edges_count = current_edges_count
                best_adj = adj_bits[:]
        else:
            # Revert flip
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            no_improve += 1
            
        T *= cooling
        
    # Final check: ensure best_adj is Q3-free and return edges
    # It's possible best_adj has Q3s if we never found a Q3-free one?
    # We initialized best with K16 which has Q3s. 
    # We only update best if q3_count == 0.
    # If we never found one, best_adj is still K16 (invalid).
    # We must return a valid graph.
    # Fallback: Greedy removal from K16 or K3,13.
    
    if best_q3_count != 0 or best_edges_count == 0:
        # Fallback to a known valid construction or greedy repair
        # Let's use the greedy repair from K16 as a safe fallback
        adj = [0]*16
        for i in range(16):
            for j in range(i+1, 16):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Greedy removal
        edges_list = [(i, j) for i in range(16) for j in range(i+1, 16)]
        random.shuffle(edges_list)
        
        # We need to remove edges to destroy all Q3s.
        # This is a hitting set problem.
        # Simple heuristic: while Q3 exists, pick a random edge from it and remove.
        
        # To speed up, we can just run a quick SA on this fallback?
        # No, let's just do a simple greedy removal.
        
        # Actually, the SA above should have found a Q3-free graph if it ran long enough.
        # If not, we return a simple valid graph: K3,13 (39 edges)
        # Or we can try to repair best_adj by removing edges from Q3s.
        
        # Let's try to repair best_adj if it has few Q3s.
        # If best_adj is K16, it has many.
        
        # Safest: Return K3,13
        final_adj = [0]*16
        for i in range(3):
            for j in range(3, 16):
                final_adj[i] |= (1 << j)
                final_adj[j] |= (1 << i)
                
        # Or try to repair the best_adj found by SA
        # If best_q3_count is small, we can remove edges.
        if best_q3_count < 10:
            final_adj = best_adj[:]
            # Find Q3s and remove edges
            for _ in range(100): # Limit iterations
                q3_found = False
                for vs in all_8sets:
                    if is_q3(vs, final_adj):
                        # Pick a random edge in this Q3 and remove
                        # Edges in Q3 induced by vs
                        local_edges = []
                        for i in range(8):
                            u = vs[i]
                            for j in range(i+1, 8):
                                v = vs[j]
                                if (final_adj[u] >> v) & 1:
                                    local_edges.append((u, v))
                        if local_edges:
                            u, v = random.choice(local_edges)
                            final_adj[u] &= ~(1 << v)
                            final_adj[v] &= ~(1 << u)
                        q3_found = True
                        break
                if not q3_found:
                    break
            
            # Verify
            if count_q3s(final_adj) == 0:
                best_adj = final_adj
                best_edges_count = sum(bin(x).count('1') for x in best_adj) // 2

    # Construct result
    result = []
    for i in range(16):
        for j in range(i+1, 16):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result

# EVOLVE-BLOCK-END
