import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n = 16
    
    # Precompute all 8-vertex subsets
    all_8sets = list(combinations(range(n), 8))
    
    # Q3 adjacency for backtracking check
    # Vertices 0-7. 0-3 and 4-7 are the bipartition?
    # Q3: 0-1-3-2-0 and 4-5-7-6-4, with 0-4, 1-5, 3-7, 2-6.
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    def is_q3(vertices, adj):
        """
        vertices: tuple of 8 ints
        adj: list of 16 ints (bitmasks)
        Returns True if the induced subgraph on vertices is isomorphic to Q3.
        """
        # Map global vertices to local indices 0-7
        # local_adj[i] is bitmask of neighbors in local indices
        local_adj = [0] * 8
        v_list = list(vertices)
        
        # Build local adjacency
        for i in range(8):
            u = v_list[i]
            mask = adj[u]
            for j in range(8):
                if j != i and (mask >> v_list[j]) & 1:
                    local_adj[i] |= (1 << j)
            
        # Check 3-regular
        for i in range(8):
            if bin(local_adj[i]).count('1') != 3:
                return False
        
        # Check bipartite and structure via backtracking
        # We try to map Q3 vertices 0..7 to local indices 0..7
        # such that edges match.
        
        # Optimization: Try to map Q3 vertex 0 to the local vertex with 
        # the "most constrained" neighborhood? 
        # All local vertices have degree 3.
        # Let's just try all 8! permutations? 40320. Too slow.
        
        # Backtracking:
        # Assign q3 vertices to local indices.
        # q3_adj is fixed.
        
        assignment = [-1] * 8 # q3_v -> local_v
        local_assigned = [False] * 8 # local_v -> is assigned?
        
        # We can reduce symmetry. Q3 is highly symmetric.
        # Fix q3_vertex 0 to local_vertex 0.
        # Then we have 7! = 5040 permutations. Still a bit high if done 12870 times.
        
        # Better: Check if local graph is isomorphic to Q3 using invariants.
        # 1. 3-regular (done)
        # 2. Bipartite?
        # 3. Number of 4-cycles? Q3 has 6 4-cycles (faces).
        
        # Count 4-cycles in local graph.
        # A 4-cycle u-v-w-x-u.
        # Iterate u < v < w < x?
        # Or just iterate pairs (u,v) and (w,x) disjoint?
        
        # Simpler check for Q3:
        # Q3 has diameter 3.
        # Distances in Q3:
        # d=1: 12 pairs (edges)
        # d=2: 12 pairs
        # d=3: 4 pairs (opposites)
        
        # Let's compute distance distribution in local graph.
        # If it matches Q3, it's likely Q3.
        
        # Compute all-pairs shortest path? 8x8 BFS.
        # Or just count pairs at distance 1, 2, 3.
        
        dist_count = [0, 0, 0, 0] # index is distance 1,2,3
        
        # We already have local_adj.
        # d=1: sum(deg)/2 = 12. (Already checked 3-regular, so 12 edges)
        dist_count[1] = 12
        
        # d=2: pairs (u,v) such that dist(u,v)=2.
        # u ~ x ~ v.
        # For each u, look at neighbors x. For each neighbor y of x, if y != u and y != v?
        # We want to count unordered pairs {u,v} with dist 2.
        
        # Let's just run BFS from each node to get distances.
        # 8 BFS runs on 8 nodes. Very fast.
        
        from collections import deque
        
        dist_matrix = [[-1]*8 for _ in range(8)]
        for start in range(8):
            dist_matrix[start][start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                d = dist_matrix[start][u]
                # Iterate neighbors
                m = local_adj[u]
                # get bits
                v = 0
                while m:
                    lsb = m & -m
                    v_idx = lsb.bit_length() - 1
                    if dist_matrix[start][v_idx] == -1:
                        dist_matrix[start][v_idx] = d + 1
                        q.append(v_idx)
                    m ^= lsb
        
        # Count distances
        # Distances should be 1, 2, or 3 for Q3.
        # If any distance is > 3, return False.
        # Count pairs (i,j) i<j with dist 2 and dist 3.
        
        d2_count = 0
        d3_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                d = dist_matrix[i][j]
                if d == 2:
                    d2_count += 1
                elif d == 3:
                    d3_count += 1
                elif d > 3:
                    return False
        
        # Q3 has 12 pairs at dist 2 and 4 pairs at dist 3.
        if d2_count == 12 and d3_count == 4:
            return True
        return False

    def find_q3(adj):
        """Find any Q3 in the graph defined by adj."""
        for vs in all_8sets:
            if is_q3(vs, adj):
                return vs
        return None

    def edge_creates_q3(u, v, adj):
        """Check if adding edge (u,v) creates a Q3."""
        # Only 8-sets containing u and v need to be checked.
        # Precompute which 8-sets contain which pair?
        # Or just iterate all 8-sets and skip those not containing u and v.
        # 12870 sets. 2/16 * 1/15 prob?
        # Number of sets containing u and v: C(14, 6) = 3003.
        
        for vs in all_8sets:
            if u in vs and v in vs:
                # Check if this set is Q3 with the new edge
                # We need to check if the subgraph induced by vs is Q3.
                # But we don't have the new adj in 'adj'.
                # We can simulate.
                
                # Actually, 'is_q3' takes 'adj'.
                # We can create a modified adj? No, too slow.
                
                # Let's modify 'is_q3' to accept a "extra edge" or just 
                # pass a temporary structure.
                
                # Better: Pre-filter. 
                # If the set was NOT a Q3 before, adding one edge might make it one.
                # But we don't track "was Q3".
                
                # Let's just call a specialized check.
                if is_q3_with_edge(vs, u, v, adj):
                    return True
        return False

    def is_q3_with_edge(vertices, u, v, adj):
        """
        Check if vertices induce Q3, assuming edge (u,v) is present.
        u, v are global indices.
        """
        # Map u, v to local indices
        local_u = -1
        local_v = -1
        v_list = list(vertices)
        for i in range(8):
            if v_list[i] == u:
                local_u = i
            if v_list[i] == v:
                local_v = i
            
        if local_u == -1 or local_v == -1:
            return False
        
        local_adj = [0] * 8
        for i in range(8):
            node = v_list[i]
            mask = adj[node]
            for j in range(8):
                if j != i and (mask >> v_list[j]) & 1:
                    local_adj[i] |= (1 << j)
        
        # Add the extra edge
        local_adj[local_u] |= (1 << local_v)
        local_adj[local_v] |= (1 << local_u)
        
        # Now check is_q3 logic with this local_adj
        # We can refactor is_q3 to take local_adj
        
        return _check_q3_local(local_adj)

    def _check_q3_local(local_adj):
        # Check 3-regular
        for i in range(8):
            if bin(local_adj[i]).count('1') != 3:
                return False
        
        # Distances
        from collections import deque
        dist_matrix = [[-1]*8 for _ in range(8)]
        for start in range(8):
            dist_matrix[start][start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                d = dist_matrix[start][u]
                m = local_adj[u]
                v_idx = 0
                while m:
                    lsb = m & -m
                    v_idx = lsb.bit_length() - 1
                    if dist_matrix[start][v_idx] == -1:
                        dist_matrix[start][v_idx] = d + 1
                        q.append(v_idx)
                    m ^= lsb
        
        d2_count = 0
        d3_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                d = dist_matrix[i][j]
                if d == 2:
                    d2_count += 1
                elif d == 3:
                    d3_count += 1
                elif d > 3:
                    return False
        
        if d2_count == 12 and d3_count == 4:
            return True
        return False

    # Main Search
    best_edges = set()
    best_count = 0
    
    # Initialize
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Start with random graph
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        # Add edges greedily if possible? 
        # Or start with K16 and remove?
        # K16 has 120 edges. Q3 free? No.
        # Let's start with K16 and remove edges to destroy Q3s.
        
        adj = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i] |= (1 << j)
        
        current_edges = set(all_pairs)
        
        # Remove edges to make Q3-free
        # While there is a Q3, remove a random edge from it.
        for _ in range(500):
            if time.time() > deadline - 5:
                break
                
            q3 = find_q3(adj)
            if q3 is None:
                break
            
            # Pick a random edge in q3
            # The edges of q3 are the pairs in q3 that are connected in adj.
            # We know it's Q3, so we can list them.
            # But we just need one edge to remove.
            # Let's pick a random pair from the 8 vertices.
            # It must be an edge in the induced subgraph.
            # Since it's Q3, there are 12 edges.
            
            # Let's find the actual edges in this Q3.
            # We can just pick a random pair (u,v) from q3 and check if adj[u] has v.
            # Since it's a Q3, 12 out of 28 pairs are edges.
            
            candidates = []
            for i in range(8):
                for j in range(i+1, 8):
                    u = q3[i]
                    v = q3[j]
                    if (adj[u] >> v) & 1:
                        candidates.append((u, v))
            
            if candidates:
                u, v = random.choice(candidates)
                # Remove edge
                current_edges.remove((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Should not happen if find_q3 is correct
                break
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            best_adj = adj[:]
            
        # Local Search on best_edges
        # Try to add missing edges
        missing = [e for e in all_pairs if e not in best_edges]
        random.shuffle(missing)
        
        current_adj = best_adj[:]
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for u, v in missing:
                if time.time() > deadline - 15:
                    break
                if (u, v) in best_edges:
                    continue
                # Check if adding (u,v) creates Q3
                if not edge_creates_q3(u, v, current_adj):
                    # Add it
                    best_edges.add((u, v))
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    best_count += 1
                    improved = True
                    # Update missing list? 
                    # Just leave it, the check "if (u,v) in best_edges" handles it.
                    # But removing from missing is faster?
                    # For now, leave it.

    result = [[u, v] for (u, v) in best_edges]
    return result

# EVOLVE-BLOCK-END
