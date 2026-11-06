import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 16
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: idx for idx, e in enumerate(all_edges)}
    num_edges_total = len(all_edges)
    
    # Bitmask representation for adjacency
    # adj_mask[i] is an integer where bit j is 1 if edge (i,j) exists
    
    def is_q3_free(adj_masks):
        """
        Fast check if the graph defined by adj_masks contains a Q3.
        A Q3 is a 3-regular bipartite graph on 8 vertices.
        It is the unique 3-regular bipartite graph on 8 vertices.
        
        Strategy for detection:
        Iterate over all 8-vertex subsets S.
        For each S, check if the induced subgraph is isomorphic to Q3.
        Since n=16, C(16,8) = 12870. This is small enough to do frequently
        if the inner check is fast.
        
        Inner check:
        1. Check if induced subgraph has exactly 12 edges.
        2. Check if it is 3-regular.
        3. Check if it is bipartite.
        (If 1 and 2 hold, and it's bipartite, it must be Q3? 
         Yes, the only 3-regular bipartite graph on 8 vertices is Q3.
         K_{3,3} is on 6 vertices. K_{4,4} is 4-regular.
         So yes, 3-regular + bipartite + 8 vertices => Q3.)
        """
        # Precompute list of 8-vertex subsets? 
        # Doing it on the fly is fine for 12870 subsets, but let's cache the vertex sets.
        
        # Optimization: 
        # A Q3 has 6 faces (4-cycles). 
        # But checking 12 edges and degrees is faster.
        
        for S in combinations(range(n), 8):
            S_set = set(S)
            # Count edges in induced subgraph
            edge_count = 0
            degrees = {v: 0 for v in S}
            
            for u in S:
                mask = adj_masks[u]
                # Intersect with S
                # We need to count bits in mask that correspond to vertices in S
                # and ensure we don't double count.
                # Let's just iterate over pairs in S. 28 pairs.
                for v in S:
                    if u < v:
                        # Check if edge exists
                        # adj_masks[u] has bit v set if edge exists
                        if (adj_masks[u] >> v) & 1:
                            edge_count += 1
                            degrees[u] += 1
                            degrees[v] += 1
            
            if edge_count != 12:
                continue
            
            # Check 3-regular
            is_reg = True
            for v in S:
                if degrees[v] != 3:
                    is_reg = False
                    break
            if not is_reg:
                continue
            
            # Check bipartite
            # BFS/DFS on the 8 vertices
            color = {}
            is_bip = True
            for start in S:
                if start in color:
                    continue
                color[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop()
                    # Get neighbors in S
                    # adj_masks[u] & mask_of_S
                    # We need a mask for S to quickly get neighbors
                    # Let's just iterate S
                    for v in S:
                        if u == v:
                            continue
                        if (adj_masks[u] >> v) & 1:
                            if v not in color:
                                color[v] = color[u] ^ 1
                                queue.append(v)
                            elif color[v] == color[u]:
                                is_bip = False
                                break
                    if not is_bip:
                        break
                if not is_bip:
                    break
            
            if is_bip:
                # Found a Q3
                return False, S
        
        return True, None

    def find_q3_and_edge(adj_masks):
        """
        Find a Q3 and return the Q3 vertices and a specific edge to remove.
        We want to remove an edge that is likely to be in many Q3s.
        Heuristic: Remove an edge with high degree in the current graph?
        Or just return the first one found and let the caller decide?
        For speed, let's just return the first Q3 found and the edge with 
        max (deg(u)+deg(v)) among the 12 edges of that Q3.
        """
        for S in combinations(range(n), 8):
            S_list = list(S)
            S_set = set(S)
            edge_count = 0
            degrees = {v: 0 for v in S}
            edges_in_s = []
            
            for i in range(8):
                u = S_list[i]
                for j in range(i+1, 8):
                    v = S_list[j]
                    if (adj_masks[u] >> v) & 1:
                        edge_count += 1
                        degrees[u] += 1
                        degrees[v] += 1
                        edges_in_s.append((u, v))
            
            if edge_count != 12:
                continue
            
            is_reg = True
            for v in S:
                if degrees[v] != 3:
                    is_reg = False
                    break
            if not is_reg:
                continue
            
            # Check bipartite
            color = {}
            is_bip = True
            for start in S:
                if start in color:
                    continue
                color[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop()
                    for v in S:
                        if u == v:
                            continue
                        if (adj_masks[u] >> v) & 1:
                            if v not in color:
                                color[v] = color[u] ^ 1
                                queue.append(v)
                            elif color[v] == color[u]:
                                is_bip = False
                                break
                    if not is_bip:
                        break
                if not is_bip:
                    break
            
            if is_bip:
                # Found Q3 on S
                # Choose edge to remove: max degree sum
                best_edge = None
                max_score = -1
                for u, v in edges_in_s:
                    score = degrees[u] + degrees[v]
                    # Tie-breaker: random? or lowest index?
                    if score > max_score:
                        max_score = score
                        best_edge = (u, v)
                return S, best_edge
        
        return None, None

    # Strategy:
    # 1. Start with a dense random graph (p=0.6)
    # 2. Repair phase: While Q3 exists, remove an edge.
    # 3. Optimization phase: Try to add back edges.
    
    best_edge_count = 0
    best_adj = [0] * n
    
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        adj_masks = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.6:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        # Phase 1: Remove Q3s
        # Limit steps to avoid getting stuck in a loop if heuristic is bad
        for step in range(1000):
            if time.time() > deadline - 5:
                break
                
            # Check if Q3 free
            free, S = is_q3_free(adj_masks)
            if free:
                break
                
            # If not free, S is the Q3 vertices.
            # Find the specific edge to remove from this Q3.
            # We already have S. Let's re-derive the edges in S to pick one.
            # To save time, we can just pick a random edge in S?
            # No, the heuristic in find_q3_and_edge is better.
            # But is_q3_free already found S. Let's just use a simple heuristic:
            # Pick the edge in S with highest degree sum.
            
            S_list = list(S)
            edges_in_s = []
            degs = {v: 0 for v in S}
            for i in range(8):
                u = S_list[i]
                for j in range(i+1, 8):
                    v = S_list[j]
                    if (adj_masks[u] >> v) & 1:
                        edges_in_s.append((u, v))
                        degs[u] += 1
                        degs[v] += 1
            
            # Pick edge with max deg sum
            best_e = None
            max_s = -1
            for u, v in edges_in_s:
                s = degs[u] + degs[v]
                if s > max_s:
                    max_s = s
                    best_e = (u, v)
            
            if best_e:
                u, v = best_e
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
            else:
                break
        
        # Check final state of this restart
        free, _ = is_q3_free(adj_masks)
        if not free:
            continue # Failed to repair
        
        # Phase 2: Add edges greedily
        # Try adding missing edges in random order
        missing_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj_masks[u] >> v) & 1):
                    missing_edges.append((u, v))
        
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline - 5:
                break
            
            # Tentatively add
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
            # Check Q3
            free, _ = is_q3_free(adj_masks)
            if not free:
                # Revert
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
        
        # Count edges
        count = sum(bin(m).count('1') for m in adj_masks) // 2
        
        if count > best_edge_count:
            best_edge_count = count
            best_adj = adj_masks[:]
            
    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result_edges.append([u, v])
                
    return result_edges

# EVOLVE-BLOCK-END
