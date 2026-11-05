import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 16
    all_vertices = set(range(n))
    
    # Q3 edge list (8 vertices, 12 edges)
    # Standard Q3: vertices 0-7, edges along 3 dimensions
    q3_edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7)
    ]
    
    # For checking if adding edge (u,v) creates a Q3:
    # We need to find 6 other vertices that, together with u,v, form Q3.
    # In Q3, u and v are adjacent. Their common neighbors in Q3: none directly,
    # but they share neighbors through the cube structure.
    # Actually in Q3, two adjacent vertices have exactly 2 common neighbors.
    # Let's think about this differently.
    
    # A Q3 containing edge (u,v): we need to assign the remaining 6 vertices
    # to positions 2,3,4,5,6,7 in Q3 such that all 12 edges are present.
    
    # Precompute for each pair of vertices in Q3, which other vertices they connect to
    # This helps with pruning.
    
    # Key insight: In Q3, vertex 0 is connected to 1,2,4. Vertex 1 is connected to 0,3,5.
    # If we fix u->0, v->1, then:
    # - Vertices assigned to positions 2,4 must be neighbors of u (and not v)
    # - Vertices assigned to positions 3,5 must be neighbors of v (and not u)  
    # - Position 3 must also be neighbor of 2
    # - Position 5 must also be neighbor of 4
    # - Position 6 must be neighbor of 2 and 4
    # - Position 7 must be neighbor of 3 and 5
    # - Position 6 must be neighbor of 7
    # - Position 3 must be neighbor of 7
    # - Position 5 must be neighbor of 7
    
    def has_q3_with_edge(adj_mask, u, v):
        """Check if adding edge u-v creates a Q3. adj_mask is list of int bitmasks."""
        # Get neighbors of u and v (excluding each other)
        neu = [x for x in range(n) if x != u and x != v and (adj_mask[u] >> x) & 1]
        nev = [x for x in range(n) if x != v and x != u and (adj_mask[v] >> x) & 1]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        # In Q3 with u=0, v=1:
        # pos 2: neighbor of u, must connect to pos 3 and pos 6
        # pos 4: neighbor of u, must connect to pos 5 and pos 6
        # pos 3: neighbor of v, must connect to pos 2 and pos 7
        # pos 5: neighbor of v, must connect to pos 4 and pos 7
        # pos 6: neighbor of pos 2 and pos 4, must connect to pos 7
        # pos 7: neighbor of pos 3 and pos 5, must connect to pos 6
        
        # Try all assignments of 2 vertices from neu to {pos2, pos4}
        # and 2 vertices from nev to {pos3, pos5}
        # Then check if pos6, pos7 exist.
        
        for pos2 in neu:
            for pos4 in neu:
                if pos2 == pos4:
                    continue
                for pos3 in nev:
                    if pos3 == pos2 or pos3 == pos4:
                        continue
                    for pos5 in nev:
                        if pos5 == pos2 or pos5 == pos3 or pos5 == pos4:
                            continue
                        
                        # Check required edges among these 4
                        if not ((adj_mask[pos2] >> pos3) & 1): continue  # 2-3
                        if not ((adj_mask[pos4] >> pos5) & 1): continue  # 4-5
                        
                        # Now find pos6: must be neighbor of pos2 and pos4
                        # and not in {u,v,pos2,pos3,pos4,pos5}
                        excluded = {u, v, pos2, pos3, pos4, pos5}
                        cand_pos6 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos2] >> x) & 1) and ((adj_mask[pos4] >> x) & 1):
                                cand_pos6.append(x)
                        
                        # Find pos7: must be neighbor of pos3 and pos5
                        cand_pos7 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos3] >> x) & 1) and ((adj_mask[pos5] >> x) & 1):
                                cand_pos7.append(x)
                        
                        # Check if any pair (pos6, pos7) works
                        for pos6 in cand_pos6:
                            for pos7 in cand_pos7:
                                if pos6 == pos7: continue
                                # Check remaining edges: pos6-pos7, pos3-pos7, pos5-pos7
                                if ((adj_mask[pos6] >> pos7) & 1) and \
                                   ((adj_mask[pos3] >> pos7) & 1) and \
                                   ((adj_mask[pos5] >> pos7) & 1):
                                    return True
        return False
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def verify_no_q3(adj_mask):
        """Verify the graph has no Q3 subgraph."""
        # For each pair of adjacent vertices, check if they're part of a Q3
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    if has_q3_with_edge(adj_mask, u, v):
                        return False
        return True
    
    best_edge_count = 0
    best_adj_mask = None
    
    # Strategy: Start with various base graphs and greedily add edges
    # Try multiple random orderings to find good solutions
    
    random.seed(42)
    
    num_restarts = 200  # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize: start with random sparse graph or specific structure
        # Try different starting points
        adj_mask = [0] * n
        
        # Start with a random set of edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Random subset to start
        if restart == 0:
            # Start with K3,13 structure
            for u in range(3):
                for v in range(3, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        elif restart == 1:
            # Start with complete bipartite K8,8
            for u in range(8):
                for v in range(8, 16):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        elif restart == 2:
            # Start with complete graph minus a matching
            for i in range(n):
                for j in range(i+1, n):
                    if (j - i) % 2 == 1:  # Skip some edges
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
        else:
            # Random start with varying density
            density = random.uniform(0.3, 0.7)
            for i, j in all_pairs:
                if random.random() < density:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Check if initial graph is Q3-free, if not, remove problematic edges
        # Actually, let's just proceed with greedy addition from this state
        # First, verify it's Q3-free or clean it up
        
        # Clean up: iteratively remove edges that create Q3
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_mask[u] >> v) & 1:
                        # Temporarily remove and check
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        if has_q3_with_edge(adj_mask, u, v):
                            changed = True
                        else:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
        
        # Now greedily add missing edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_mask[i] >> j) & 1)]
        
        # Shuffle for diversity
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            
            # Try adding
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if has_q3_with_edge(adj_mask, u, v):
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        # Count edges
        ec = count_edges(adj_mask)
        
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj_mask = adj_mask[:]
    
    # Final verification
    if best_adj_mask is None:
        # Fallback: return K3,13
        best_adj_mask = [0] * n
        for u in range(3):
            for v in range(3, 16):
                best_adj_mask[u] |= (1 << v)
                best_adj_mask[v] |= (1 << u)
    
    # Verify no Q3
    if not verify_no_q3(best_adj_mask):
        # Shouldn't happen, but fallback
        best_adj_mask = [0] * n
        for u in range(3):
            for v in range(3, 16):
                best_adj_mask[u] |= (1 << v)
                best_adj_mask[v] |= (1 << u)
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_mask[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
