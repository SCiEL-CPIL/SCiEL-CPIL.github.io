import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Q3 adjacency: 
    # Vertices 0..7
    # 0: 000, 1: 001, 2: 010, 3: 011, 4: 100, 5: 101, 6: 110, 7: 111
    # Edges:
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    q3_adj = [[0]*8 for _ in range(8)]
    for u, v in q3_edges:
        q3_adj[u][v] = 1
        q3_adj[v][u] = 1

    def has_q3_in_subset(subset, adj):
        # subset is a tuple of 8 vertex indices from 0..15
        # adj is the 16x16 adjacency matrix of the current graph
        # Check if the induced subgraph on 'subset' contains Q3
        # 1. Degree check
        # Map subset vertices to 0..7 for local indexing
        # local_adj[i][j] = adj[subset[i]][subset[j]]
        
        # Quick check: count edges in induced subgraph
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]][subset[j]]:
                    edge_count += 1
        if edge_count < 12:
            return False
            
        # Degree check: all 8 vertices must have degree >= 3 in induced subgraph
        for i in range(8):
            deg = 0
            for j in range(8):
                if i != j and adj[subset[i]][subset[j]]:
                    deg += 1
            if deg < 3:
                return False
                
        # Backtracking search for isomorphism
        # Fix Q3 vertex 0 to subset vertex 0
        # Q3 neighbors of 0 are 1, 2, 4
        # Subset neighbors of subset[0]
        neighbors_of_0 = [j for j in range(8) if j != 0 and adj[subset[0]][subset[j]]]
        
        if len(neighbors_of_0) < 3:
            return False
            
        # Try all combinations of 3 neighbors for Q3 vertices 1, 2, 4
        for combo in itertools.combinations(neighbors_of_0, 3):
            # combo[0] -> Q3 1, combo[1] -> Q3 2, combo[2] -> Q3 4
            # Remaining Q3 vertices: 3, 5, 6, 7
            # Remaining subset vertices: 8 - 1 (for 0) - 3 (for 1,2,4) = 4
            remaining_subset = [j for j in range(8) if j not in [0] and j not in combo]
            
            # Q3 3 is connected to 1, 2, 7
            # Q3 5 is connected to 1, 4, 7
            # Q3 6 is connected to 2, 4, 7
            # Q3 7 is connected to 3, 5, 6
            
            # We need to assign remaining_subset to Q3 vertices 3, 5, 6, 7
            # Let's try permutations of remaining_subset for 3, 5, 6, 7
            for perm in itertools.permutations(remaining_subset):
                v3, v5, v6, v7 = perm
                
                # Check edges
                # 3 connected to 1, 2, 7
                if not (adj[subset[combo[0]]][subset[v3]] and 
                        adj[subset[combo[1]]][subset[v3]] and 
                        adj[subset[v3]][subset[v7]]):
                    continue
                # 5 connected to 1, 4, 7
                if not (adj[subset[combo[0]]][subset[v5]] and 
                        adj[subset[combo[2]]][subset[v5]] and 
                        adj[subset[v5]][subset[v7]]):
                    continue
                # 6 connected to 2, 4, 7
                if not (adj[subset[combo[1]]][subset[v6]] and 
                        adj[subset[combo[2]]][subset[v6]] and 
                        adj[subset[v6]][subset[v7]]):
                    continue
                # 7 connected to 3, 5, 6 (already checked 3-7, 5-7, 6-7 above? 
                # 3-7 checked in 3's block. 5-7 in 5's block. 6-7 in 6's block.
                # We also need to check 3-5? No, 3 and 5 are not adjacent in Q3.
                # 3-6? No.
                # 5-6? No.
                # So we just need to ensure the required edges exist.
                
                # Wait, did I miss any edges?
                # Q3 edges:
                # 0-1, 0-2, 0-4 (checked by combo selection)
                # 1-3, 1-5 (checked in 3 and 5 blocks)
                # 2-3, 2-6 (checked in 3 and 6 blocks)
                # 4-5, 4-6 (checked in 5 and 6 blocks)
                # 3-7, 5-7, 6-7 (checked)
                
                # If all these are 1, then we have a Q3 subgraph.
                return True
                
        return False

    def graph_has_q3(adj):
        # Check all 8-vertex subsets
        for subset in itertools.combinations(range(16), 8):
            if has_q3_in_subset(subset, adj):
                return True
        return False

    def count_q3s(adj):
        count = 0
        for subset in itertools.combinations(range(16), 8):
            if has_q3_in_subset(subset, adj):
                count += 1
        return count

    # Initial graph: Random dense
    best_adj = [[0]*16 for _ in range(16)]
    best_edges = 0
    
    # Try a few random starts
    for _ in range(5):
        if time.time() > deadline:
            break
            
        adj = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                if random.random() < 0.6:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
        # Remove Q3s
        # While there is a Q3, remove an edge from one
        # To be efficient, find one Q3 and remove a random edge from it
        # We can reuse has_q3_in_subset to find one? 
        # Let's write a function to find one Q3
        def find_one_q3(adj):
            for subset in itertools.combinations(range(16), 8):
                if has_q3_in_subset(subset, adj):
                    return subset
            return None

        while time.time() < deadline:
            subset = find_one_q3(adj)
            if subset is None:
                break
            
            # Pick a random edge in this subset to remove
            # We need the actual vertices
            # The subset is a tuple of 8 vertices
            # We need to pick an edge (u,v) from the Q3 edges that exists in adj
            # We can just pick a random pair from subset that is an edge in adj
            # But it must be part of the Q3 structure?
            # Removing any edge from the subset that is part of the Q3 copy will break that specific copy.
            # However, we don't know which edges form the Q3 copy easily from just the subset.
            # But has_q3_in_subset returns True if ANY Q3 is present.
            # If we remove a random edge from the induced subgraph, it might not break the Q3.
            # Better: modify find_one_q3 to return the edges of the found Q3.
            
            # Let's refactor has_q3_in_subset to return the mapping or edges if found.
            
            # For now, let's just remove a random edge from the complete graph on the subset?
            # No, that's risky.
            
            # Let's implement a function to return the edges of a Q3 if found.
            pass # Placeholder
            
            # Simpler approach: 
            # If has_q3, pick a random edge (u,v) from the whole graph and remove it?
            # No, that might take many iterations.
            
            # Let's stick to the subset.
            # If subset has a Q3, removing any edge from the subset that is in the Q3 will break it.
            # Since we don't know which edges are in the Q3, we can try removing each edge in the subset until the Q3 is gone?
            # Or just remove a random edge from the subset.
            
            # Let's pick a random edge from the induced subgraph on subset
            edges_in_subset = []
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[subset[i]][subset[j]]:
                        edges_in_subset.append((subset[i], subset[j]))
                        
            if edges_in_subset:
                u, v = random.choice(edges_in_subset)
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                break # Should not happen if has_q3 is true

        # Now adj is Q3-free
        # Try to add edges
        # Iterate through all missing edges
        # If adding edge doesn't create Q3, add it
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(16):
                for j in range(i+1, 16):
                    if adj[i][j] == 0:
                        # Try adding
                        adj[i][j] = 1
                        adj[j][i] = 1
                        if not graph_has_q3(adj):
                            improved = True
                        else:
                            adj[i][j] = 0
                            adj[j][i] = 0
                            
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [row[:] for row in adj]
            
    # Convert best_adj to list of edges
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if best_adj[i][j]:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
