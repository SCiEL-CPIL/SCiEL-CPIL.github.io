import time
import random
from itertools import combinations

def construct():
        start_time = time.time()
        deadline = start_time + 290  # 10s buffer
        
        # 1. Generate Paley(37)
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        # adj[i] is a set of neighbors or a bitset
        # Using list of sets for simplicity, or integers for speed
        # n=37, so integers (Python big ints) are fine.
        adj = [0] * q
        for i in range(q):
            for j in range(i+1, q):
                if (j - i) % q in residues:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        
        n = q
        
        # Helper to check K4 in mask
        def has_k4(adj, mask):
            # Get list of vertices in mask
            verts = [i for i in range(len(adj)) if (mask >> i) & 1]
            # If len < 4, return False
            if len(verts) < 4: return False
            # Check all 4-combinations
            for combo in combinations(verts, 4):
                # Check if all 6 edges exist
                # Edge (u,v) exists if (adj[u] >> v) & 1
                if ((adj[combo[0]] >> combo[1]) & 1) and \
                   ((adj[combo[0]] >> combo[2]) & 1) and \
                   ((adj[combo[0]] >> combo[3]) & 1) and \
                   ((adj[combo[1]] >> combo[2]) & 1) and \
                   ((adj[combo[1]] >> combo[3]) & 1) and \
                   ((adj[combo[2]] >> combo[3]) & 1):
                    return True
            return False

        # Helper to check Independent 4 in mask (no edges)
        def has_ind_4(adj, mask):
            verts = [i for i in range(len(adj)) if (mask >> i) & 1]
            if len(verts) < 4: return False
            for combo in combinations(verts, 4):
                if not ((adj[combo[0]] >> combo[1]) & 1) and \
                   not ((adj[combo[0]] >> combo[2]) & 1) and \
                   not ((adj[combo[0]] >> combo[3]) & 1) and \
                   not ((adj[combo[1]] >> combo[2]) & 1) and \
                   not ((adj[combo[1]] >> combo[3]) & 1) and \
                   not ((adj[combo[2]] >> combo[3]) & 1):
                    return True
            return False

        # Function to try to add a vertex
        def try_add_vertex(adj, n):
            # We want to find x in {0,1}^n such that
            # has_k4(adj_extended, mask_neighbors) is False
            # has_ind_4(adj_extended, mask_nonneighbors) is False
            
            # Note: When checking K4 in N(v), the K4 must be formed by vertices in N(v).
            # The edges between those vertices are defined by the existing adj.
            # So we just need to check if the subgraph induced by N(v) in the EXISTING graph has a K4.
            # Similarly for Ind 4 in N(v)^c.
            
            # Initial guess: Random
            best_x = None
            best_score = float('inf')
            
            # Simulated Annealing / Local Search
            x = random.getrandbits(n)
            # Convert to int mask
            # We need to evaluate cost
            def cost(x_mask):
                # Neighbors mask
                # x_mask bit i is 1 if v is connected to i
                # We need to check K4 in the subgraph of G induced by x_mask
                # and Ind4 in the subgraph of G induced by (~x_mask & ((1<<n)-1))
                
                # Optimization: If we flip one bit, we can update cost incrementally?
                # For simplicity, recalculate. n is small.
                
                # Check K4 in neighbors
                # Extract vertices
                # This is slow. 
                # Let's precompute K4s in the graph? 
                # No, the set of neighbors changes.
                
                # Let's just run the check.
                # To speed up, we can break early.
                
                # Count K4s? Or just boolean?
                # Boolean is enough for validity.
                # For search, we need a gradient. 
                # Let's count the number of K4s and Ind4s.
                
                k4_count = 0
                ind4_count = 0
                
                verts = [i for i in range(n) if (x_mask >> i) & 1]
                if len(verts) >= 4:
                    for combo in combinations(verts, 4):
                        # Check K4
                        if ((adj[combo[0]] >> combo[1]) & 1) and \
                           ((adj[combo[0]] >> combo[2]) & 1) and \
                           ((adj[combo[0]] >> combo[3]) & 1) and \
                           ((adj[combo[1]] >> combo[2]) & 1) and \
                           ((adj[combo[1]] >> combo[3]) & 1) and \
                           ((adj[combo[2]] >> combo[3]) & 1):
                            k4_count += 1
                            
                non_verts = [i for i in range(n) if not ((x_mask >> i) & 1)]
                if len(non_verts) >= 4:
                    for combo in combinations(non_verts, 4):
                        if not ((adj[combo[0]] >> combo[1]) & 1) and \
                           not ((adj[combo[0]] >> combo[2]) & 1) and \
                           not ((adj[combo[0]] >> combo[3]) & 1) and \
                           not ((adj[combo[1]] >> combo[2]) & 1) and \
                           not ((adj[combo[1]] >> combo[3]) & 1) and \
                           not ((adj[combo[2]] >> combo[3]) & 1):
                            ind4_count += 1
                            
                return k4_count + ind4_count

            current_cost = cost(x)
            if current_cost == 0:
                return x
                
            # Local Search
            for _ in range(1000): # Fixed number of steps per try
                if time.time() > deadline:
                    return None
                # Pick random bit to flip
                i = random.randint(0, n-1)
                old_bit = (x >> i) & 1
                x ^= (1 << i)
                new_cost = cost(x)
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        return x
                else:
                    # Revert
                    x ^= (1 << i)
            return None

        # Iterate adding vertices
        for target_n in range(n + 1, 43): # Try to reach 42
            if time.time() > deadline:
                break
            added = False
            # Try multiple times to add a vertex
            for attempt in range(50):
                if time.time() > deadline:
                    break
                x = try_add_vertex(adj, n)
                if x is not None:
                    # Add vertex n
                    # Update adj
                    adj.append(0)
                    for i in range(n):
                        if (x >> i) & 1:
                            adj[i] |= (1 << n)
                            adj[n] |= (1 << i)
                    n += 1
                    added = True
                    break
            if not added:
                # If we fail to add, we might try to restart with a different seed or break
                # For now, break
                break
                
        # Convert to edges
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
                    
        return n, edges
