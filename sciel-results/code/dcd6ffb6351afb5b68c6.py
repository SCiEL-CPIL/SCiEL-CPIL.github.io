import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a valid graph on 35 vertices
    N = 35
    # We can also try 36, but 35 is the known record. Let's stick to 35 to be safe, 
    # or try 36 briefly? No, 36 is likely impossible. 
    # Actually, if we find one on 35, score is 35/17 = 2.05.
    
    # Initial seed: Random graph
    # We can use multiple restarts
    
    best_graph = None
    best_violations = float('inf')
    
    # Helper to check violations and repair
    # We will run a local search
    
    for restart in range(20): # Max 20 restarts
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # Use adjacency list or bitmask
        # Bitmask is faster for small N
        adj = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local Search
        # We want to eliminate K4s and Ind6s.
        # Strategy:
        # 1. Count K4s and Ind6s.
        # 2. If 0, return.
        # 3. Pick a random K4, remove an edge.
        # 4. Pick a random Ind6, add an edge.
        # Alternate or randomize.
        
        current_violations = float('inf')
        for step in range(1000): # Max steps per restart
            if time.time() > deadline:
                break
                
            # Find K4s
            # A K4 is a set of 4 vertices {a,b,c,d} such that all 6 edges exist.
            # Efficiently finding all K4s in a 35-vertex graph is doable.
            # We can iterate over pairs of edges (a,b) and (c,d) sharing a vertex?
            # Or iterate over triangles and check the 4th vertex.
            
            # Let's find one K4 and one Ind6.
            # Finding a K4:
            k4_found = False
            ind6_found = False
            k4_edges = None
            ind6_pair = None
            
            # Find K4
            # Iterate over all pairs of vertices (u, v)
            # For each pair, find common neighbors.
            # If two common neighbors x, y are connected, {u,v,x,y} is a K4.
            # This is O(N^2 * N^2) worst case, but with bitmasks it's fast.
            
            found_k4 = False
            for u in range(N):
                for v in range(u+1, N):
                    if not (adj[u] & (1 << v)):
                        continue
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    # Remove u and v from common
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Iterate over pairs in common
                    # If x, y are in common and connected, we have K4 {u,v,x,y}
                    # We just need one K4.
                    # Get list of vertices in common
                    # Since N=35, common has at most 33 bits.
                    # We can extract bits.
                    bits = []
                    temp = common
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        temp ^= lsb
                    
                    for i in range(len(bits)):
                        x = bits[i]
                        for j in range(i+1, len(bits)):
                            y = bits[j]
                            if adj[x] & (1 << y):
                                # Found K4: u, v, x, y
                                # Remove a random edge from this K4
                                # Edges: (u,v), (u,x), (u,y), (v,x), (v,y), (x,y)
                                edge_to_remove = random.choice([(u,v), (u,x), (u,y), (v,x), (v,y), (x,y)])
                                a, b = edge_to_remove
                                adj[a] &= ~(1 << b)
                                adj[b] &= ~(1 << a)
                                found_k4 = True
                                break
                        if found_k4:
                            break
                    if found_k4:
                        break
                if found_k4:
                    break
            
            if found_k4:
                continue # Loop again to check for more violations or ind6s
            
            # Find Ind6
            # An independent set of 6 is a set of 6 vertices with no edges between them.
            # This is equivalent to a K6 in the complement graph.
            # Complement adjacency: comp_adj[u] = ((1<<N)-1) ^ (1<<u) ^ adj[u]
            
            # We can use the same K4 finding logic on the complement?
            # No, we need K6 in complement.
            # Finding a K6 in a 35-vertex graph is harder than K4.
            # However, we can use a greedy approach or backtracking.
            # Or, we can just check if the current graph has an independent set of size 6.
            # If it does, we need to add an edge.
            
            # Efficient check for Independent Set of size k:
            # Use Bron-Kerbosch or similar on the complement?
            # For k=6, N=35, it's manageable.
            
            # Let's try to find an independent set of size 6.
            # We can use a recursive search.
            # Or, simpler: iterate over all 6-subsets? C(35,6) = 1.6 million.
            # 1.6 million checks per step is too slow in Python if done naively.
            # But we can prune.
            
            # Alternative: Use the fact that we just removed an edge or added one.
            # If we didn't find a K4, we should look for Ind6.
            
            # Let's implement a fast check for Ind6.
            # We can use a bitset based approach.
            # Start with all vertices.
            # Pick a vertex v.
            # The Ind6 must be either in {v} U Ind5(N[v]) or Ind6(N - {v} - N[v])? No.
            # Standard MIS algorithm.
            
            # Given the time constraint, maybe we should prioritize K4 removal first,
            # as K4s are easier to find and remove.
            # If we eliminate all K4s, we are left with a K4-free graph.
            # Then we check for Ind6.
            
            # Let's restructure the loop:
            # 1. Find and remove K4s until none left.
            # 2. Check for Ind6. If found, add an edge.
            # 3. Repeat.
            
            # This might oscillate.
            # Better: Randomly choose to fix a K4 or an Ind6.
            
            # For now, let's just do the K4 removal loop as above.
            # If no K4 is found, we need to find an Ind6.
            
            # Finding Ind6:
            # We can use a simple backtracking with pruning.
            # Or, we can just use a precomputed list of all 6-subsets? No.
            
            # Let's try a heuristic:
            # If no K4 found, pick a random pair of non-adjacent vertices and add an edge?
            # No, that might create K4s.
            # We should specifically target Ind6s.
            
            # Let's implement a function to find an independent set of size 6.
            # If found, add a random edge within it.
            
            # To speed up, we can cache the complement?
            
            # Let's try a different approach:
            # Use a known valid graph if possible?
            # I don't have it.
            
            # Let's stick to the local search.
            # The key is to make the Ind6 detection fast.
            # Since N=35, we can represent the graph as a list of bitmasks.
            # To find an independent set of size 6:
            # We can use the following logic:
            # An independent set is a clique in the complement.
            # We can run a clique finder on the complement graph.
            # Bron-Kerbosch with pivot is efficient for small N.
            
            # However, implementing a fast Bron-Kerbosch in Python within the loop might be slow.
            # Let's try a simpler approach:
            # Check if the graph has an independent set of size 6 by trying to extend small independent sets.
            
            # For the purpose of this rewrite, I will implement a robust local search that:
            # 1. Tries to find a valid graph on 35 vertices.
            # 2. If it fails, it tries 34, then 33, etc.
            # 3. It uses a time limit.
            
            # To ensure we don't timeout, I will wrap the search in a tight loop with time checks.
            
            pass # Placeholder for the actual loop logic below
