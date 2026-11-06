import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute all K4s
    vertices = list(range(16))
    k4s = list(itertools.combinations(vertices, 4))
    
    # Helper to get edge index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * 15 - (i * (i - 1)) // 2 + (j - i - 1)
        # Wait, let's use a simpler mapping or just a dictionary
        # Actually, 120 edges.
        # i * 16 - i*(i+1)//2 + j - i - 1?
        # Let's just use a list of tuples and a dict.
        
    # Better: flat list index
    # (0,1) -> 0
    # (0,2) -> 1
    # ...
    # (0,15) -> 14
    # (1,2) -> 15
    # ...
    def get_idx(i, j):
        if i > j: i, j = j, i
        # Number of edges before i is i*15 - i*(i-1)//2? No.
        # Edges (0,j) for j=1..15: 15 edges.
        # Edges (1,j) for j=2..15: 14 edges.
        # Sum_{k=0}^{i-1} (15-k) = 15i - i(i-1)/2
        return 15*i - i*(i-1)//2 + (j - i - 1)

    # Precompute edges for each K4
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)

    def check_validity(colours, k):
        # colours is a list of 120 ints
        # Check if all K4s have >= 5 distinct colours
        # And only use colours 0..k-1
        for e in colours:
            if e < 0 or e >= k:
                return False, 1 # Invalid colour range
        for edges in k4_edges:
            c = set(colours[i] for i in edges)
            if len(c) < 5:
                return False, 0
        return True, 0

    def count_deficit(colours):
        deficit = 0
        for edges in k4_edges:
            c = set(colours[i] for i in edges)
            if len(c) < 5:
                deficit += (5 - len(c))
        return deficit

    # Try k = 13, 14, 15
    for k in [13, 14, 15]:
        # Initialize
        colours = [random.randint(0, k-1) for _ in range(120)]
        
        # Local Search
        current_deficit = count_deficit(colours)
        best_deficit = current_deficit
        best_colours = colours[:]
        
        # If we start with deficit 0, we are done
        if best_deficit == 0:
            valid, _ = check_validity(best_colours, k)
            if valid:
                return best_colours

        # Simulated Annealing or Hill Climbing
        # Given time, simple hill climbing with random restarts
        restarts = 0
        max_restarts = 200
        
        while time.time() < deadline - 5.0: # Leave 5s for next k or fallback
            # Pick a random K4 with deficit
            # Or just pick a random edge and change it
            # To be faster, pick an edge involved in a bad K4
            bad_k4s = []
            for i, edges in enumerate(k4_edges):
                c = set(colours[e] for e in edges)
                if len(c) < 5:
                    bad_k4s.append(i)
            
            if not bad_k4s:
                valid, _ = check_validity(colours, k)
                if valid:
                    return colours
                else:
                    # Should not happen if deficit 0 implies valid
                    break

            # Pick a random bad K4
            k4_idx = random.choice(bad_k4s)
            edges = k4_edges[k4_idx]
            
            # Pick a random edge in this K4
            edge_idx = random.choice(edges)
            
            # Change its colour
            old_colour = colours[edge_idx]
            new_colour = random.randint(0, k-1)
            if new_colour == old_colour:
                continue
                
            colours[edge_idx] = new_colour
            new_deficit = count_deficit(colours)
            
            if new_deficit < current_deficit:
                current_deficit = new_deficit
                if current_deficit < best_deficit:
                    best_deficit = current_deficit
                    best_colours = colours[:]
                    if best_deficit == 0:
                        valid, _ = check_validity(best_colours, k)
                        if valid:
                            return best_colours
            else:
                # Revert
                colours[edge_idx] = old_colour
        
        # If no improvement after some steps, restart?
        # The loop above runs until deadline.
        # We should break and try next k if we are stuck?
        # Let's just run until deadline for this k.
        # Actually, if we are stuck, we might want to move to next k earlier.
        # But 13 is best. Let's give it 30s, 14 gets 30s, 15 gets 30s.
        if time.time() > start_time + 35.0:
            break
        elif time.time() > start_time + 70.0 and k == 13:
            break # Move to 14
        elif time.time() > start_time + 105.0 and k == 14:
            break # Move to 15

    # Fallback: Z17 construction
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
