import random
import time

def _compute_max_book(edges, n=40):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    max_book = 0
    for u, v in edges:
        # Count common neighbors using bit operations
        common = adj[u] & adj[v]
        # Use bin(x).count('1') for portability and speed in Py3
        cnt = bin(common).count('1')
        if cnt > max_book:
            max_book = cnt
    return max_book

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Initial Construction: Random Graph
    # Edge probability p = 401 / (40*39/2) = 401/780 ~ 0.514
    # Random graphs are often closer to optimal book size than structured ones.
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    
    # Initialize with a random set of 401 edges
    initial_edges = random.sample(all_pairs, target_edges)
    
    # Setup bitsets
    adj = [0] * n
    for u, v in initial_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Compute initial max book
    max_book = 0
    for u, v in initial_edges:
        common = adj[u] & adj[v]
        cnt = bin(common).count('1')
        if cnt > max_book:
            max_book = cnt
            
    best_edges = list(initial_edges)
    best_max_book = max_book
    current_max_book = max_book
    
    # 2. Simulated Annealing
    # We will perform edge swaps: remove (u,v), add (x,y)
    # To make this efficient, we need to quickly identify edges and non-edges.
    # We maintain a set of edges for O(1) lookup.
    edge_set = set(initial_edges)
    
    T = 1.0
    cooling_rate = 0.999
    
    # Precompute list of all pairs for random selection
    # We can't just sample from all_pairs every time because we need to ensure one is in edge_set and one is not.
    # Strategy:
    # 1. Pick random edge from edge_set to remove.
    # 2. Pick random non-edge to add.
    # To pick a random non-edge efficiently, we can maintain a list of non-edges or just sample.
    # Given N=40, total pairs 780. Sampling is fast.
    
    while time.time() < deadline:
        # Pick an edge to remove
        u_rem, v_rem = random.choice(list(edge_set))
        
        # Pick a non-edge to add
        added = False
        # Try a few times to find a valid non-edge
        for _ in range(10):
            x, y = random.choice(all_pairs)
            if (x, y) not in edge_set:
                # Perform move
                # Remove (u_rem, v_rem)
                edge_set.remove((u_rem, v_rem))
                adj[u_rem] ^= (1 << v_rem)
                adj[v_rem] ^= (1 << u_rem)
                
                # Add (x, y)
                edge_set.add((x, y))
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                
                # Calculate new max book
                # Optimization: We only need to update max_book if the removed edge was the unique maximum
                # or if the added edge creates a new maximum.
                # For simplicity and correctness, we recompute max_book.
                # With bitsets, this is very fast (~780 operations).
                new_max_book = 0
                for u, v in edge_set:
                    common = adj[u] & adj[v]
                    cnt = bin(common).count('1')
                    if cnt > new_max_book:
                        new_max_book = cnt
                
                delta = new_max_book - current_max_book
                
                if delta < 0 or (T > 0 and random.random() < (2.71828 ** (-delta / T))):
                    # Accept move
                    current_max_book = new_max_book
                    if new_max_book < best_max_book:
                        best_max_book = new_max_book
                        best_edges = list(edge_set)
                else:
                    # Revert
                    edge_set.remove((x, y))
                    adj[x] ^= (1 << y)
                    adj[y] ^= (1 << x)
                    
                    edge_set.add((u_rem, v_rem))
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
                
                added = True
                break
        
        T *= cooling_rate
        
        # If we hit the theoretical lower bound (6), we can stop early?
        # The theorem says >= 40/6 = 6.66, so 7 is the integer lower bound from the theorem?
        # Wait, Edwards and Khadzhiivanov-Nikiforov: score >= n/6. For n=40, 40/6 = 6.66.
        # So max book must be at least 7?
        # "The theorem ... gives score >= 40/6". Since score is integer, score >= 7.
        # However, the prompt says "Minimize; the theorem ... gives score >= 40/6".
        # And "the least achievable value ... is not documented".
        # If the theorem is tight for large n, maybe 7 is the minimum?
        # But wait, if the theorem says >= 6.66, it doesn't rule out 7. It rules out 6.
        # So 7 is the theoretical minimum integer.
        # If we find 7, we can't do better unless the theorem is loose for n=40.
        # Let's keep searching to see if 7 is stable or if we can prove it's minimal.
        # Actually, if 7 is the lower bound, we just need to find ANY graph with max book 7.
        # The current score is 1.5385 which corresponds to value 7 (seed/7 = 1.5385 => seed ~ 10.77? No, score = seed/value. 
        # If score 1.5385 and value 7, seed = 10.769.
        # If we get value 6, score would be 1.79.
        # But if 6 is impossible, we are stuck at 7.
        # Let's assume the goal is to find 7 reliably and quickly.
        
        if best_max_book == 7:
             # We found the theoretical lower bound (integer). 
             # We can return immediately to save time? 
             # Or keep going to ensure we didn't miss a 6?
             # Given the theorem, 6 is impossible. 
             # So returning 7 is optimal.
             break

    # Return best edges found
    return [[u, v] for u, v in best_edges]
