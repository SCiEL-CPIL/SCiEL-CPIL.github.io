import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    # Prioritize 46 (likely) and 47 (stretch).
    # Then fallbacks.
    target_ns = [46, 47, 45, 44, 43, 42, 41, 40]
    
    # Time allocation: 
    # Give 20s to 46, 20s to 47, 10s to others?
    # Or just iterate and check time.
    
    for n in target_ns:
        if time.time() > deadline - 10: # Keep 10s buffer
            break
            
        target_degree = n - 39
        
        # Run restarts for this n
        while time.time() < deadline - 10:
            neighbors = [set() for _ in range(n)]
            edges = []
            
            stuck = False
            # Random seed? No, random module is fine.
            
            # Greedy with min-degree priority
            while True:
                # Check success
                # Optimization: if min degree >= target, break
                # Using min() is O(n).
                if min(len(nei) for nei in neighbors) >= target_degree:
                    break
                
                # Find u with min degree
                u = min(range(n), key=lambda i: len(neighbors[i]))
                
                # If u is already at target, min() would have caught it?
                # If all are at target, min >= target, loop breaks.
                # So deg(u) < target_degree.
                
                # Find v
                indices = list(range(n))
                random.shuffle(indices)
                
                found_v = -1
                for v in indices:
                    if v == u:
                        continue
                    if v in neighbors[u]:
                        continue
                    if neighbors[u].isdisjoint(neighbors[v]):
                        found_v = v
                        break
                        
                if found_v == -1:
                    stuck = True
                    break
                
                neighbors[u].add(found_v)
                neighbors[found_v].add(u)
                edges.append([u, found_v])
            
            if not stuck:
                # We exited the while True loop via the break in the success check
                # Or did we? 
                # The inner while True breaks when min_deg >= target.
                # If it breaks due to success, stuck is False.
                # If it breaks due to no candidate, stuck is True.
                # But what if it runs forever? 
                # We should add a step limit.
                pass

            # Actually, the inner loop logic above doesn't have a step limit.
            # It will terminate when success or stuck.
            # Success is guaranteed to terminate.
            # Stuck is guaranteed to terminate (no candidates).
            # So the inner loop terminates.
            
            if not stuck:
                best_n = n
                best_edges = edges
                # If we found a solution for a higher n, we might want to stop?
                # The list is ordered 46, 47...
                # If we found 46, we move to 47.
                # If we found 47, we move to 45 (but we already have 47).
                # We should probably stop if we find 47?
                # Or just continue to use remaining time for other 47 attempts?
                # No, once we have a valid 47, that's the max possible (given 47 is max n).
                # So if n==47 and success, we can break everything.
                if n == 47:
                    break
                # If n==46, we continue to try 47.
                # But we should probably break the restart loop for 46 to save time for 47?
                # Yes.
                break

        if n == 47:
            break

    return {'n': best_n, 'edges': best_edges}
