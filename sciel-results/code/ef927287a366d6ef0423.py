import time
import random
import itertools
from collections import defaultdict

def popcount(x):
    return bin(x).count("1")

def construct():
    start_time = time.time()
    timeout = 110  # 110 seconds limit to be safe
    
    # Try to construct a graph with n=20
    # If that fails, try n=19, etc.
    for n in [20, 19, 18]:
        if time.time() - start_time > timeout - 5:
            break
            
        # Initialize random graph
        # neigh[v] is an integer bitmask of neighbors of v
        neigh = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    neigh[i] |= (1 << j)
                    neigh[j] |= (1 << i)
                    edges.append([i, j])
        
        # Precompute all subsets of size 6
        # And for each pair (u, v), the list of subsets containing them
        subsets_6 = list(itertools.combinations(range(n), 6))
        subset_masks = [0] * len(subsets_6)
        for idx, s in enumerate(subsets_6):
            mask = 0
            for v in s:
                mask |= (1 << v)
            subset_masks[idx] = mask
            
        # Map pair (u,v) to list of subset indices
        pair_to_subsets = defaultdict(list)
        for idx, s in enumerate(subsets_6):
            for u in s:
                for v in s:
                    if u < v:
                        pair_to_subsets[(u, v)].append(idx)
        
        def is_regular(mask, verts):
            # verts is a tuple of vertices in the subset
            degs = set()
            for v in verts:
                d = popcount(neigh[v] & mask)
                degs.add(d)
                if len(degs) > 1:
                    return False
            return True
        
        def count_bad_6():
            count = 0
            for idx, s in enumerate(subsets_6):
                if is_regular(subset_masks[idx], s):
                    count += 1
            return count
        
        # Initial count
        bad_6 = count_bad_6()
        
        # Local Search
        # We want to minimize bad_6.
        # If bad_6 == 0, check larger sizes.
        
        best_bad_6 = bad_6
        best_neigh = neigh[:]
        best_edges = edges[:]
        
        # Simple Hill Climbing with restarts
        restarts = 0
        max_restarts = 50
        
        while time.time() - start_time < timeout - 2:
            if bad_6 == 0:
                # Check larger sizes
                valid = True
                # Check sizes 7 to n
                for k in range(7, n + 1):
                    if time.time() - start_time > timeout - 2:
                        break
                    subsets_k = list(itertools.combinations(range(n), k))
                    for s in subsets_k:
                        mask = 0
                        for v in s:
                            mask |= (1 << v)
                        if is_regular(mask, s):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return {'n': n, 'edges': best_edges}
                
                # If not valid, we need to break the large regular subgraphs.
                # This is harder. Let's try to minimize bad_6 + bad_7... 
                # For now, if we find bad_6=0 but larger ones exist, we might just return 
                # the best n we've found so far if we can't fix it quickly.
                # Or we can try to flip edges to break the specific large subgraph found.
                # Given complexity, let's just continue searching for a "better" graph 
                # by minimizing bad_6 primarily, hoping for one that has no large ones.
                # Or we can accept this if it's the best we can do? 
                # No, we must return a VALID graph.
                # If we can't make it valid, we should try smaller n or different search.
                
            # Move: Flip a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Calculate delta for bad_6
            # Only subsets containing u and v are affected
            affected_indices = pair_to_subsets.get((min(u, v), max(u, v)), [])
            
            delta = 0
            # Check old state
            old_bad_count = 0
            for idx in affected_indices:
                if is_regular(subset_masks[idx], subsets_6[idx]):
                    old_bad_count += 1
            
            # Flip edge
            if neigh[u] & (1 << v):
                neigh[u] ^= (1 << v)
                neigh[v] ^= (1 << u)
            else:
                neigh[u] |= (1 << v)
                neigh[v] |= (1 << u)
            
            # Check new state
            new_bad_count = 0
            for idx in affected_indices:
                if is_regular(subset_masks[idx], subsets_6[idx]):
                    new_bad_count += 1
            
            delta = new_bad_count - old_bad_count
            
            # Acceptance rule: Hill Climbing (accept if delta < 0)
            # Or Simulated Annealing? 
            # Let's use Hill Climbing first.
            if delta < 0:
                bad_6 += delta
                # Update edges list
                if neigh[u] & (1 << v):
                    edges.append([u, v])
                else:
                    edges.remove([u, v]) if [u, v] in edges else edges.remove([v, u])
                    
                if bad_6 < best_bad_6:
                    best_bad_6 = bad_6
                    best_neigh = neigh[:]
                    best_edges = edges[:]
            else:
                # Revert
                if neigh[u] & (1 << v):
                    neigh[u] ^= (1 << v)
                    neigh[v] ^= (1 << u)
                else:
                    neigh[u] |= (1 << v)
                    neigh[v] |= (1 << u)
                
                # If stuck (no improvement in many steps), restart
                # Simple counter
                # We don't have a counter here, let's just do random restarts periodically
                # Or just continue.
                
            # Periodic restart if stuck
            # Check time
            if random.random() < 0.01: # 1% chance to restart
                restarts += 1
                if restarts > max_restarts:
                    break
                # Reinitialize
                neigh = [0] * n
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            neigh[i] |= (1 << j)
                            neigh[j] |= (1 << i)
                            edges.append([i, j])
                bad_6 = count_bad_6()
                
        # If we exhausted time for this n, try next n (smaller)
        # But we must return SOMETHING valid.
        # The seed returned n=5. We can always return n=5 if we fail.
        # But we want to maximize n.
        
    # Fallback: Return the best valid graph found?
    # We didn't track "valid" graphs explicitly in the loop above, only best_bad_6.
    # We need to ensure we return a VALID graph.
    # Let's do a final check on best_neigh/best_edges if best_bad_6 == 0.
    
    # Actually, the loop above returns immediately if it finds a valid graph.
    # If it doesn't, it falls through.
    # We should store the best candidate that had bad_6=0 and check it.
    
    # Let's restructure slightly to keep track of the best "bad_6=0" candidate.
    # For simplicity, if the loop finishes without returning, we return n=5.
    # But we can do better:
    # Keep a list of candidates with bad_6=0.
    
    # Given the complexity, let's just return n=5 if we fail to find a valid n>5.
    # But we expect to find n=20.
    
    return {'n': 5, 'edges': []}

# Wait, the logic above has a flaw: if it finds bad_6=0 but larger regular subgraphs exist, it doesn't return, 
# but it also doesn't save that candidate to try to fix later. 
# And if it fails, it returns n=5.
# We should try to fix the larger subgraphs.
# However, fixing larger subgraphs is computationally expensive.
# Given the 120s limit, finding a graph with NO regular induced subgraphs of size 6 is the main task.
# It is highly likely that such a graph also has no regular induced subgraphs of size > 6, 
# or we can find one that does by chance.
# Let's assume the search for bad_6=0 will yield a valid graph.

# Refined Plan:
# 1. Search for n=20, minimize bad_6.
# 2. If bad_6 becomes 0, verify full validity.
# 3. If valid, return.
# 4. If not valid, continue search (maybe flip edges to break the specific large subgraph).
# 5. If time runs out, return n=5.

# Let's implement the "fix large subgraph" part simply:
# If we find a large regular subgraph S, flip a random edge inside S or incident to S?
# Flipping an edge inside S might break regularity.
# We can just continue the local search, but now penalize bad_k for k>6.
# To keep it simple, we will just rely on the random search finding a "clean" graph.

# One more optimization: 
# The `edges.remove` is O(N). 
# We can maintain the edge set as a set of tuples for O(1) add/remove.
# But we need to return list of lists.

# Let's rewrite construct with better state management.

def construct():
    start_time = time.time()
    timeout = 115
    
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
        if time.time() - start_time > timeout - 10:
            break
            
        # Init
        neigh = [0] * n
        edge_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    neigh[i] |= (1 << j)
                    neigh[j] |= (1 << i)
                    edge_set.add((i, j))
        
        # Precompute subsets 6
        subsets_6 = list(itertools.combinations(range(n), 6))
        subset_masks = [0] * len(subsets_6)
        for idx, s in enumerate(subsets_6):
            mask = 0
            for v in s:
                mask |= (1 << v)
            subset_masks[idx] = mask
            
        pair_to_subsets = defaultdict(list)
        for idx, s in enumerate(subsets_6):
            for u in s:
                for v in s:
                    if u < v:
                        pair_to_subsets[(u, v)].append(idx)
        
        def is_regular(mask, verts):
            degs = set()
            for v in verts:
                d = popcount(neigh[v] & mask)
                degs.add(d)
                if len(degs) > 1:
                    return False
            return True
        
        def count_bad_6():
            count = 0
            for idx, s in enumerate(subsets_6):
                if is_regular(subset_masks[idx], s):
                    count += 1
            return count
        
        def is_valid_graph():
            # Check all subsets size >= 6
            for k in range(6, n + 1):
                for s in itertools.combinations(range(n), k):
                    mask = 0
                    for v in s:
                        mask |= (1 << v)
                    if is_regular(mask, s):
                        return False
            return True
        
        bad_6 = count_bad_6()
        
        # Local Search
        steps = 0
        max_steps = 50000 # per n
        
        while steps < max_steps and time.time() - start_time < timeout - 2:
            if bad_6 == 0:
                if is_valid_graph():
                    # Convert edge_set to list of lists
                    edges = [[u, v] for u, v in edge_set]
                    return {'n': n, 'edges': edges}
                # If not valid, we might try to break the large subgraph
                # For now, just continue search, hoping to find a different bad_6=0 graph
                # Or we can force a move that breaks the specific regular subgraph
                # Let's just continue.
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            key = (min(u, v), max(u, v))
            affected = pair_to_subsets.get(key, [])
            
            old_bad = 0
            for idx in affected:
                if is_regular(subset_masks[idx], subsets_6[idx]):
                    old_bad += 1
            
            # Flip
            if key in edge_set:
                neigh[u] ^= (1 << v)
                neigh[v] ^= (1 << u)
                edge_set.remove(key)
            else:
                neigh[u] |= (1 << v)
                neigh[v] |= (1 << u)
                edge_set.add(key)
            
            new_bad = 0
            for idx in affected:
                if is_regular(subset_masks[idx], subsets_6[idx]):
                    new_bad += 1
            
            delta = new_bad - old_bad
            
            # Accept if improvement, or with small probability if neutral (to escape local minima)
            if delta < 0 or (delta == 0 and random.random() < 0.1):
                bad_6 += delta
            else:
                # Revert
                if key in edge_set:
                    neigh[u] ^= (1 << v)
                    neigh[v] ^= (1 << u)
                    edge_set.remove(key)
                else:
                    neigh[u] |= (1 << v)
                    neigh[v] |= (1 << u)
                    edge_set.add(key)
            
            steps += 1
            
            # Restart if stuck
            if steps % 1000 == 0 and bad_6 > 0:
                # Simple restart
                neigh = [0] * n
                edge_set = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            neigh[i] |= (1 << j)
                            neigh[j] |= (1 << i)
                            edge_set.add((i, j))
                bad_6 = count_bad_6()
                steps = 0

    # Fallback
    return {'n': 5, 'edges': []}
