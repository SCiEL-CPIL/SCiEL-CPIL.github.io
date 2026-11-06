"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
import math

def _has_violation_fast(adj, n):
    """
    Check if ANY induced subgraph of size >= 6 is regular.
    Returns (True, subset) if found, else (False, None).
    Optimized for n <= 22.
    """
    # Precompute popcount for masks? No, int.bit_count() is fast in Python 3.8+.
    
    # We check k from 6 to n.
    # To save time, we can stop early if we find a violation.
    # Heuristic: Smaller k are more likely to be violated in random graphs? 
    # Actually, regular subgraphs are rare in random graphs for large k?
    # No, for k=6, the number of possible degree sequences is small.
    # Let's just iterate k from 6 to n.
    
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Calculate degrees
            d0 = -1
            is_reg = True
            for v in subset:
                d = (adj[v] & mask).bit_count()
                if d0 == -1:
                    d0 = d
                elif d != d0:
                    is_reg = False
                    break
            
            if is_reg:
                return True, subset
    return False, None

def _generate_1_factorization(n):
    """
    Generate a random 1-factorization of K_n.
    Returns a list of n-1 matchings, where each matching is a list of (u, v) pairs.
    """
    if n % 2 != 0:
        # Add a dummy vertex to make it even, factorize, then remove dummy edges
        # But the task says n <= 22, and we want n=20 or 21.
        # If n is odd, we can't have a perfect matching.
        # The directive says "1-factorization of K_22". 
        # If we target n=21, we can't use perfect matchings directly on 21 vertices.
        # We can use a 1-factorization of K_22, remove vertex 21 and its incident edges.
        # This results in a graph on 21 vertices which is a union of near-perfect matchings.
        
        # For simplicity in this block, we will assume we generate for even N, 
        # and if target n is odd, we generate for n+1 and drop one vertex.
        pass
        
    # Standard algorithm for 1-factorization of K_{2m}
    # Vertices 0..2m-1
    # Fix vertex 0.
    # The other vertices are arranged in a circle.
    
    # Actually, a simpler random method:
    # 1. Create list of all edges.
    # 2. Randomly pick a perfect matching.
    # 3. Remove those edges.
    # 4. Repeat.
    # This is $O(n^3)$ or so, which is fine for n=22.
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    matchings = []
    remaining_edges = set(edges)
    
    for _ in range(n - 1):
        # Pick a random perfect matching from remaining_edges
        # Greedy: pick random edge, then complete matching? 
        # This might fail if we pick edges that isolate vertices.
        # Better: Use a randomized algorithm for perfect matching.
        # Or just: 
        # 1. Shuffle remaining edges.
        # 2. Iterate through shuffled edges, if both endpoints are free, add to matching.
        # This does NOT guarantee a perfect matching (it guarantees a maximal matching, which might not be perfect).
        
        # Correct approach:
        # We need a uniform random perfect matching.
        # For small n, we can use backtracking or just pick random permutations.
        
        # Let's use the "circle method" which is deterministic but we can randomize the initial permutation.
        # This generates a specific 1-factorization.
        # To get random, we can randomly permute vertices first.
        
        # However, we want *different* factorizations or just one?
        # The strategy is: Generate ONE random 1-factorization, then select a random subset of its matchings.
        
        # Let's implement the circle method for K_n (n even).
        # Vertices: 0, 1, ..., n-1.
        # Fix 0.
        # Others: 1, ..., n-1 in a circle.
        # Step i: 
        #   Edge (0, i)
        #   Edge (i+1, i-1), (i+2, i-2), ... mod (n-1) for the others.
        
        # This produces n-1 matchings.
        
        # To randomize, we shuffle the vertices 1..n-1.
        perm = list(range(1, n))
        random.shuffle(perm)
        
        # Now perm[k] is the vertex at position k in the circle.
        # Circle indices 0..n-2 correspond to vertices perm[0]..perm[n-2].
        
        for i in range(n - 1):
            matching = []
            # Edge with 0
            # Vertex at position i is perm[i].
            # Wait, the standard construction:
            # v_0 = 0
            # v_1, ..., v_{n-1} are the others.
            # Matching i:
            # (0, v_i)
            # (v_{i+1}, v_{i-1}), (v_{i+2}, v_{i-2}), ...
            # Indices mod (n-1).
            
            v_i = perm[i % (n-1)]
            matching.append((0, v_i))
            
            for j in range(1, (n-1)//2 + 1):
                # j steps away from i
                idx1 = (i + j) % (n-1)
                idx2 = (i - j) % (n-1)
                u = perm[idx1]
                v = perm[idx2]
                matching.append((u, v))
            
            matchings.append(matching)
            
    return matchings

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Targets: 21, 20, 19
    # Note: n=21 is odd. We will generate for n=22 and drop one vertex.
    # n=20 is even.
    
    target_ns = [21, 20, 19, 18]
    
    for n_target in target_ns:
        if time.time() > deadline - 30:
            break
            
        # Determine generation size
        # If n_target is odd, generate for n_target + 1 (even) and drop 1 vertex.
        # If n_target is even, generate for n_target.
        n_gen = n_target if n_target % 2 == 0 else n_target + 1
        
        max_restarts = 100
        if n_target == 21:
            max_restarts = 200 # More effort for the harder target
            
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # 1. Generate 1-factorization
            # We need a valid 1-factorization for n_gen.
            # If n_gen > 22, we can't. But max is 22.
            # If n_target=21, n_gen=22.
            
            try:
                matchings = _generate_1_factorization(n_gen)
            except Exception:
                continue
                
            # 2. Select a random subset of matchings
            # Probability of including each matching?
            # If we include all, we get K_n (or K_{n+1} minus vertex). K_n has many regular subgraphs.
            # If we include few, we get sparse graphs.
            # We want a "middle ground".
            # Let's try random subsets of size k, where k is between n/4 and n/2.
            
            num_matchings = len(matchings)
            # k from 5 to 10 for n=20 (19 matchings available)
            # k from 5 to 11 for n=22 (21 matchings available)
            
            min_k = max(2, num_matchings // 4)
            max_k = min(num_matchings - 1, num_matchings // 2 + 2)
            
            k_sel = random.randint(min_k, max_k)
            selected_indices = random.sample(range(num_matchings), k_sel)
            
            # Build adjacency
            adj = [0] * n_gen
            for idx in selected_indices:
                for u, v in matchings[idx]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # 3. If n_target < n_gen, drop a random vertex
            if n_gen > n_target:
                # Drop vertex n_gen - 1 (or random)
                drop_v = random.randint(0, n_gen - 1)
                # Remap vertices
                new_adj = [0] * n_target
                mapping = {}
                new_idx = 0
                for v in range(n_gen):
                    if v == drop_v:
                        continue
                    mapping[v] = new_idx
                    new_idx += 1
                
                for u in range(n_gen):
                    if u == drop_v:
                        continue
                    for v in range(n_gen):
                        if v == drop_v:
                            continue
                        if adj[u] & (1 << v):
                            new_adj[mapping[u]] |= (1 << mapping[v])
                            new_adj[mapping[v]] |= (1 << mapping[u])
                
                adj = new_adj
                n = n_target
            else:
                n = n_gen
            
            # 4. Check for violations
            has_v, viol_subset = _has_violation_fast(adj, n)
            
            if not has_v:
                # Valid!
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    # If we found n=21, we can stop immediately? 
                    # The task says "Maximize n". 21 is the max possible improvement over 20?
                    # "Graphs with 20 vertices are known; a 21-vertex graph would show G(6) >= 22."
                    # So 21 is the holy grail.
                    if n == 21:
                        return {'n': best_n, 'edges': best_edges}
                    # If we found 20, we can keep looking for 21, but 20 is a good fallback.
                    # Let's continue to see if we can find 21 in remaining time.
                
                # If we found a valid graph, we can break restarts for this n if we want to save time?
                # No, we want the MAX n. So if n_target=20 and we found one, we should switch to n_target=21.
                # The outer loop handles n_target descending? No, ascending?
                # target_ns = [21, 20, 19, 18].
                # So we try 21 first. If we find 21, we return.
                # If we fail 21, we try 20.
                # So we don't need to break restarts if we found a valid 20, because we are already in the 20 loop.
                # But if we found 20, we should stop searching for 20 and move to 19? No, 20 is better than 19.
                # So if we find a valid graph for current n_target, we should break the restart loop and let the outer loop continue to the next (smaller) n?
                # Wait, the outer loop goes 21 -> 20 -> 19.
                # If we are in n_target=21 and fail, we go to 20.
                # If we are in n_target=20 and succeed, we have best_n=20.
                # We should then break the n_target=20 loop and go to 19? No, 19 is worse.
                # So if we find a valid graph for n_target, we should break the restart loop and BREAK the outer loop?
                # No, because we might find a 21 later? No, we already tried 21.
                # So if we succeed at n_target, we can break the outer loop entirely?
                # Yes, because we are iterating from highest to lowest.
                
                # However, we might want to try more restarts for n_target to see if we can find a "better" graph? 
                # The score is just n. So any valid graph on n is equal.
                # So we can break.
                
                break # Break restart loop
                
            # If has_v, we could try to repair?
            # For now, just restart.
            
        # If we broke the restart loop due to success, we should break the outer loop?
        # Only if we found the maximum possible n?
        # No, if we found n=20, we can stop trying n=19.
        # So:
        if best_n == n_target:
             # We found a valid graph for this n.
             # Since we are iterating from high to low, this is the best we can get?
             # Wait, if n_target=21 and we found 21, we return.
             # If n_target=20 and we found 20, we should stop trying 19.
             # So we break the outer loop.
             break
             
    # Fallback: if we didn't find anything good, return best found (likely 5 or 0)
    # But the random 1-factorization approach should find valid graphs for n=20 easily.
    # If not, we return the best we have.
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
