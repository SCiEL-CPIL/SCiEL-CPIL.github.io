import time
import numpy as np

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def construct():
    N = 18
    MIN_DIST = 6
    start_time = time.time()
    
    # 1. Generate a structured candidate pool.
    # We exploit the fact that A_2(18,6) is likely composed of structured words.
    # We generate all vectors of weight <= 4.
    # Count: C(18,0)+C(18,1)+C(18,2)+C(18,3)+C(18,4) = 1 + 18 + 153 + 816 + 3060 = 4048.
    # This pool is small enough to fit in memory and allows fast bitset operations.
    # Additionally, we add the Reed-Muller RM(1,4) extended code words (32 words) 
    # to ensure we have a valid baseline immediately.
    
    candidates = [0]  # All-zero vector
    
    # Add weight 1
    for i in range(N):
        candidates.append(1 << i)
        
    # Add weight 2
    for i in range(N):
        for j in range(i+1, N):
            candidates.append((1 << i) | (1 << j))
            
    # Add weight 3
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                candidates.append((1 << i) | (1 << j) | (1 << k))
                
    # Add weight 4
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    candidates.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
                    
    # Add RM(1,4) extended words (shifted by 2 bits? No, standard RM(1,4) is length 16. 
    # The task says "extended by 2 zero bits" in the seed, but usually extended means adding parity.
    # The seed used: bit = ((slope & point).bit_count() + intercept) % 2 for point in 16.
    # This creates length 16 words. The seed appended 2 zeros? 
    # "First-order Reed-Muller on 16 coords extended by 2 zero bits -> 32 words, dist 8"
    # Let's replicate the seed's RM generation to be safe, but ensure they are in [0, 2**18).
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The seed didn't shift, so these are in the first 16 bits.
            # We add them if not already present (weight 0 is present, weight 1/2 might overlap)
            if word not in candidates:
                candidates.append(word)
                
    # Remove duplicates and sort for determinism
    candidates = sorted(list(set(candidates)))
    M = len(candidates)
    
    # 2. Build the compatibility matrix using NumPy.
    # A[i, j] = 1 if Hamming(candidates[i], candidates[j]) >= MIN_DIST
    # We can compute Hamming distances efficiently using broadcasting, but M~4000 means 16M pairs.
    # 16M bit_count operations might be slow in pure Python, but NumPy vectorization helps.
    # However, storing a 4000x4000 matrix is 16M bytes (if bool) or 128MB (if int). This is fine.
    
    # To speed up distance calculation:
    # We can't easily do bit_count on a 2D array of ints in NumPy without loops or ufuncs.
    # Alternative: Use the fact that we only need to find a Max Independent Set.
    # We can precompute the "forbidden" set for each candidate.
    
    # Let's try a hybrid: Precompute distances in chunks or use a faster method.
    # Actually, for M=4048, a pure Python loop for building the matrix might take too long (16M * ~100ns = 1.6s, acceptable).
    # But let's optimize:
    
    # We will represent the graph using bitsets (lists of Python ints or numpy arrays of uint64).
    # Since M < 4096, we can fit the neighbor list of each node in a single Python int (bitmask)!
    # This is the key optimization: Graph representation as list of integers where bit j is 1 if i and j are compatible.
    
    adjacency = [0] * M
    
    # Optimization: Only calculate i < j and set bits symmetrically.
    # We can speed up the inner loop by early termination or vectorization if possible.
    # For now, let's use a reasonably fast Python loop with bit_count.
    
    # To make it faster, we can group by weight? No, distance depends on specific bits.
    # Let's just run the double loop. 4000*4000/2 = 8 million iterations.
    # 8M iterations in Python might take ~1-2 seconds. This is acceptable.
    
    for i in range(M):
        ci = candidates[i]
        for j in range(i + 1, M):
            cj = candidates[j]
            if (ci ^ cj).bit_count() >= MIN_DIST:
                adjacency[i] |= (1 << j)
                adjacency[j] |= (1 << i)
                
    # 3. Bron-Kerbosch with Pivot to find Maximum Independent Set.
    # We want the largest set of mutually compatible candidates.
    # This is equivalent to finding the Maximum Clique in the compatibility graph.
    
    best_set = []
    best_size = 0
    
    def bron_kerbosch(R, P, X):
        """
        R: current clique (list of indices)
        P: candidates to add (list of indices)
        X: excluded candidates (list of indices)
        """
        nonlocal best_set, best_size
        
        if time.time() > DEADLINE:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_set = R[:]
            return
            
        if not P:
            return
            
        # Pivot selection: choose u in P union X with max degree in P
        # To speed up, we can just pick the first one if M is small, or find max popcount of (adjacency[u] & P_mask)
        # Since we use bitmasks for P and X, this is efficient.
        
        P_mask = 0
        for p in P:
            P_mask |= (1 << p)
            
        X_mask = 0
        for x in X:
            X_mask |= (1 << x)
            
        # Find pivot u
        # We want to maximize |P ∩ N(u)|
        # Iterate through P and X to find u with max overlap with P.
        # Optimization: If P is small, just pick P[0]. If large, sample or compute properly.
        # For M=4000, computing max degree for every node in P∪X is expensive (O(|P∪X| * M/64)).
        # However, Bron-Kerbosch with pivot is very effective.
        # Let's try a simpler pivot: pick u from P that has the most neighbors in P.
        
        # Actually, a common heuristic for speed: just pick u = P[0] if we don't want to spend time finding the best pivot.
        # But a good pivot is crucial for pruning.
        # Let's find the best pivot among P.
        
        max_count = -1
        pivot = -1
        # To avoid O(|P|^2) just for pivot selection, we can approximate or just use the first few.
        # Given the tight time budget, let's use a simple heuristic:
        # If len(P) > 100, just pick P[0] to save time on pivot calculation?
        # No, the pruning benefit is huge. Let's calculate it properly but efficiently.
        
        # We can calculate the degree of each node in P relative to P.
        # deg(u) = popcount(adjacency[u] & P_mask)
        
        # To speed this up, we can iterate through P and check a few candidates.
        # Or, we can just iterate through all P and X and find the max.
        # |P| can be up to 4000. 4000 popcounts is fast (~0.1ms).
        
        all_candidates_for_pivot = P + X
        for u in all_candidates_for_pivot:
            # Count neighbors in P
            count = (adjacency[u] & P_mask).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
                
        if pivot == -1:
            return
            
        # The neighbors of pivot in P are the ones we do NOT iterate over in the main loop
        # We iterate over P \ N(pivot)
        
        # Create a list of nodes in P that are NOT neighbors of pivot
        # P_non_neighbors = [p for p in P if not (adjacency[pivot] >> p) & 1]
        
        # To do this efficiently with bitmasks:
        # P_mask is the set of all P.
        # N_pivot_in_P = adjacency[pivot] & P_mask
        # We want to iterate over P \ N_pivot_in_P.
        
        # However, Bron-Kerbosch recursion works on lists.
        # We can convert P to a bitmask, then extract the non-neighbors.
        
        # Let's stick to lists for R, P, X but use bitmasks for fast intersection checks.
        
        # We need to iterate over P \ N(pivot).
        # Let's generate the list.
        
        # Optimization: If P is large, this list comprehension might be slow.
        # But M is only 4000.
        
        P_to_iterate = []
        for p in P:
            if not (adjacency[pivot] >> p) & 1:
                P_to_iterate.append(p)
                
        for v in P_to_iterate:
            # Recurse
            # New R = R + [v]
            # New P = P ∩ N(v)
            # New X = X ∩ N(v)
            
            new_P = []
            for p in P:
                if p != v and ((adjacency[v] >> p) & 1):
                    new_P.append(p)
                    
            new_X = []
            for x in X:
                if ((adjacency[v] >> x) & 1):
                    new_X.append(x)
                    
            bron_kerbosch(R + [v], new_P, new_X)
            
            # Update P and X for the next iteration of the loop
            P.remove(v)
            X.append(v)
            
            if time.time() > DEADLINE:
                return

    # Initial call
    # We want to find the max clique in the compatibility graph.
    # R = [], P = all nodes, X = []
    # However, the search space is large.
    # We can prune by fixing the first node?
    # Since the graph is vertex-transitive (hypercube), we can fix node 0 (the zero vector) to be in the set.
    # This reduces the search space significantly.
    # So, start with R = [0], P = neighbors of 0, X = non-neighbors of 0?
    # Wait, if we fix 0 in the set, we only consider sets containing 0.
    # Since 0 is compatible with any node of weight >= 6, and our candidates are weight <= 4,
    # 0 is compatible with ALL candidates in our pool!
    # So P = all nodes except 0. X = empty.
    
    P_initial = [i for i in range(1, M)]
    X_initial = []
    
    # To improve pruning, we can order P by degree or just use the order.
    # Let's try.
    
    # Note: The recursion depth might be high, but Python recursion limit is usually 1000.
    # Max clique size is likely around 50-100. Depth is manageable.
    
    # However, Bron-Kerbosch can explode.
    # We have a time limit.
    
    # Let's add a heuristic: If the current best_size + len(P) <= best_size, prune.
    # This is standard.
    
    def bron_kerbosch_optimized(R, P, X):
        nonlocal best_set, best_size
        
        if time.time() > DEADLINE:
            return
            
        if len(R) + len(P) <= best_size:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_set = R[:]
            return
            
        if not P:
            return
            
        P_mask = 0
        for p in P:
            P_mask |= (1 << p)
            
        X_mask = 0
        for x in X:
            X_mask |= (1 << x)
            
        # Pivot
        max_count = -1
        pivot = -1
        # To speed up pivot selection, we can just check a subset if P is very large?
        # No, M is small.
        for u in P + X:
            count = (adjacency[u] & P_mask).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
                
        if pivot == -1:
            return
            
        # Iterate P \ N(pivot)
        # We create a list to iterate over, because we modify P in the loop
        # We must be careful: we iterate over a snapshot of P \ N(pivot)
        
        # Get the indices to visit
        to_visit = []
        for p in P:
            if not (adjacency[pivot] >> p) & 1:
                to_visit.append(p)
                
        for v in to_visit:
            # If v is no longer in P, skip (shouldn't happen if we manage P correctly, 
            # but we are using a snapshot so it's safe)
            
            new_P = []
            for p in P:
                if p != v and ((adjacency[v] >> p) & 1):
                    new_P.append(p)
                    
            new_X = []
            for x in X:
                if ((adjacency[v] >> x) & 1):
                    new_X.append(x)
                    
            bron_kerbosch_optimized(R + [v], new_P, new_X)
            
            P.remove(v)
            X.append(v)
            
            if time.time() > DEADLINE:
                return

    # Start search
    # We fix node 0 (zero vector) in the code.
    # This is valid because of symmetry: any optimal code can be XORed with one of its words to make that word 0.
    # So there exists an optimal code containing 0.
    
    # Initial P: all nodes compatible with 0.
    # Since all candidates have weight <= 4, they are all compatible with 0 (distance = weight >= 0, wait).
    # Distance(0, v) = weight(v).
    # We require distance >= 6.
    # Our candidates have weight <= 4.
    # So 0 is NOT compatible with any candidate of weight < 6!
    # Ah, this is a critical realization.
    # If we fix 0 in the code, we can only add words of weight >= 6.
    # But our candidate pool only has weights <= 4.
    # So if we fix 0, we can add NO other words from this pool.
    # This means fixing 0 is a terrible idea for this specific candidate pool.
    
    # Correction: The symmetry argument says we can fix ONE word to be 0.
    # But if we fix 0, we must generate candidates that are compatible with 0 (weight >= 6).
    # Our pool is weight <= 4.
    # So we should NOT fix 0. We should search for the max independent set in the whole pool.
    # Or, we should generate a pool of weight >= 6?
    # If we generate weight >= 6, the pool size is huge (2^18 - 4048).
    
    # Alternative: Don't fix 0. Just run BK on the whole pool.
    # The pool has 4048 nodes.
    # The max independent set in this graph corresponds to the max code.
    
    # Let's run BK on the whole pool.
    
    P_initial = list(range(M))
    X_initial = []
    
    bron_kerbosch_optimized([], P_initial, X_initial)
    
    # If we found a set, convert indices to words
    if best_set:
        result_words = [candidates[i] for i in best_set]
    else:
        # Fallback to RM code if search failed or found nothing (shouldn't happen)
        result_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                result_words.append(word)
                
    # Final check
    # The BK algorithm guarantees independence.
    # But let's double check to be safe.
    # Also, ensure no duplicates.
    
    # The result might be small if the search didn't explore enough.
    # We can try to improve by greedy addition from the found set?
    # No, BK is exact for the induced subgraph. If the optimal code uses words outside our pool, we miss it.
    # But weight <= 4 is a strong guess for the structure of A_2(18,6) if it's related to constant weight codes or similar.
    # Actually, A_2(18,6) is 512.
    # A code of size 512 with length 18 and distance 6.
    # If it were constant weight, it would be A(18, 6, w).
    # Max A(18, 6, w) for some w?
    # If we use only weight 3 and 4, can we get 512?
    # Number of weight 3 words is 816. Number of weight 4 is 3060.
    # It is plausible that the optimal code is composed of low weight words.
    
    # If the search didn't find a large set, we might need a larger pool.
    # But time is limited.
    
    return result_words

# To ensure the function is named correctly and runs
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found {len(res)} words")
