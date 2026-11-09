import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # We aim to find a valid tournament with n < 83.
    # Start with a known valid small n? No, we construct from scratch or optimize.
    # Let's try to construct a valid tournament with n=60 or so using greedy addition.
    # Or better, start with a random tournament of size n and try to fix it?
    # The greedy construction approach:
    # Start with n=5. Check if valid.
    # Add vertices one by one.
    # When adding vertex v, we must decide its orientation against existing vertices.
    # This is hard because the constraint is global on 4-sets.
    
    # Alternative Strategy: Local Search on a fixed n, but smarter initialization.
    # Let's try n=60.
    # If we can find a valid tournament on n=60, score is 60/83 > 1.
    
    # Let's use a simulated annealing / local search approach on the adjacency matrix.
    # Objective: Minimize the number of "bad" 4-sets.
    # A 4-set is bad if no vertex dominates all others in the set.
    # In a tournament on 4 vertices, the score sequences are:
    # (3,0,0,0): Dominated by 1 (Good)
    # (2,1,0,1) -> (2,1,1,0) is NOT dominated by any single vertex?
    # Let's check:
    # Vertices A,B,C,D.
    # If A->B, A->C, A->D, then A dominates.
    # Score sequence (3,0,0,0) -> Good.
    # Score sequence (2,1,1,0):
    #   Let scores be s1=2, s2=1, s3=1, s4=0.
    #   s4=0 means it loses to everyone.
    #   s1=2 means it beats 2, loses to 1.
    #   s2=1 means it beats 1, loses to 2.
    #   s3=1 means it beats 1, loses to 2.
    #   Who dominates? A vertex dominates if it beats all other 3.
    #   Max score is 2. So no vertex beats all 3.
    #   Therefore, any 4-set with score sequence (2,1,1,0) is BAD.
    #   Are there other bad sequences?
    #   (1,1,1,3) is same as (3,0,0,0) reordered.
    #   (2,2,1,1)?
    #   Sum of scores in a tournament on 4 vertices is C(4,2)=6.
    #   Possible partitions of 6 into 4 parts (0<=si<=3, sorted):
    #   3,0,0,3 -> 3,3,0,0? No, max score is 3.
    #   3,0,0,3 is not a tournament score sequence.
    #   Valid score sequences for n=4:
    #   3,0,0,3? No.
    #   Let's list them:
    #   0+0+0+6 (No, max 3)
    #   0+0+1+5 (No)
    #   0+0+2+4 (No)
    #   0+0+3+3 -> Transitive? No. If two have 3, they must beat everyone. But they play each other. One must win. So one has 3, other has 2?
    #   Let's use Landau's Theorem or just enumerate.
    #   Actually, it is known that a tournament is "locally transitive" or similar?
    #   No, the condition is: Every 4-set has a "king" or "dominator".
    #   A vertex v dominates a set S if v -> u for all u in S.
    #   This is equivalent to: The induced subgraph on S has a vertex with out-degree 3 within S.
    #   So we want to minimize the number of 4-sets where the maximum out-degree within the set is < 3.
    #   Max out-degree in a 4-vertex tournament can be 3, 2, or 1 (min is 0? No, min is 0 in transitive, max 3. If max is 2, it's a cycle of 3 with one isolated? No, 4 vertices.
    #   If max out-degree is 2, the scores are likely (2,2,1,1) or (2,1,1,2) -> (2,2,1,1).
    #   Let's check (2,2,1,1):
    #   Sum = 6.
    #   Can a 4-tournament have scores 2,2,1,1?
    #   Yes. A cycle of 4 vertices? 1->2->3->4->1.
    #   Out-degrees: 1:1, 2:1, 3:1, 4:1. Sum 4. No.
    #   In a 4-cycle, each vertex has out-degree 1 within the cycle, but there are 2 chords.
    #   If chords are 1->3 and 2->4.
    #   1 beats 2,4. Loses to 3. Out-degree 2.
    #   2 beats 3,1? No 1->2. 2 beats 3. Loses to 4. Out-degree 1.
    #   3 beats 4,2? No 2->3. 3 beats 4. Loses to 1. Out-degree 1.
    #   4 beats 1,3? No 1->4. 4 beats 1. Loses to 2. Out-degree 1.
    #   Scores: 2,1,1,1? Sum 5. No.
    #   Let's just count violations directly.
    
    n = 60
    # Initialize with a random tournament
    # Use a fast NumPy-based violation counter.
    # Violation = 4-set where no vertex has out-degree 3.
    
    def count_violations(adj):
        # adj is n x n boolean matrix. adj[i,j] = 1 if i->j.
        # For each 4-set {a,b,c,d}, check if max out-degree in induced subgraph is 3.
        # This is O(n^4). For n=60, C(60,4) ~ 487635.
        # 487k * (check 4 vertices) is fast enough in Python? Maybe.
        # But we need to do local search.
        # Let's try to vectorize the check for a specific 4-set?
        # Or just use Python loops for now and see timing.
        
        # Actually, let's use a different metric:
        # Sum over all 4-sets of (1 if valid else 0).
        # We want to maximize valid sets.
        
        violations = 0
        n = len(adj)
        # Precompute out-neighbors?
        # For a set {i,j,k,l}, it is valid if:
        # (adj[i,j] and adj[i,k] and adj[i,l]) OR
        # (adj[j,i] and adj[j,k] and adj[j,l]) OR
        # (adj[k,i] and adj[k,j] and adj[k,l]) OR
        # (adj[l,i] and adj[l,j] and adj[l,k])
        
        # This is still O(n^4).
        # Let's optimize by checking only if necessary.
        
        # Alternative:
        # A 4-set is invalid iff it contains a directed 4-cycle?
        # No.
        # If it's a transitive tournament, max out-degree 3. Valid.
        # If it's a cycle of 3 + 1 vertex connected?
        
        # Let's just implement the O(n^4) check in Python first to see if it's too slow.
        # 487k iterations. Each iteration does ~12 boolean ops.
        # ~6 million ops. Should take < 1 second in PyPy/NumPy? Pure Python maybe 0.5s.
        # If we do 1000 steps, that's 500s. Too slow.
        
        # We need a faster violation count.
        # Let's use NumPy to compute the violation count for the whole graph at once?
        # Hard to vectorize O(n^4).
        
        # Strategy: Local search with a FAST approximation or incremental update?
        # Or just use a smaller n?
        # If we use n=40, C(40,4) = 91,390.
        # 91k * 12 ops ~ 1M ops. 10ms per count.
        # 10,000 steps in 100s.
        
        # Let's try n=50. C(50,4) = 230,300.
        # ~2.7M ops. 30ms per count.
        # 3000 steps in 90s.
        
        # Let's start with n=50.
        n = 50
        rng = np.random.default_rng(42)
        adj = (rng.random((n, n)) < 0.5).astype(np.int8)
        np.fill_diagonal(adj, 0)
        # Ensure antisymmetry
        adj = (adj + adj.T) // 2 # No, this makes it symmetric.
        # Correct:
        upper = np.triu(adj, 1)
        lower = upper.T
        adj = upper + (1 - lower) # This is wrong.
        
        # Proper random tournament:
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 0
                else:
                    adj[i, j] = 0
                    adj[j, i] = 1
                    
        # Faster init:
        rand_mat = rng.random((n, n))
        adj = (rand_mat > 0.5).astype(np.int8)
        np.fill_diagonal(adj, 0)
        # Enforce antisymmetry: if adj[i,j]=1, adj[j,i]=0.
        # We can just take the upper triangle and mirror.
        upper = np.triu(adj, 1)
        lower = (1 - upper.T) # If upper[i,j]=1, lower[j,i]=0.
        adj = upper + lower
        
        def get_violation_count(adj):
            # Use a list of indices for speed?
            # Or just loops.
            v_count = 0
            # Iterate over all 4-combinations
            # Using itertools.combinations is slow.
            # Manual loops are faster in Python?
            # Let's try a hybrid: Precompute out-neighbors as bitsets?
            # n=50 fits in 64 bits.
            # out_neighbors[i] is an integer where bit j is 1 if i->j.
            
            out_neighbors = np.zeros(n, dtype=np.uint64)
            for i in range(n):
                mask = 0
                # Set bits for j where adj[i,j] == 1
                # This loop is O(n^2) per count. 2500 ops. Negligible.
                for j in range(n):
                    if adj[i, j]:
                        mask |= (1 << j)
                out_neighbors[i] = mask
            
            # Now check 4-sets
            # For i<j<k<l:
            # Check if any of i,j,k,l dominates the others.
            # i dominates j,k,l if (out_neighbors[i] has bits j,k,l set)
            
            for i in range(n-3):
                oi = out_neighbors[i]
                for j in range(i+1, n-2):
                    oj = out_neighbors[j]
                    for k in range(j+1, n-1):
                        ok = out_neighbors[k]
                        for l in range(k+1, n):
                            ol = out_neighbors[l]
                            
                            # Check i
                            if (oi & (1<<j)) and (oi & (1<<k)) and (oi & (1<<l)):
                                continue
                            # Check j
                            if (oj & (1<<i)) and (oj & (1<<k)) and (oj & (1<<l)):
                                continue
                            # Check k
                            if (ok & (1<<i)) and (ok & (1<<j)) and (ok & (1<<l)):
                                continue
                            # Check l
                            if (ol & (1<<i)) and (ol & (1<<j)) and (ol & (1<<k)):
                                continue
                            
                            v_count += 1
            return v_count

        # This Python loop for n=50 might be too slow.
        # Let's estimate: 230,300 iterations.
        # Each iteration does ~12 bitwise ops and branches.
        # ~3 million ops.
        # Python does ~10-50 million ops/sec.
        # So ~0.06 - 0.3 seconds per count.
        # If we can do 100 steps in 120s, that's tight.
        
        # Let's try n=40.
        # C(40,4) = 91,390.
        # ~1M ops. 0.02s per count.
        # 5000 steps in 100s.
        
        n = 40
        adj = np.zeros((n, n), dtype=np.int8)
        upper = np.triu((rng.random((n, n)) > 0.5).astype(np.int8), 1)
        adj = upper + (1 - upper.T)
        
        def get_violation_count_fast(adj):
            out_neighbors = np.zeros(n, dtype=np.uint64)
            for i in range(n):
                mask = 0
                # Vectorized bit setting?
                # Just loop, it's small.
                for j in range(n):
                    if adj[i, j]:
                        mask |= (1 << j)
                out_neighbors[i] = mask
            
            v_count = 0
            for i in range(n-3):
                oi = out_neighbors[i]
                for j in range(i+1, n-2):
                    oj = out_neighbors[j]
                    for k in range(j+1, n-1):
                        ok = out_neighbors[k]
                        for l in range(k+1, n):
                            ol = out_neighbors[l]
                            
                            if (oi & (1<<j)) and (oi & (1<<k)) and (oi & (1<<l)):
                                continue
                            if (oj & (1<<i)) and (oj & (1<<k)) and (oj & (1<<l)):
                                continue
                            if (ok & (1<<i)) and (ok & (1<<j)) and (ok & (1<<l)):
                                continue
                            if (ol & (1<<i)) and (ol & (1<<j)) and (ol & (1<<k)):
                                continue
                            
                            v_count += 1
            return v_count

        # Initial count
        current_v = get_violation_count_fast(adj)
        best_adj = adj.copy()
        best_v = current_v
        
        # Local Search
        # Randomly flip an edge (i,j).
        # Calculate delta in violations.
        # This is hard to calculate incrementally without O(n^2) work per flip?
        # Actually, flipping (i,j) only affects 4-sets containing i and j.
        # Number of such sets is C(n-2, 2) = (n-2)(n-3)/2.
        # For n=40, 38*37/2 = 703 sets.
        # Checking 703 sets per flip is fast.
        
        def calc_delta(adj, i, j):
            # Flip i->j to j->i (or vice versa)
            # Current state: adj[i,j]
            # New state: 1 - adj[i,j]
            
            # We need to check all pairs (k,l) such that {i,j,k,l} is a 4-set.
            # k,l must be distinct from i,j and each other.
            
            delta = 0
            # Precompute out_neighbors for current adj?
            # We can update out_neighbors incrementally in the main loop,
            # but for delta calculation, we need to know the state before/after.
            
            # Let's just re-evaluate the 703 sets using the current adj,
            # and the hypothetical adj (with i,j flipped).
            
            # To make it faster, let's use the bitset representation.
            # But we need to pass adj or bitsets.
            
            # Let's optimize:
            # The 4-set {i,j,k,l} is valid if any vertex dominates.
            # Flipping i,j only changes the out-degrees of i and j within the set.
            # It does not change the dominance status of k or l *relative to the set*,
            # EXCEPT if k or l dominated i or j, or vice versa.
            # Wait, "dominates the set" means dominating ALL other 3 vertices.
            # If k dominated the set before, it beat i, j, l.
            # After flip, if i->j becomes j->i:
            # k still beats i, j, l. So k still dominates.
            # So if k or l dominated before, they still dominate after?
            # Yes, because the flip is between i and j. k's edges to i,j,l are unchanged.
            # So the ONLY sets that can change validity are those where:
            # 1. They were valid before, and become invalid.
            #    This can only happen if the dominator was i or j.
            # 2. They were invalid before, and become valid.
            #    This can only happen if the new dominator is i or j.
            
            # So we only need to check if i or j was/is the dominator.
            
            # Case 1: i dominates {i,j,k,l} before.
            # This requires i->j, i->k, i->l.
            # After flip (j->i), i no longer dominates.
            # Does the set become invalid?
            # Only if no other vertex dominates.
            # So we check if j, k, or l dominates after flip.
            
            # Case 2: j dominates {i,j,k,l} before.
            # This requires j->i, j->k, j->l.
            # After flip (i->j), j no longer dominates.
            # Check if i, k, or l dominates after flip.
            
            # Case 3: Set was invalid before.
            # Check if i or j dominates after flip.
            
            # This logic is complex to implement correctly and fast.
            # Given time constraints, let's just re-evaluate the 703 sets fully.
            # 703 sets * 12 ops = 8k ops.
            # 5000 steps * 8k ops = 40M ops.
            # 40M ops in Python is ~4-8 seconds? No, 10M ops/sec is optimistic for Python.
            # Maybe 1-2M ops/sec.
            # 40M / 1M = 40 seconds.
            # This fits in the budget!
            
            # Let's implement this.
            
            # We need the out_neighbors of the current graph.
            # We will maintain out_neighbors in the main loop.
            
            pass # Placeholder

        # Let's refine the main loop to maintain out_neighbors.
        out_neighbors = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            out_neighbors[i] = mask

        def is_valid_set(i, j, k, l, out_neigh):
            # Check if any of i,j,k,l dominates the others
            oi = out_neigh[i]
            oj = out_neigh[j]
            ok = out_neigh[k]
            ol = out_neigh[l]
            
            if (oi & (1<<j)) and (oi & (1<<k)) and (oi & (1<<l)): return True
            if (oj & (1<<i)) and (oj & (1<<k)) and (oj & (1<<l)): return True
            if (ok & (1<<i)) and (ok & (1<<j)) and (ok & (1<<l)): return True
            if (ol & (1<<i)) and (ol & (1<<j)) and (ol & (1<<k)): return True
            return False

        def get_delta(adj, out_neigh, i, j):
            # Flip edge i-j
            # i->j becomes j->i (or vice versa)
            
            # Identify all k, l such that k,l != i,j and k<l
            # We iterate over all pairs (k,l)
            
            delta = 0
            # We need to check the validity of {i,j,k,l} before and after.
            
            # To do this efficiently, we can just call is_valid_set
            # with the current out_neigh, and a modified out_neigh.
            
            # But constructing a modified out_neigh for every flip is expensive.
            # Instead, we can pass the "virtual" state.
            
            # Let's just re-evaluate using the bit masks directly in a helper.
            
            # For each pair (k, l) with k,l not in {i,j}:
            for k in range(n):
                if k == i or k == j: continue
                for l in range(k+1, n):
                    if l == i or l == j: continue
                    if k > l: continue # Ensure unique pair
                    
                    # Actually, we just need to iterate over all distinct k,l from the remaining n-2 vertices.
                    # This is C(n-2, 2) pairs.
                    
                    # Check current validity
                    # We need out_neigh for i,j,k,l.
                    # Current:
                    valid_before = is_valid_set(i, j, k, l, out_neigh)
                    
                    # Hypothetical: flip i,j.
                    # This changes out_neigh[i] and out_neigh[j].
                    # new_oi = out_neigh[i] with bit j flipped.
                    # new_oj = out_neigh[j] with bit i flipped.
                    # ok, ol unchanged.
                    
                    # We can create temporary masks.
                    oi = out_neigh[i]
                    oj = out_neigh[j]
                    ok = out_neigh[k]
                    ol = out_neigh[l]
                    
                    # Flip bit j in oi, bit i in oj
                    new_oi = oi ^ (1 << j)
                    new_oj = oj ^ (1 << i)
                    
                    # Check new validity
                    # We can inline the check for speed
                    valid_after = False
                    # i dominates?
                    if (new_oi & (1<<j)) and (new_oi & (1<<k)) and (new_oi & (1<<l)):
                        valid_after = True
                    else:
                        # j dominates?
                        if (new_oj & (1<<i)) and (new_oj & (1<<k)) and (new_oj & (1<<l)):
                            valid_after = True
                        else:
                            # k dominates?
                            if (ok & (1<<i)) and (ok & (1<<j)) and (ok & (1<<l)):
                                valid_after = True
                            else:
                                # l dominates?
                                if (ol & (1<<i)) and (ol & (1<<j)) and (ol & (1<<k)):
                                    valid_after = True
                    
                    if valid_before and not valid_after:
                        delta += 1
                    elif not valid_before and valid_after:
                        delta -= 1
                        
            return delta

        # Main Loop
        steps = 0
        while time.time() < deadline:
            # Pick random i, j
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j: continue
            
            d = get_delta(adj, out_neighbors, i, j)
            
            # Accept if delta < 0 (improvement)
            # Or with some probability if delta > 0 (Simulated Annealing)
            # For now, greedy (Hill Climbing)
            if d < 0:
                # Apply flip
                adj[i, j] = 1 - adj[i, j]
                adj[j, i] = 1 - adj[j, i]
                
                # Update out_neighbors
                out_neighbors[i] ^= (1 << j)
                out_neighbors[j] ^= (1 << i)
                
                current_v += d
                if current_v < best_v:
                    best_v = current_v
                    best_adj = adj.copy()
                
                steps += 1
                if best_v == 0:
                    break
            
            # Optional: Random restarts if stuck?
            # For now, just keep going.

        # Return the best result
        if best_v == 0:
             # Convert best_adj to orientation list
             orientation = []
             for i in range(n):
                 for j in range(i+1, n):
                     orientation.append(int(best_adj[i, j]))
             return {'n': n, 'orientation': orientation}
        else:
             # Return the seed result if we didn't improve
             p = 83
             squares = {x * x % p for x in range(1, p)}
             return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}

