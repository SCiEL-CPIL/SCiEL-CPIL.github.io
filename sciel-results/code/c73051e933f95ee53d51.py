"""Ternary covering code via Minimum Dominating Set with vectorized dominance pruning."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    
    Strategy:
    1. Precompute the adjacency matrix A where A[i, j] = 1 if dist(i, j) <= radius.
       This matrix represents the graph where we seek a Minimum Dominating Set.
    2. Use a randomized greedy set-cover to get an initial valid dominating set.
    3. Apply iterative "dominance pruning":
       A codeword c can be removed if for all words w it covers, there exists another codeword c' != c 
       that also covers w. In terms of the dominating set S, this means the "private" part of c's 
       neighborhood (N(c) \ union_{c' in S\{c} N(c')) is empty.
       We vectorize the check for redundancy using matrix operations on the current set of codewords.
    4. To avoid getting stuck, we perform multiple restarts with different random seeds/orderings.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = q ** n
    if N <= 0:
        return []
    
    # Precompute all words as integer indices 0..N-1
    # We need the actual values for the final output
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q
            
    # Precompute adjacency matrix (Dominating Set graph)
    # A[i, j] = 1 if dist(i, j) <= radius
    # For q=3, n=7, radius=1, this is a sparse-ish matrix but dense enough for NumPy bool ops.
    # N=2187, so 2187x2187 bool matrix is ~4.7MB. Very fast to operate on.
    
    if radius == 1:
        # Efficient generation for radius 1
        A = np.zeros((N, N), dtype=bool)
        # Diagonal is always 1
        np.fill_diagonal(A, True)
        
        # For each word, find neighbors
        # We can vectorize this somewhat, but simple loop is okay for N=2187
        # Actually, let's just compute it directly using the digit representation logic
        # but stored in the matrix.
        
        # Faster: generate all pairs? No, N^2 is 4.7M. 
        # Let's use the digit difference method.
        # For each position p and change delta, update the matrix.
        # This is complex to vectorize perfectly for arbitrary q, n.
        # Given N is small, O(N^2) with a fast inner loop or even O(N^2 * n) in Python is ~30s.
        # We need it faster.
        
        # Alternative: Use the property that dist <= 1 means they are equal or differ in exactly 1 pos.
        # Let's just compute the matrix using NumPy broadcasting on the all_words array.
        # all_words is (N, n).
        # diff = all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :]  # (N, N, n)
        # dist = diff.sum(axis=2)  # (N, N)
        # A = dist <= radius
        
        # Memory: 2187 * 2187 * 7 * 1 byte (bool) = ~33 MB. Acceptable.
        # Time: Broadcasting is fast.
        
        # Let's do it.
        # To save memory, we can compute dist in chunks or just let NumPy handle it.
        # For N=2187, this is fine.
        
        # Optimization: We only need A for the local search.
        # Let's compute A.
        
        # Note: all_words[i] is the word for index i.
        # dist(i, j) is number of positions where all_words[i] != all_words[j].
        
        # We'll compute A in a memory-efficient way if possible, but standard broadcasting is likely fine for 2187.
        
        # Let's try the broadcasting approach.
        # It creates an intermediate array of size N*N*n.
        # 2187^2 * 7 ~ 33 million booleans. ~33 MB.
        # This is fast in NumPy.
        
        # However, creating the full (N, N, n) array might be slow to allocate.
        # Let's do it in a loop over positions to keep memory lower and potentially faster due to cache?
        # Or just do it all at once. Let's try all at once first, it's the most "NumPy" way.
        
        # Actually, for radius 1, we can be smarter.
        # A[i, j] is True if i==j OR (i, j differ in exactly 1 position).
        
        # Let's stick to the simple, robust method for general radius if needed, but optimized for radius 1.
        
        # Method:
        # 1. Initialize A to False.
        # 2. Set diagonal to True.
        # 3. For each position p in 0..n-1:
        #    For each pair of values a, b in 0..q-1 where a != b:
        #       Find all words with value a at p.
        #       Find all words with value b at p.
        #       The cross product of these two sets are neighbors.
        #       Set A[set_a, set_b] = True and A[set_b, set_a] = True.
        
        # This is O(n * q^2 * (N/q)^2) operations roughly, but vectorized.
        # N/q is ~729. 729^2 is ~530k. n*q^2 is 7*9=63. 63 * 530k ~ 33M ops.
        # This is very fast in NumPy.
        
        A = np.zeros((N, N), dtype=bool)
        np.fill_diagonal(A, True)
        
        # Precompute indices for each position and value
        # indices[p][v] = array of indices i where all_words[i, p] == v
        indices = [[np.where(all_words[:, p] == v)[0] for v in range(q)] for p in range(n)]
        
        for p in range(n):
            for v1 in range(q):
                set1 = indices[p][v1]
                if len(set1) == 0: continue
                for v2 in range(q):
                    if v1 == v2: continue
                    set2 = indices[p][v2]
                    if len(set2) == 0: continue
                    # All pairs (i, j) with i in set1, j in set2 are at distance 1 (differing at p)
                    # Note: They might differ elsewhere too? No, if they differ at p, they are neighbors.
                    # Wait, "neighbors" in the graph means dist <= 1.
                    # If they differ at p, dist is at least 1.
                    # If they differ ONLY at p, dist is 1.
                    # If they differ at p AND elsewhere, dist > 1.
                    # So we cannot just mark all cross products as True.
                    # We need to check if they differ ONLY at p.
                    
                    # This breaks the simple vectorization.
                    # We must go back to the definition: dist(i, j) <= 1.
                    
                    # Let's revert to the broadcasting method for correctness and simplicity, 
                    # as N is small enough.
        
        # Let's do the broadcasting method properly.
        # It is O(N^2 * n) time and O(N^2 * n) space (temporarily).
        # 33MB is fine.
        
        # To avoid the huge temporary array, we can compute distance in chunks or use a loop over positions accumulating distance.
        dist = np.zeros((N, N), dtype=np.int8)
        for p in range(n):
            # dist[i, j] += 1 if all_words[i, p] != all_words[j, p]
            # all_words[:, p] is shape (N,)
            # diff = all_words[:, p, np.newaxis] != all_words[np.newaxis, :, p]  # (N, N)
            # This creates a (N, N) bool array. 4.7MB.
            # Summing 7 of these is fine.
            
            col = all_words[:, p]
            diff = col[:, np.newaxis] != col[np.newaxis, :]
            dist += diff.astype(np.int8)
            
        A = dist <= radius
        # Ensure symmetry (should be already) and diagonal
        # A = (A | A.T) # Just in case, though dist is symmetric
        
    else:
        # General radius fallback (slower, but N is small)
        dist = np.zeros((N, N), dtype=np.int8)
        for p in range(n):
            col = all_words[:, p]
            diff = col[:, np.newaxis] != col[np.newaxis, :]
            dist += diff.astype(np.int8)
        A = dist <= radius

    # Phase 1: Greedy Set Cover (Dominating Set)
    # We want to select a set S such that union of N(c) for c in S is all vertices.
    # Greedy: Pick c that covers the most currently uncovered vertices.
    
    def greedy_dominating_set(seed):
        """
        Returns a list of codeword indices forming a dominating set.
        """
        rng = np.random.RandomState(seed)
        
        # uncovered: boolean array of size N. True if not yet dominated.
        # Initially all True.
        uncovered = np.ones(N, dtype=bool)
        
        # covered_mask: boolean array of size N. True if dominated by current S.
        # We can just use ~uncovered.
        
        S = []
        
        # Precompute row sums? No, we need dynamic counts.
        # We maintain `uncovered`.
        # For each candidate c, count = sum(A[c, uncovered])
        # We pick c with max count.
        
        # To speed up, we can maintain an array `gain` where gain[c] = number of new vertices covered by c.
        # Initially gain[c] = A[c].sum()
        # When we pick c*, we add c* to S.
        # Then for all c, gain[c] -= sum(A[c, A[c*, uncovered]]).
        # This is O(N^2) per step if done naively.
        # Since N=2187, and S will be ~150-200, total ops ~ 200 * 2187^2 ~ 1e9. Too slow in Python.
        
        # Better: Just recompute the best candidate from scratch each time?
        # O(N) candidates, each O(N) to count. O(N^2) per step.
        # 200 steps * 4.7M ops = 1e9 ops. Still too slow in pure Python loops.
        
        # We need vectorization.
        # gain = A @ uncovered  # Vector of length N.
        # This is a matrix-vector multiplication. O(N^2) per step.
        # 200 steps * 4.7M flops = 1e9 flops. NumPy is fast enough? 
        # 1e9 flops is ~0.1s - 1s. This is acceptable!
        
        # Let's use this approach.
        
        # uncovered is bool. We can convert to float for matmul.
        # Or just use A @ uncovered.astype(float)
        
        # Optimization: Only consider c not in S.
        # We can set gain[c] = -1 for c in S.
        
        S_set = set()
        
        while True:
            if time.time() > deadline - 20:
                break
                
            # If no uncovered, break
            if not np.any(uncovered):
                break
                
            # Compute gain for all c
            # gain[c] = number of uncovered vertices dominated by c
            # gain = A @ uncovered_float
            uncovered_float = uncovered.astype(np.float32)
            gain = A @ uncovered_float
            
            # Mask out already selected
            for c in S_set:
                gain[c] = -1.0
                
            # Find best
            best_c = np.argmax(gain)
            best_gain = gain[best_c]
            
            if best_gain <= 0:
                # No improvement possible? Should not happen if uncovered exists and A is valid.
                # But if graph is not connected or something? 
                # In covering code, every vertex dominates itself, so gain >= 1 for any c dominating an uncovered vertex.
                break
                
            S.append(best_c)
            S_set.add(best_c)
            
            # Update uncovered
            # uncovered = uncovered & ~A[best_c]
            uncovered &= ~A[best_c]
            
            # Check time
            if time.time() > deadline - 10:
                break
                
        return S

    # Phase 2: Local Search / Pruning
    # We have a valid dominating set S.
    # We try to remove vertices from S if they are redundant.
    # A vertex c in S is redundant if for all v in N(c), there exists c' in S\{c} such that v in N(c').
    # Equivalently: N(c) is subset of union_{c' in S\{c}} N(c').
    # Or: The "private" part of c is empty.
    # Private(c) = N(c) \ union_{c' in S\{c}} N(c')
    
    # We can check redundancy vectorized.
    # Let M be the matrix of A restricted to rows in S.
    # We want to know for each row c in S, if any column v in N(c) has 0s in all other rows S\{c}.
    
    # Algorithm for pruning one step:
    # 1. For each c in S, compute private_count[c] = number of v such that A[c, v]=1 and for all c' in S\{c}, A[c', v]=0.
    # 2. If private_count[c] == 0, c is redundant. Remove it.
    
    # How to compute private_count efficiently?
    # Let S be a list of indices.
    # Let A_S be A[S, :] (shape |S| x N).
    # For a specific c at index i in S:
    # We want to count v where A_S[i, v] == 1 AND sum(A_S[:i, v] + A_S[i+1:, v]) == 0.
    
    # This is hard to vectorize across all i at once without O(|S|^2 * N) or similar.
    # However, |S| is ~150. N is 2187.
    # 150^2 * 2187 ~ 50M ops. This is fast in NumPy if done cleverly, or even in a Python loop over i with vectorized inner ops.
    
    def prune_dominating_set(S, A, deadline):
        """
        Iteratively remove redundant vertices from S.
        Returns the pruned list.
        """
        S = list(S)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Try to remove from the end to avoid index shifting issues, or rebuild list
            # We'll iterate and mark for removal
            
            to_remove = []
            
            # We can parallelize the check for each c in S?
            # For each i, c = S[i]:
            #   neighbors = A[c]
            #   others_union = np.zeros(N, dtype=bool)
            #   for j in range(len(S)):
            #       if j == i: continue
            #       others_union |= A[S[j]]
            #   private = neighbors & ~others_union
            #   if not np.any(private): to_remove.append(i)
            
            # This is O(|S|^2 * N). 150^2 * 2187 ~ 50M.
            # In Python loop over i (150 iters), inner is NumPy ops on 2187 arrays.
            # 150 * (150 * 2187) ops? No, inner loop is 150 OR operations on 2187-length arrays.
            # 150 * 150 * 2187 ~ 50M boolean ops. NumPy handles this in < 0.1s.
            
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                c = S[i]
                neighbors = A[c]
                
                # Compute union of neighbors of all other codewords
                others_union = np.zeros(N, dtype=bool)
                for j in range(len(S)):
                    if j == i:
                        continue
                    others_union |= A[S[j]]
                    
                private = neighbors & ~others_union
                
                if not np.any(private):
                    # Redundant
                    S.pop(i)
                    changed = True
                    # Since we popped, indices shift. But we are iterating backwards, 
                    # so previous indices are safe. 
                    # However, the `others_union` calculation for previous i's 
                    # used the old S. Since we are removing, the set of "others" shrinks.
                    # This is fine for correctness (we might miss some redundancies 
                    # that only appear after removal, which is why we loop `while changed`).
                    
        return S

    # Phase 3: Multi-start search
    best_S = None
    best_size = N + 1
    
    # We will try a few random seeds for the greedy initialization
    # and then prune.
    
    num_restarts = 5
    seed = 42
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomize the order of candidates in greedy?
        # The greedy algorithm picks max gain. If there are ties, we can break randomly.
        # Or we can just use different random seeds for the RNG used in tie-breaking.
        # My current greedy doesn't use RNG for tie-breaking, it picks the first max.
        # To get diversity, we can add a small random noise to the gain?
        # Or just run the same greedy multiple times? No, it's deterministic.
        
        # Let's modify greedy to accept a seed and use it for tie-breaking.
        
        def greedy_dominating_set_random(seed):
            rng = np.random.RandomState(seed)
            uncovered = np.ones(N, dtype=bool)
            S = []
            S_set = set()
            
            while True:
                if time.time() > deadline - 10:
                    break
                if not np.any(uncovered):
                    break
                    
                uncovered_float = uncovered.astype(np.float32)
                gain = A @ uncovered_float
                
                # Add small random noise to break ties
                # Noise should be small enough not to change the choice if gain is significantly different
                # but large enough to break exact ties.
                noise = rng.uniform(0, 0.01, size=N)
                gain += noise
                
                for c in S_set:
                    gain[c] = -1.0
                    
                best_c = np.argmax(gain)
                best_gain = gain[best_c]
                
                if best_gain < 0.005: # Threshold for noise
                    break
                
                S.append(best_c)
                S_set.add(best_c)
                uncovered &= ~A[best_c]
                
            return S

        S = greedy_dominating_set_random(seed + r)
        
        if not S:
            continue
            
        # Prune
        S = prune_dominating_set(S, A, deadline)
        
        # Verify validity (should be valid by construction, but just in case)
        # Check if union of A[c] for c in S is all True
        if len(S) > 0:
            union = np.zeros(N, dtype=bool)
            for c in S:
                union |= A[c]
            if np.all(union):
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = S.copy()
        
        # If we found a very good one, we can stop early?
        # Best known is 186. If we hit 186, we can't do much better easily.
        # But let's keep searching.

    # If no valid set found (shouldn't happen), fallback to full space? No, too big.
    # Fallback to greedy without pruning?
    if best_S is None:
        S = greedy_dominating_set_random(0)
        best_S = S
        best_size = len(S)
        
    return [list(all_words[i]) for i in best_S]

# EVOLVE-BLOCK-END
