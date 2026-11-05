import time
import numpy as np
import itertools

def construct(n):
    start_time = time.time()
    time_limit = 115.0
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Generate standard superpermutation word
    # This gives a sequence of permutations
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_k = min(len(new), len(block))
            k = 0
            for k_try in range(max_k, 0, -1):
                if new[-k_try:] == block[:k_try]:
                    k = k_try
                    break
            new += block[k:]
        word = new
    
    # Extract order of first appearance of each permutation
    seen_perms = set()
    perm_seq = []
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p not in seen_perms:
            seen_perms.add(p)
            perm_seq.append(p)
            
    if len(perm_seq) != n ** 0 * 5040: # n=7 check, but keep general
        # Actually just check if we have n! perms
        if len(perm_seq) != 5040 and n == 7:
             return "".join(map(str, word))
        if len(perm_seq) < 1:
             return "".join(map(str, word))
        
    N = len(perm_seq)
    
    # Map permutation tuple to index
    perm_to_idx = {p: i for i, p in enumerate(perm_seq)}
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perm_seq[i][n-k:] == perm_seq[j][:k]
    # We use a numpy array for speed
    overlap = np.zeros((N, N), dtype=np.uint8)
    
    # Create a dictionary for suffixes of length k
    # prefix_maps[k][prefix] = list of indices j such that perm_seq[j][:k] == prefix
    prefix_maps = [{} for _ in range(n)]
    for i, p in enumerate(perm_seq):
        for k in range(1, n):
            pref = p[:k]
            if pref not in prefix_maps[k]:
                prefix_maps[k][pref] = []
            prefix_maps[k][pref].append(i)
            
    # Now compute overlap
    for i in range(N):
        pi = perm_seq[i]
        # Check max possible overlap
        for k in range(n - 1, 0, -1):
            suffix = pi[n-k:]
            # Find all j such that perm_seq[j][:k] == suffix
            if suffix in prefix_maps[k]:
                for j in prefix_maps[k][suffix]:
                    overlap[i][j] = k
            # If we found matches for k, we don't need to check smaller k for those j?
            # No, overlap is the MAX k. So if we found k, it's the max for that i,j pair.
            # But we need to fill the matrix.
            # Wait, the loop above sets overlap[i][j] = k.
            # If we iterate k from n-1 down to 1, the first match is the max.
            # So we can break? No, we need to fill all j.
            # But for a specific i, if we find a j with overlap k, we don't care about smaller overlaps for that j.
            # However, we are filling the matrix.
            # This approach is O(N * (sum of counts)).
            # Sum of counts for a specific k is N.
            # So total ops is roughly N * N * (1/k factor)?
            # Actually, for each i, we check k=n-1, then k=n-2, etc.
            # The number of j's for a specific suffix is small.
            # This should be much faster than N^2 * n.
            
    # Compute initial score
    # score = sum of overlaps
    # The length of the word is N + n - sum(overlaps).
    # To minimize length, we maximize sum of overlaps.
    
    # Strategy:
    # 1. Use "Bridge-and-Tunnel" heuristic to find a good initial ordering.
    #    - Build a graph where nodes are permutations.
    #    - Edge weight is overlap (higher is better).
    #    - We want a path that visits all nodes (Hamiltonian path) maximizing sum of weights.
    #    - This is NP-hard.
    #    - Heuristic: Start with the node that has the highest average overlap?
    #    - Or use a greedy nearest neighbor based on overlap.
    
    # Let's try a greedy construction first to get a decent starting point for 2-opt.
    # Greedy: Start with any node (e.g., 0). At each step, go to the unvisited node with max overlap.
    # This is O(N^2).
    
    visited = [False] * N
    current_seq = [0]
    visited[0] = True
    current_overlap_sum = 0
    
    # Precompute sorted neighbors for each node?
    # Or just scan.
    # Scanning is O(N^2) total for the construction.
    
    for _ in range(N - 1):
        curr = current_seq[-1]
        best_next = -1
        best_ov = -1
        # Optimization: check unvisited nodes
        for j in range(N):
            if not visited[j]:
                ov = overlap[curr][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        
        if best_next == -1:
            # Should not happen
            break
            
        current_seq.append(best_next)
        visited[best_next] = True
        current_overlap_sum += best_ov
        
    # Now apply 2-Opt Local Search to improve the path
    # We want to maximize sum of overlaps.
    # Delta for reversing segment [i, j]:
    # Remove edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
    # Add edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
    # Note: internal edges (k, k+1) become (k+1, k).
    
    # To speed up 2-opt, we can precompute:
    # rev_overlap[i][j] = overlap[j][i]
    # But overlap is not symmetric.
    
    # Let's implement 2-opt with first-best improvement.
    
    improved = True
    # Limit iterations or time
    while improved and (time.time() - start_time) < time_limit - 10:
        improved = False
        # First-Best strategy
        for i in range(N - 1):
            if (time.time() - start_time) > time_limit - 10:
                break
            for j in range(i + 1, N):
                # Calculate delta
                # Edges removed:
                # 1. (i-1, i) if i > 0
                # 2. (j, j+1) if j < N-1
                # 3. Internal: (k, k+1) for k in [i, j-1]
                
                # Edges added:
                # 1. (i-1, j) if i > 0
                # 2. (i, j+1) if j < N-1
                # 3. Internal: (k+1, k) for k in [i, j-1]
                
                delta = 0
                
                if i > 0:
                    delta -= overlap[current_seq[i-1]][current_seq[i]]
                    delta += overlap[current_seq[i-1]][current_seq[j]]
                else:
                    # If i == 0, we don't have a left neighbor.
                    # The path starts at current_seq[0].
                    # If we reverse [0, j], the new start is current_seq[j].
                    # The edge removed is (0, 1) -> (1, 0)? No.
                    # Let's trace:
                    # Seq: ... A, B, C ...
                    # i=0, j=1.
                    # Old: A, B. Edge (A,B).
                    # New: B, A. Edge (B,A).
                    # So we remove (A,B) and add (B,A).
                    # My formula:
                    # if i > 0: ...
                    # else:
                    #   We are reversing from the start.
                    #   The "left" boundary is open.
                    #   The edge (i, i+1) is internal? No, it's the first edge.
                    #   Let's stick to the general formula which handles boundaries.
                    #   If i == 0:
                    #   Removed: (0, 1) is part of internal?
                    #   Internal loop k in [0, j-1]:
                    #   Removes (k, k+1). Adds (k+1, k).
                    #   So for i=0, j=1:
                    #   Internal: k=0. Removes (0,1), Adds (1,0).
                    #   Boundary left: None.
                    #   Boundary right: (j, j+1) removed, (i, j+1) added.
                    #   So:
                    #   delta = -overlap[seq[0]][seq[1]] + overlap[seq[1]][seq[0]]
                    #          - overlap[seq[1]][seq[2]] + overlap[seq[0]][seq[2]]
                    
                    pass # Handled by internal and right boundary
                
                if j < N - 1:
                    delta -= overlap[current_seq[j]][current_seq[j+1]]
                    delta += overlap[current_seq[i]][current_seq[j+1]]
                
                # Internal
                # Sum of (overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]]) for k in [i, j-1]
                # This can be precomputed if we maintain a prefix sum of differences?
                # diff[k] = overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]]
                # Sum diff[i:j]
                
                # Since N is 5040, N^2 is 25M.
                # Inner loop is O(j-i).
                # Total work per pass is O(N^3) in worst case?
                # 5040^3 is too big.
                # We need to optimize.
                
                # Optimization:
                # Precompute prefix sums of the "reverse difference".
                # Let D[k] = overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]]
                # Then internal delta = sum(D[i] ... D[j-1])
                # We can maintain an array D and a prefix sum array P.
                # But D changes when we reverse a segment.
                # Reversing [i, j] changes:
                # 1. The elements in seq[i..j] are reversed.
                # 2. The edges inside [i, j] are reversed (direction).
                #    So D[k] for k in [i, j-1] becomes -D[k]?
                #    Let's check:
                #    Old: ... A, B, C ...
                #    D_AB = overlap[B][A] - overlap[A][B]
                #    D_BC = overlap[C][B] - overlap[B][C]
                #    New: ... C, B, A ...
                #    D_CB = overlap[B][C] - overlap[C][B] = -D_BC
                #    D_BA = overlap[A][B] - overlap[B][A] = -D_AB
                #    So yes, the differences inside the segment are negated.
                # 3. The edges at the boundaries change.
                
                # Updating P after a reversal is tricky but doable.
                # However, for N=5040, a simple O(N^2) 2-opt with early termination
                # and first-best might be fast enough if we limit the search space
                # or use a smarter heuristic.
                
                # Let's try a simpler approach: 2-Swap or Or-Opt?
                # Or just run 2-opt but with a time limit per iteration?
                
                # Given the time limit, let's try to optimize the inner loop.
                # Instead of calculating the sum every time, we can use the prefix sum idea
                # and update it.
                
                # But implementing the update correctly is error-prone.
                # Let's try a different heuristic: Simulated Annealing?
                # Or just a faster 2-opt by limiting j?
                
                # Let's try "First-Best" 2-opt but skip if delta is clearly negative?
                # We can't know delta without calculating it.
                
                # Alternative: Use a library? No.
                
                # Let's try to speed up the Python loop.
                # N=5040.
                # 5040 * 5040 / 2 = 12.6M pairs.
                # For each pair, we do a sum of length (j-i).
                # Average length is N/3 ~ 1700.
                # 12.6M * 1700 = 21 Billion ops. Too slow.
                
                # We MUST use the prefix sum optimization.
                
                # Let's restart the 2-opt with prefix sums.
                break # Break out to implement the optimized version
                
        # If we broke out to implement optimized version, we do it here.
        # But I am in the middle of the function.
        # I will replace the entire 2-opt block.
        
    # --- Optimized 2-Opt with Prefix Sums ---
    
    # Re-initialize seq if needed? No, we have current_seq from greedy.
    # But we broke out of the loop.
    # Let's restart the optimization logic properly.
    
    # Recalculate current_overlap_sum from current_seq to be safe
    current_overlap_sum = 0
    for i in range(N - 1):
        current_overlap_sum += overlap[current_seq[i]][current_seq[i+1]]
        
    # Precompute D[k] = overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]]
    # and prefix sums P[k] = sum(D[0]...D[k-1])
    # P[0] = 0
    # P[k+1] = P[k] + D[k]
    
    def build_prefix(seq):
        D = np.zeros(N - 1, dtype=np.int32)
        for k in range(N - 1):
            D[k] = overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]]
        P = np.zeros(N, dtype=np.int32)
        P[1:] = np.cumsum(D)
        return D, P
        
    D, P = build_prefix(current_seq)
    
    def get_internal_delta(i, j):
        # Sum of D[i] to D[j-1]
        # Note: D[k] corresponds to edge (k, k+1) in current seq.
        # If we reverse [i, j], the internal edges are reversed.
        # The contribution to the score change from internal edges is:
        # New internal sum - Old internal sum
        # Old internal sum = sum(overlap[seq[k]][seq[k+1]] for k in i..j-1)
        # New internal sum = sum(overlap[seq[k+1]][seq[k]] for k in i..j-1)
        # Diff = sum(overlap[seq[k+1]][seq[k]] - overlap[seq[k]][seq[k+1]]) = sum(D[k] for k in i..j-1)
        # Using prefix sums: P[j] - P[i]
        return P[j] - P[i]
        
    improved = True
    while improved and (time.time() - start_time) < time_limit - 10:
        improved = False
        for i in range(N - 1):
            if (time.time() - start_time) > time_limit - 10:
                break
            for j in range(i + 1, N):
                # Boundary deltas
                delta = 0
                if i > 0:
                    delta -= overlap[current_seq[i-1]][current_seq[i]]
                    delta += overlap[current_seq[i-1]][current_seq[j]]
                if j < N - 1:
                    delta -= overlap[current_seq[j]][current_seq[j+1]]
                    delta += overlap[current_seq[i]][current_seq[j+1]]
                
                # Internal delta
                delta += get_internal_delta(i, j)
                
                if delta > 0:
                    # Perform reversal
                    current_seq[i:j+1] = current_seq[i:j+1][::-1]
                    current_overlap_sum += delta
                    improved = True
                    
                    # Update D and P
                    # The segment [i, j] is reversed.
                    # 1. D values inside [i, j-1] are negated and reversed?
                    #    Old D[k] for k in [i, j-1].
                    #    New edges are reversed.
                    #    The new D' for the segment corresponds to the new edges.
                    #    Let's just rebuild D and P for simplicity?
                    #    N=5040. Rebuilding is O(N).
                    #    If we do this for every improvement, and there are many improvements...
                    #    How many improvements?
                    #    In 2-opt, usually a few hundred to a few thousand.
                    #    1000 * 5040 = 5M ops. Fast enough.
                    
                    D, P = build_prefix(current_seq)
                    
                    # First-best: restart scan
                    break
        
    # Reconstruct word
    res = list(perm_seq[current_seq[0]])
    for i in range(1, N):
        prev = perm_seq[current_seq[i-1]]
        curr = perm_seq[current_seq[i]]
        k = overlap[current_seq[i-1]][current_seq[i]]
        res.extend(curr[k:])
        
    return "".join(map(str, res))
