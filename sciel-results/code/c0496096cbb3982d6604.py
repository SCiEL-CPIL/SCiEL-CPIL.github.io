import itertools
import numpy as np
import random

def construct(n):
    if n <= 4:
        # Classic construction is optimal for small n
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute suffixes and prefixes for overlap calculation
    # We need to check if suffix of A matches prefix of B
    # Overlap k means A[n-k:] == B[:k]
    
    # Convert perms to list of strings for easier slicing if needed, 
    # but for matrix we can use tuples or arrays.
    # Let's use a list of tuples for the perms themselves.
    
    # Build Overlap Matrix
    # ov[i][j] = largest k in [1..n-1] such that perms[i][n-k:] == perms[j][:k]
    # If no overlap, ov[i][j] = 0.
    
    # Vectorized approach for speed:
    # For each k from n-1 down to 1:
    #   Check if suffix_k[i] == prefix_k[j]
    #   This is hard to vectorize directly across all i,j for a specific k without large memory.
    #   
    # Alternative: 
    # Since n is small (7), we can just loop.
    # N=5040. N^2 = 25M. 
    # Python loop over 25M items is ~5-10s. Acceptable if we do it once.
    # We can speed it up by using string comparison.
    
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    ov_matrix = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: For each k, create a dict mapping suffix to list of indices
    # Then for each prefix, check if it matches.
    # Actually, simpler:
    # For k in range(n-1, 0, -1):
    #   suffixes = {}
    #   for i in range(N):
    #       s = perm_strs[i][n-k:]
    #       if s not in suffixes:
    #           suffixes[s] = []
    #       suffixes[s].append(i)
    #   for j in range(N):
    #       p = perm_strs[j][:k]
    #       if p in suffixes:
    #           for i in suffixes[p]:
    #               if ov_matrix[i][j] == 0: # Only set if not already set to larger k? 
    #                   # Wait, we iterate k from large to small.
    #                   # So first match is the largest k.
    #                   ov_matrix[i][j] = k
    
    # This is O(N * n * (avg list size)). 
    # With n=7, it's 7 passes.
    # Let's try this.
    
    for k in range(n - 1, 0, -1):
        suffix_map = {}
        for i in range(N):
            s = perm_strs[i][n-k:]
            if s not in suffix_map:
                suffix_map[s] = []
            suffix_map[s].append(i)
        
        for j in range(N):
            p = perm_strs[j][:k]
            if p in suffix_map:
                for i in suffix_map[p]:
                    # We only want to set if it's 0, because we are going from large k to small k.
                    # So the first time we find a match for (i,j) is the max k.
                    if ov_matrix[i][j] == 0:
                        ov_matrix[i][j] = k

    # Cost matrix: cost[i][j] = n - ov_matrix[i][j]
    # We want to minimize sum of costs.
    
    def path_cost(path):
        total = n # First perm
        for i in range(len(path) - 1):
            total += n - ov_matrix[path[i]][path[i+1]]
        return total

    # Greedy Randomized Construction (GRCS)
    # Pick start node
    # At each step, pick next node from candidates with overlap >= threshold
    # Threshold can be dynamic or fixed.
    
    best_path = None
    best_cost = float('inf')
    
    # Run multiple iterations with different random seeds/thresholds
    # Time limit is 120s.
    # Matrix build ~5s.
    # GRCS + 2Opt per iteration?
    
    # Let's do 10 iterations of GRCS, then 2Opt on the best one found?
    # Or 2Opt on each?
    # 2Opt is O(N^2) per pass. N=5040. N^2=25M.
    # One pass of 2Opt in Python is slow (~2-5s).
    # So we can afford maybe 5-10 passes of 2Opt on one path.
    
    # Strategy:
    # 1. Generate 5 diverse paths using GRCS with varying alpha (randomness).
    # 2. Apply 2Opt to each path for a few iterations.
    # 3. Keep the best.
    
    # To save time, maybe just 1 GRCS path with good params and many 2Opt passes?
    # Or 3 paths and 2Opt on the best 2?
    
    # Let's try:
    # 1. GRCS with alpha=0.1 (mostly greedy)
    # 2. 2Opt for 5 passes.
    # 3. Repeat 3 times with different random states.
    
    def grcs(alpha):
        unvisited = set(range(N))
        # Start with a random node? Or 0?
        # Symmetry: 0 is fine, but random start might help.
        start = random.randint(0, N-1)
        path = [start]
        unvisited.remove(start)
        current = start
        
        for _ in range(N - 1):
            # Find candidates
            # We want nodes j such that ov_matrix[current][j] is high.
            # Sort unvisited by overlap? O(N log N) per step -> O(N^2 log N) total. Too slow.
            # Instead, just iterate and keep a set of "good" ones.
            
            # Threshold: max_overlap - alpha * max_possible?
            # Max overlap is n-1 = 6.
            # Let's pick from top 5% of overlaps?
            
            # To avoid sorting, we can just collect all overlaps and pick max, 
            # but with randomness:
            # With prob p, pick best. Else pick random from top K?
            
            # Simpler:
            # Calculate overlaps for all unvisited.
            # Find max overlap.
            # Pick random from those with overlap >= max_overlap - 1 (if exists) or just best?
            
            # Let's use a "candidate pool" approach.
            # Find max overlap among unvisited.
            max_ov = 0
            for j in unvisited:
                if ov_matrix[current][j] > max_ov:
                    max_ov = ov_matrix[current][j]
            
            # Candidates are those with overlap == max_ov or max_ov - 1 (if alpha > 0)
            # To keep it simple and fast, let's just pick best, but with a random restart if stuck?
            # No, let's do:
            # Candidates = { j in unvisited | ov[current][j] >= max_ov - 1 }
            
            candidates = []
            for j in unvisited:
                if ov_matrix[current][j] >= max_ov - 1:
                    candidates.append(j)
            
            if not candidates:
                # Should not happen if max_ov is calculated correctly
                candidates = list(unvisited)
            
            next_node = random.choice(candidates)
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
            
        return path

    def two_opt(path, max_iters=10):
        # 2Opt: Reverse segment [i, j]
        # Cost change:
        # Old: cost(prev, i) + cost(j, next) + sum_internal
        # New: cost(prev, j) + cost(i, next) + sum_internal_reversed
        
        # Note: cost(i, j) = n - ov[i][j].
        # The internal cost of a reversed segment is NOT the same as forward.
        # sum_{k=i}^{j-1} cost(path[k], path[k+1]) vs sum_{k=i}^{j-1} cost(path[k+1], path[k])
        
        # So we must calculate the difference in internal costs too.
        # This makes 2Opt O(N) per check (to sum internal costs).
        # Total O(N^3) for one pass. Too slow for N=5040.
        
        # Optimization:
        # Precompute prefix sums of costs?
        # Let path be p.
        # Let cost(k, k+1) be c_k.
        # Internal cost of p[i..j] is sum(c_i .. c_{j-1}).
        # Reversed internal cost is sum(ov[p[i+1]][p[i]] .. ov[p[j]][p[j-1]])?
        # No, cost is n - ov.
        # So we need sum(n - ov[p[k+1]][p[k]]).
        
        # This requires a separate matrix for reverse costs or just using ov_matrix[p[k+1]][p[k]].
        
        # Given the complexity and time, let's use a simpler 1-Opt (swap adjacent) or "Insertion" heuristic which is O(N) per check and O(1) cost update if we precompute?
        # Actually, 2Opt is standard. Is there a faster way?
        # If we accept O(N^2) per pass, we need O(1) cost diff.
        # Cost diff = (cost(prev, j) - cost(prev, i)) + (cost(i, next) - cost(j, next)) + (Internal_New - Internal_Old)
        
        # Internal_New = sum_{k=i}^{j-1} (n - ov[p[k+1]][p[k]])
        # Internal_Old = sum_{k=i}^{j-1} (n - ov[p[k]][p[k+1]])
        
        # We can precompute prefix sums of forward costs and backward costs?
        # Let F[k] = sum_{t=0}^{k-1} (n - ov[p[t]][p[t+1]])
        # Let B[k] = sum_{t=0}^{k-1} (n - ov[p[t+1]][p[t]])
        
        # Then Internal_Old(i, j) = F[j] - F[i]
        # Internal_New(i, j) = B[j] - B[i]  (Wait, indices need care)
        
        # Let's implement this.
        
        p = list(path)
        Np = len(p)
        
        # Precompute F and B
        F = np.zeros(Np, dtype=np.int32)
        B = np.zeros(Np, dtype=np.int32)
        
        # F[k] is cost from p[0] to p[k] (exclusive of p[k] end cost? No, sum of edges 0..k-1)
        # F[0] = 0
        # F[k] = F[k-1] + cost(p[k-1], p[k])
        
        # B[k] is sum of reverse costs for edges 0..k-1?
        # B[0] = 0
        # B[k] = B[k-1] + cost(p[k], p[k-1])
        
        for k in range(1, Np):
            F[k] = F[k-1] + (n - ov_matrix[p[k-1]][p[k]])
            B[k] = B[k-1] + (n - ov_matrix[p[k]][p[k-1]])
            
        # Total cost of path is F[Np-1] + n (first element) ?
        # My path_cost function: total = n + sum(costs).
        # F[Np-1] is sum of costs for edges 0..Np-2.
        # So total = n + F[Np-1].
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            for i in range(Np - 2):
                # i is start of segment
                # j is end of segment, j > i+1
                for j in range(i + 2, Np):
                    # Segment is p[i]...p[j]
                    # Predecessor: p[i-1] (if i>0)
                    # Successor: p[j+1] (if j<Np-1)
                    
                    old_internal = F[j] - F[i]
                    new_internal = B[j] - B[i] # Sum of reverse costs for edges i..j-1?
                    # Check:
                    # Edges in segment: (p[i], p[i+1]), ..., (p[j-1], p[j])
                    # Reverse edges: (p[i+1], p[i]), ..., (p[j], p[j-1])
                    # B[k] - B[i] sums reverse costs for edges i..k-1?
                    # B[k] = sum_{t=1}^{k} cost_rev(p[t-1], p[t])?
                    # My B definition: B[k] = B[k-1] + cost(p[k], p[k-1]).
                    # So B[j] - B[i] = sum_{t=i+1}^{j} cost(p[t], p[t-1]).
                    # This corresponds to edges (p[i+1], p[i]), ..., (p[j], p[j-1]).
                    # Yes, this is the internal cost of the reversed segment.
                    
                    delta_internal = new_internal - old_internal
                    
                    delta_ext = 0
                    if i > 0:
                        # Edge (p[i-1], p[i]) becomes (p[i-1], p[j])
                        delta_ext += (n - ov_matrix[p[i-1]][p[j]]) - (n - ov_matrix[p[i-1]][p[i]])
                    else:
                        # Start of path. No cost change for start node itself?
                        # The first node is fixed in cost calculation (just n).
                        # The edge from start is internal to the segment?
                        # If i=0, the segment starts at 0.
                        # The "predecessor" doesn't exist.
                        # The edge (p[0], p[1]) is part of the internal sum?
                        # Yes, F[j]-F[0] includes edge (p[0], p[1]).
                        # So no external cost on the left.
                        pass
                    
                    if j < Np - 1:
                        # Edge (p[j], p[j+1]) becomes (p[i], p[j+1])
                        delta_ext += (n - ov_matrix[p[i]][p[j+1]]) - (n - ov_matrix[p[j]][p[j+1]])
                    else:
                        # End of path. No external cost on the right.
                        pass
                    
                    if delta_internal + delta_ext < 0:
                        # Perform reversal
                        p[i:j+1] = reversed(p[i:j+1])
                        improved = True
                        # Update F and B?
                        # This is expensive.
                        # For simplicity, we can just break and restart the pass?
                        # Or just mark improved and continue (using stale F/B is wrong).
                        # Let's just break and recompute F/B for next iteration?
                        # That makes it O(N^3) effectively if we recompute every time.
                        # Better: Just do one pass, find all improvements, apply them, then recompute F/B.
                        # Or just accept that this is a heuristic and maybe 1-Opt is safer?
                        
                        # Given time constraints, let's stick to 1-Opt (adjacent swaps) which is O(1) cost update?
                        # No, 1-Opt also changes internal costs if we consider the segment of length 2.
                        # Swap p[i], p[i+1].
                        # Old internal: cost(p[i], p[i+1])
                        # New internal: cost(p[i+1], p[i])
                        # This is O(1).
                        
                        # Let's switch to 1-Opt (Adjacent Swap) for speed and reliability.
                        # It's less powerful than 2-Opt but much faster.
                        break
                if improved:
                    break # Restart scan from beginning? Or continue?
                    # Continuing with stale F/B is invalid.
                    # So we must restart the pass or recompute.
                    # Let's restart the pass.
                    break
            
        return p

    # Actually, 1-Opt is:
    # for i in range(N-1):
    #   swap p[i], p[i+1] if it reduces cost.
    # Cost change:
    # Left: (p[i-1], p[i]) -> (p[i-1], p[i+1])
    # Mid: (p[i], p[i+1]) -> (p[i+1], p[i])
    # Right: (p[i+1], p[i+2]) -> (p[i], p[i+2])
    
    def one_opt(path, max_iters=5):
        p = list(path)
        Np = len(p)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(Np - 1):
                # Calculate cost diff
                delta = 0
                
                # Left
                if i > 0:
                    old_l = n - ov_matrix[p[i-1]][p[i]]
                    new_l = n - ov_matrix[p[i-1]][p[i+1]]
                    delta += new_l - old_l
                else:
                    # No left cost
                    pass
                
                # Middle
                old_m = n - ov_matrix[p[i]][p[i+1]]
                new_m = n - ov_matrix[p[i+1]][p[i]]
                delta += new_m - old_m
                
                # Right
                if i < Np - 2:
                    old_r = n - ov_matrix[p[i+1]][p[i+2]]
                    new_r = n - ov_matrix[p[i]][p[i+2]]
                    delta += new_r - old_r
                else:
                    pass
                
                if delta < 0:
                    p[i], p[i+1] = p[i+1], p[i]
                    improved = True
                    # Continue scanning? Or restart?
                    # Continuing is usually fine for 1-Opt.
        
        return p

    # Let's use 1-Opt. It's fast.
    
    best_cost = float('inf')
    best_path = None
    
    for run in range(5):
        random.seed(run)
        path = grcs(0.1)
        # Apply 1-Opt
        path = one_opt(path, max_iters=10)
        cost = path_cost(path)
        
        if cost < best_cost:
            best_cost = cost
            best_path = path
            
    # Construct final string
    if best_path is None:
        # Fallback to classic if something went wrong
        word_classic = [1]
        for m in range(2, n + 1):
            perms_local, seen = [], set()
            for i in range(len(word_classic) - m + 2):
                window = tuple(word_classic[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_local.append(window)
            new = []
            for p in perms_local:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word_classic = new
        return "".join(map(str, word_classic))

    res = perm_strs[best_path[0]]
    for i in range(1, N):
        ov = ov_matrix[best_path[i-1]][best_path[i]]
        res += perm_strs[best_path[i]][ov:]
    return res
