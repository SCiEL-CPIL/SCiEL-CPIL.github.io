import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Generate permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # 2. Precompute suffix maps for fast overlap checking
    # map_k[suffix] = list of indices of perms having that suffix
    # We only need up to n-1
    suffix_maps = [None] * (n + 1)
    for k in range(1, n):
        suffix_maps[k] = {}
        for i, p in enumerate(perms):
            suffix = p[n-k:]
            if suffix not in suffix_maps[k]:
                suffix_maps[k][suffix] = []
            suffix_maps[k][suffix].append(i)
            
    def get_overlap(i1, i2):
        """Calculate overlap between perms[i1] and perms[i2]"""
        p1 = perms[i1]
        p2 = perms[i2]
        # Max possible overlap is n-1
        for k in range(n-1, 0, -1):
            # Check if suffix of p1 of length k matches prefix of p2 of length k
            if p1[n-k:] == p2[:k]:
                return k
        return 0

    # To speed up get_overlap, we can use the maps?
    # Actually, for local search, we just call get_overlap on specific pairs.
    # The dictionary approach is useful for finding *candidates* in Greedy.
    # For a specific pair (i1, i2), the linear check is fast enough (max 6 checks).
    
    # 3. Greedy Construction
    # We want to build a sequence of indices.
    # Start with a random permutation.
    # At each step, find the unvisited permutation with max overlap.
    
    def greedy_build():
        visited = [False] * N
        seq = []
        
        # Start with random
        start_idx = random.randint(0, N - 1)
        seq.append(start_idx)
        visited[start_idx] = True
        
        curr_idx = start_idx
        for _ in range(N - 1):
            best_next = -1
            best_overlap = -1
            
            # We can search by decreasing overlap k
            # But it's easier to just scan all unvisited?
            # Scanning 5040 items is fast if we use the map trick?
            # Let's try the map trick for speed.
            
            # We want to find any unvisited j such that overlap(curr, j) is high.
            # Check k from n-1 down to 1.
            found = False
            for k in range(n-1, 0, -1):
                suffix = perms[curr_idx][n-k:]
                if suffix in suffix_maps[k]:
                    candidates = suffix_maps[k][suffix]
                    for j in candidates:
                        if not visited[j]:
                            # Verify overlap is exactly k? 
                            # If we found it in map_k, overlap is at least k.
                            # Since we check high k first, this is the max.
                            best_next = j
                            best_overlap = k
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                # Fallback: scan all (should not happen if superperm exists, but maybe overlap 0?)
                # Actually, there is always some overlap?
                # In worst case, overlap 0.
                # If no candidate found in maps, it means no unvisited perm shares even suffix 1?
                # That's impossible unless we are stuck.
                # But wait, suffix_maps[1] contains all perms grouped by last digit.
                # There are always unvisited perms with same last digit?
                # Not necessarily.
                # If found is still False, we must pick a random unvisited perm.
                # This breaks the "max overlap" guarantee but ensures progress.
                unvisited_indices = [i for i in range(N) if not visited[i]]
                if unvisited_indices:
                    best_next = random.choice(unvisited_indices)
                else:
                    break # Should not happen
            
            seq.append(best_next)
            visited[best_next] = True
            curr_idx = best_next
            
        return seq

    # 4. Local Search (Or-opt)
    
    def calculate_length(seq):
        if not seq:
            return 0
        total = n
        for i in range(len(seq) - 1):
            total += n - get_overlap(seq[i], seq[i+1])
        return total

    def or_opt_move(seq, i, j, k):
        """
        Move segment of length k from position i to position j.
        Returns new seq and delta length.
        i: start index of segment
        j: insertion index (before j)
        k: length of segment
        """
        # Handle boundaries and relative positions
        # This is tricky. Let's implement a simpler 2-opt or swap first?
        # Or-opt is better.
        
        # Let's stick to a robust 2-opt reversal for simplicity and speed of implementation?
        # 2-opt: reverse seq[i:j+1]
        # Delta depends on 4 boundary connections.
        
        pass

    # Let's implement a simple 2-opt local search.
    # It's easier to get right.
    
    def two_opt_move(seq, i, j):
        """
        Reverse segment seq[i:j+1].
        Returns delta_len.
        """
        # Indices involved: i-1, i, j, j+1
        # Need to handle i=0 and j=N-1 carefully.
        
        # Helper to get cost between two indices, or 0 if out of bounds?
        # No, we only look at internal connections.
        
        # Connections removed:
        # (i-1, i) if i > 0
        # (j, j+1) if j < N-1
        # Connections added:
        # (i-1, j) if i > 0
        # (i, j+1) if j < N-1
        
        # Note: The segment is reversed.
        # Internal connections within [i, j] are reversed but their costs are symmetric?
        # Cost(A, B) is NOT symmetric. Overlap(A, B) != Overlap(B, A).
        # So internal costs change!
        # This makes 2-opt evaluation O(length of segment).
        # For random i, j, segment length is large.
        # So 2-opt is slow to evaluate.
        
        # Or-opt (move small segment) is better because internal costs of the moved segment are preserved.
        # Let's implement Or-opt for k=1, 2, 3.
        
        pass

    # Let's implement Or-opt properly.
    
    def evaluate_or_opt(seq, i, j, k):
        """
        Move segment of length k starting at i to position j.
        j is the index where the segment will be inserted (before j).
        If i < j, the segment shifts left.
        If i > j, the segment shifts right.
        
        Returns delta_len.
        """
        # Case 1: i == j (no move)
        if i == j:
            return 0
            
        # Extract segment
        # seq: ... A, S, B ...
        # A ends at i-1. B starts at i+k.
        # We remove S.
        # New seq: ... A, B ...
        
        # Indices in original seq:
        # seg_start = i
        # seg_end = i + k - 1
        
        # Left neighbor of segment: L = i - 1
        # Right neighbor of segment: R = i + k
        
        # If we insert at j:
        # We need to determine the new neighbors of the segment.
        
        # This is getting complex with index shifting.
        # Let's use a simpler heuristic: **Swap** two random elements.
        # Swap is O(1) to evaluate if we precompute overlaps?
        # No, swap changes neighbors.
        # Swap i, j.
        # Affected connections: (i-1, i), (i, i+1), (j-1, j), (j, j+1).
        # And potentially (i-1, j), (j, i+1) etc if they are close.
        # If |i - j| > 1, the sets of neighbors are disjoint.
        # Then delta is sum of changes at i and j.
        
        # Let's use Swap for simplicity and speed.
        # It's a weaker operator but very fast.
        
        pass

    # Given the time limit, let's stick to a fast Greedy + limited Swap/Or-opt.
    # Actually, the Greedy construction itself is quite strong.
    # Let's just do Greedy and then a few iterations of 1-opt (move one element to best position).
    
    # 1-opt: Remove element at i, insert at best j.
    # Cost: O(N) to find best j for a fixed i.
    # Total O(N^2) per pass.
    # N=5040. N^2 = 25M.
    # One pass might take 10-20s.
    # We can do 1-2 passes.
    
    def one_opt_pass(seq):
        N = len(seq)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                # Remove seq[i]
                elem = seq[i]
                # Calculate cost of removing
                # Neighbors: i-1, i+1
                # Old cost: cost(i-1, i) + cost(i, i+1)
                # New cost (after removal): cost(i-1, i+1)
                
                # Find best insertion point j
                # We want to minimize cost(j-1, elem) + cost(elem, j)
                # Note: if we insert at j, the indices shift.
                
                # This is complex to evaluate in-place.
                # Let's just do random swaps for now, it's safer.
                
                pass
        return seq

    # Let's go with the **Greedy + Random Restarts** approach.
    # Run Greedy multiple times with different starting points.
    # Keep the best one.
    
    best_seq = None
    best_len = float('inf')
    
    # Time budget for Greedy restarts
    # Greedy is O(N^2) worst case, but with maps it's O(N * n * avg_candidates).
    # Should be fast.
    
    while time.time() < deadline - 5:
        seq = greedy_build()
        curr_len = calculate_length(seq)
        
        if curr_len < best_len:
            best_len = curr_len
            best_seq = seq[:]
            
        # Simple local search: Random swaps
        # Let's do a few iterations of swap on the best_seq found so far?
        # Or just rely on multiple restarts.
        
    # Final polish on best_seq with swaps
    if best_seq:
        seq = best_seq[:]
        curr_len = best_len
        T = 10.0
        for _ in range(1000): # Fixed small number of steps
            if time.time() > deadline - 2:
                break
            i, j = random.sample(range(N), 2)
            # Swap
            seq[i], seq[j] = seq[j], seq[i]
            new_len = calculate_length(seq)
            
            if new_len < curr_len:
                curr_len = new_len
                if curr_len < best_len:
                    best_len = curr_len
                    best_seq = seq[:]
            else:
                # Revert
                seq[i], seq[j] = seq[j], seq[i]
                # Accept with prob?
                # For now, just revert if not improved (Hill Climbing)
                # To escape local minima, we could accept worse moves.
                # But with only 1000 steps, Hill Climbing is safer.
                
    # Construct result
    if not best_seq:
        return "".join(map(str, perms[0]))
        
    result = list(perms[best_seq[0]])
    for i in range(1, len(best_seq)):
        p_prev = perms[best_seq[i-1]]
        p_curr = perms[best_seq[i]]
        max_overlap = get_overlap(best_seq[i-1], best_seq[i])
        result.extend(p_curr[max_overlap:])
        
    return "".join(map(str, result))
