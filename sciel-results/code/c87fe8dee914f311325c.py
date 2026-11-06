import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(16))
    all_triples_list = list(itertools.combinations(all_points, 3))
    num_triples = len(all_triples_list)
    
    # Precompute masks and triple-to-index mapping
    triple_masks = {}
    triple_to_idx = {}
    for i, t in enumerate(all_triples_list):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = mask
        triple_to_idx[t] = i
        
    # Precompute conflict bitsets
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[all_triples_list[i]]
        for j in range(i+1, num_triples):
            mj = triple_masks[all_triples_list[j]]
            inter = mi & mj
            if inter & (inter - 1):  # share > 1 point
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    def count_ones(mask):
        return bin(mask).count('1')
    
    def find_max_independent_set(active_mask):
        """
        Find the size of the largest subset of points that contains no triple.
        Uses branch-and-bound / backtracking approach.
        """
        if active_mask == 0:
            return 16
        
        # Get list of active triple indices
        active_indices = [i for i in range(num_triples) if (active_mask >> i) & 1]
        if not active_indices:
            return 16
        
        # For each point, compute which active triples contain it
        # point_triples[p] = list of triple indices that contain point p
        point_triples = [[] for _ in range(16)]
        for idx in active_indices:
            t = all_triples_list[idx]
            for p in t:
                point_triples[p].append(idx)
        
        # Greedy upper bound: for each point, if we include it, we must exclude
        # all points that form a triple with it in the active set.
        # Actually, let's use a simpler approach:
        # The independence number is the max k such that there exists a subset S of size k
        # where no triple is entirely contained in S.
        
        # Branch and bound: try to build an independent set greedily with backtracking.
        # Start with all points available, try to include points one by one.
        # If including a point forces us to exclude too many, backtrack.
        
        # Alternative: iterate k from 16 down to 1, check if any subset of size k is independent.
        # For k=16: check if no triple is active. (Already handled)
        # For k=15: check if there's a point p such that no active triple is contained in {0..15}\{p}.
        #   i.e., every active triple contains p.
        # For k=14: check if there's a pair {p,q} such that every active triple intersects {p,q}.
        #   i.e., no active triple is disjoint from {p,q}.
        
        # This is equivalent to: the complement of an independent set is a hitting set for all active triples.
        # So alpha = 16 - (minimum hitting set size for active triples)
        # Minimum hitting set (or minimum vertex cover in hypergraph) is NP-hard, but for 16 points it's small.
        
        # Let's compute minimum hitting set size h, then alpha = 16 - h.
        # h is the min number of points that intersect every active triple.
        
        # Use branch and bound for minimum hitting set:
        # State: set of remaining triples to hit, set of available points.
        # Pick a triple, try each point in it, recurse.
        
        # For efficiency, we'll use a recursive approach with memoization on the set of remaining triples.
        # But the state space might be large. Let's use a greedy + local search for hitting set,
        # or just brute force since 16 points is small.
        
        # Actually, let's just check subsets directly for alpha.
        # For k from 16 down to 1:
        #   For each subset S of size k:
        #     Check if any active triple is entirely in S.
        #     If not, return k.
        
        # To speed up: for a given S, check if any active triple is subset of S.
        # Precompute for each subset mask, which active triples are contained in it?
        # That's too much memory.
        
        # Let's optimize the check: for a given subset S (as a 16-bit mask),
        # we need to check if any active triple mask is a subset of S.
        # i.e., (triple_mask & S) == triple_mask.
        
        # We can precompute for each active triple its mask.
        # Then for a given S, iterate active triples and check.
        
        # For k=16: only one subset. Check all active triples.
        # For k=15: 16 subsets. Check all active triples for each.
        # For k=14: C(16,14)=120 subsets.
        # For k=13: C(16,13)=560 subsets.
        # For k=12: C(16,12)=1820 subsets.
        # For k=11: C(16,11)=4368 subsets.
        # For k=10: C(16,10)=8008 subsets.
        
        # Total subsets to check in worst case: ~14893.
        # For each, we check up to ~560 triples.
        # Total operations: ~8.3 million. This is fast in Python if optimized.
        
        active_masks = [triple_masks[all_triples_list[i]] for i in active_indices]
        
        for k in range(16, 0, -1):
            for subset in itertools.combinations(all_points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                is_ind = True
                for tm in active_masks:
                    if (tm & smask) == tm:
                        is_ind = False
                        break
                if is_ind:
                    return k
            if time.time() > deadline - 5:
                return k  # Conservative estimate
        return 0
    
    def generate_initial():
        """Generate a random linear triple system using greedy method."""
        active_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
        return active_mask
    
    def local_search(mask, max_iterations=100):
        """Local search to minimize independence number."""
        current_alpha = find_max_independent_set(mask)
        current_mask = mask
        
        for it in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Pick a random point from the independent set? 
            # We don't have the independent set, only its size.
            # Let's pick a random triple from the active set, remove it, and try to add others.
            # Or: pick a random triple not in active set, add it, remove conflicts, refill.
            
            # Strategy: pick a random inactive triple, add it, remove conflicts, greedy refill.
            inactive_indices = [i for i in range(num_triples) if not ((current_mask >> i) & 1)]
            if not inactive_indices:
                break
            
            target_idx = random.choice(inactive_indices)
            
            # Remove conflicts
            conflicts = conflict_bits[target_idx] & current_mask
            new_mask = current_mask & ~conflicts
            new_mask |= (1 << target_idx)
            
            # Greedy refill
            refill_indices = list(range(num_triples))
            random.shuffle(refill_indices)
            added = 0
            for idx in refill_indices:
                if (new_mask >> idx) & 1:
                    continue
                if (conflict_bits[idx] & new_mask) == 0:
                    new_mask |= (1 << idx)
                    added += 1
                    if added > 50:  # Limit refill for speed
                        break
            
            new_alpha = find_max_independent_set(new_mask)
            
            if new_alpha <= current_alpha:
                current_alpha = new_alpha
                current_mask = new_mask
        
        return current_mask, current_alpha
    
    best_alpha = 16
    best_mask = 0
    
    # Multiple restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate initial solution
        current_mask = generate_initial()
        current_mask, current_alpha = local_search(current_mask, max_iterations=50)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_mask = current_mask
        
        # If we've reached a very good solution, do more polishing
        if best_alpha <= 6 and time.time() < deadline - 20:
            best_mask, best_alpha = local_search(best_mask, max_iterations=100)
    
    # Final polish
    if time.time() < deadline - 5:
        best_mask, best_alpha = local_search(best_mask, max_iterations=50)
    
    # Convert mask to list of triples
    result = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result.append(list(all_triples_list[i]))
    
    return result
