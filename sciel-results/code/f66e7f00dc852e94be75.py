import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all triples as masks and indices
    # Map mask to index for quick lookup?
    # There are 560 triples.
    triple_list = []
    mask_to_idx = {}
    idx_to_mask = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(triple_list)
                triple_list.append(mask)
                mask_to_idx[mask] = idx
                idx_to_mask[idx] = mask
                
    num_triples = len(triple_list)
    
    # Precompute conflicts
    # conflicts[i] is a set of indices j such that triple i and j share >= 2 points
    conflicts = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        mi = triple_list[i]
        for j in range(i + 1, num_triples):
            mj = triple_list[j]
            if (mi & mj) and (mi & mj) != (mi & mj) & 0: # just count bits
                # faster: bin(x).count('1')
                if bin(mi & mj).count('1') >= 2:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    
    # Precompute masks by size for alpha calculation
    # We need to check independent sets of size k
    # Independent set S: no edge is subset of S
    # Equivalent: for all e in edges, (S & e) != e
    
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        k = bin(mask).count('1')
        if k > 0:
            masks_by_size[k].append(mask)
            
    # For fast alpha check, we only care about sizes 5, 4, 3, 2, 1
    # If no independent set of size 5, check 4, etc.
    
    def is_independent(mask, edges_set):
        """
        edges_set: set of indices
        mask: bitmask of the candidate independent set
        """
        for e_idx in edges_set:
            e_mask = idx_to_mask[e_idx]
            if (mask & e_mask) == e_mask:
                return False
        return True
        
    def calc_alpha(edges_set):
        if not edges_set:
            return n
        # Check size 5 down to 1
        # Optimization: if we find an independent set of size k, alpha is at least k.
        # We want the max.
        # But for the search, we primarily care if alpha > 4.
        # So check 5 first.
        for m in masks_by_size[5]:
            if is_independent(m, edges_set):
                return 5
        for m in masks_by_size[4]:
            if is_independent(m, edges_set):
                return 4
        for m in masks_by_size[3]:
            if is_independent(m, edges_set):
                return 3
        for m in masks_by_size[2]:
            if is_independent(m, edges_set):
                return 2
        return 1

    def is_valid_add(triple_idx, edges_set):
        # Check if triple_idx conflicts with any in edges_set
        for c in conflicts[triple_idx]:
            if c in edges_set:
                return False
        return True

    def greedy_pack():
        current_triples = set()
        available = list(range(num_triples))
        random.shuffle(available)
        
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
        
        return current_triples

    def random_swap(current_triples):
        """
        Perform a random swap: remove one edge, add one valid edge.
        Returns new set or None if no move possible.
        """
        if not current_triples:
            return None
            
        to_remove = random.choice(list(current_triples))
        new_triples = set(current_triples)
        new_triples.remove(to_remove)
        
        # Find a valid add
        # Try random candidates
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        # Optimization: prioritize candidates that are "close" or random
        for idx in candidates:
            if idx in new_triples:
                continue
            if is_valid_add(idx, new_triples):
                new_triples.add(idx)
                return new_triples
        return None

    def targeted_swap(current_triples, witness_mask):
        """
        Try to fix the independent set 'witness_mask' by adding an edge inside it.
        If direct add fails, try swapping a conflicting edge.
        """
        # Get all triples inside witness_mask
        # witness_mask has 5 bits set.
        # Extract points
        points = []
        for i in range(n):
            if witness_mask & (1 << i):
                points.append(i)
                
        # Generate all triples from these 5 points
        local_triples = []
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    mask = (1 << points[i]) | (1 << points[j]) | (1 << points[k])
                    idx = mask_to_idx.get(mask)
                    if idx is not None:
                        local_triples.append(idx)
                        
        random.shuffle(local_triples)
        
        for t_idx in local_triples:
            if t_idx in current_triples:
                continue # already there? shouldn't be if witness is independent
                
            # Check if t_idx can be added directly
            if is_valid_add(t_idx, current_triples):
                new_triples = set(current_triples)
                new_triples.add(t_idx)
                return new_triples
            else:
                # Find conflict
                # We need to find an edge e in current_triples that conflicts with t_idx
                # and check if t_idx is valid if e is removed.
                # Note: t_idx might conflict with multiple edges.
                # We can only swap one edge at a time.
                # So we need a conflict e such that t_idx is valid against (current_triples - {e}).
                
                # Get all conflicts of t_idx that are in current_triples
                conflicts_in_set = [c for c in conflicts[t_idx] if c in current_triples]
                
                for c_idx in conflicts_in_set:
                    # Try swapping c_idx with t_idx
                    new_triples = set(current_triples)
                    new_triples.remove(c_idx)
                    # Check if t_idx is valid against new_triples
                    if is_valid_add(t_idx, new_triples):
                        new_triples.add(t_idx)
                        return new_triples
        return None

    best_alpha = n
    best_triples = None
    current_triples = greedy_pack()
    current_alpha = calc_alpha(current_triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = set(current_triples)
        
    # Local Search
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.05
    
    while time.time() < deadline:
        # Calculate alpha
        # Optimization: if we just did a swap, we can update alpha incrementally?
        # Hard to do correctly. Just recalc. It's fast enough.
        current_alpha = calc_alpha(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = set(current_triples)
            if best_alpha <= 4:
                # We found a very good solution.
                # Try to see if we can get 3? Unlikely.
                # Continue searching to ensure robustness or find even better (impossible if 3 is min for n=16? No, 3 is possible for small n).
                # For n=16, alpha=4 is likely the minimum for linear systems?
                # Actually, if we have alpha=4, we can't do much better unless alpha=3.
                # Let's keep searching.
                pass

        # Generate Move
        move_type = random.random()
        
        new_triples = None
        
        if current_alpha > 4 and move_type < 0.7:
            # Targeted move on a witness
            # Find a witness of size current_alpha (or 5 if alpha>=5)
            target_size = 5 if current_alpha >= 5 else current_alpha
            # Find one witness
            witness = None
            # Sample random masks of target_size
            # To be fast, don't check all. Check a few.
            candidates = random.sample(masks_by_size[target_size], min(50, len(masks_by_size[target_size])))
            for m in candidates:
                if is_independent(m, current_triples):
                    witness = m
                    break
            
            if witness:
                new_triples = targeted_swap(current_triples, witness)
        
        if new_triples is None:
            # Random swap
            new_triples = random_swap(current_triples)
            
        if new_triples is None:
            # Stuck? Perturb
            # Remove 2-3 random edges
            for _ in range(random.randint(2, 3)):
                if current_triples:
                    current_triples.remove(random.choice(list(current_triples)))
            continue
            
        new_alpha = calc_alpha(new_triples)
        
        # Acceptance
        delta = new_alpha - current_alpha
        if delta <= 0:
            current_triples = new_triples
            current_alpha = new_alpha
            # Reset T if improved?
            if delta < 0:
                T = max(T, 0.5)
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < (2.71828 ** (-delta / T)):
                current_triples = new_triples
                current_alpha = new_alpha
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # Perturb if stuck at low T
            if random.random() < 0.1:
                for _ in range(2):
                    if current_triples:
                        current_triples.remove(random.choice(list(current_triples)))

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    # Convert indices back to points
    result = []
    for idx in best_triples:
        mask = idx_to_mask[idx]
        pts = []
        for i in range(n):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
