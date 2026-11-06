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
    
    # Precompute for each triple, the set of pairs it covers
    triple_pairs = {}
    for i, t in enumerate(all_triples_list):
        pairs = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
        triple_pairs[i] = pairs
    
    # Precompute point-to-triples mapping
    point_to_triples = [[] for _ in range(16)]
    for i, t in enumerate(all_triples_list):
        for p in t:
            point_to_triples[p].append(i)
    
    def find_max_independent_set(active_mask):
        """
        Find the size of the largest subset of points that contains no triple.
        Uses a branch-and-bound approach.
        """
        if active_mask == 0:
            return 16
        
        # Get list of active triple masks
        active_masks = []
        active_indices = []
        for i in range(num_triples):
            if (active_mask >> i) & 1:
                active_masks.append(triple_masks[all_triples_list[i]])
                active_indices.append(i)
        
        if not active_masks:
            return 16
        
        best_k = 0
        
        # Check k from 16 down to 1
        for k in range(16, 0, -1):
            found = False
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
                    best_k = k
                    found = True
                    break
            if found:
                break
            if time.time() > deadline - 5:
                return best_k if best_k > 0 else k
        return best_k if best_k > 0 else 1
    
    def generate_greedy_initial():
        """
        Greedily construct a linear triple system by adding triples that cover
        the most uncovered point-pairs, ensuring linearity.
        """
        active_mask = 0
        covered_pairs = set()
        remaining_triples = list(range(num_triples))
        
        while remaining_triples:
            if time.time() > deadline - 15:
                break
            
            # Find the triple that covers the most uncovered pairs
            best_idx = -1
            best_new_pairs = -1
            
            # Sample a subset for speed if too many remaining
            candidates = remaining_triples if len(remaining_triples) <= 100 else random.sample(remaining_triples, 100)
            
            for idx in candidates:
                if (conflict_bits[idx] & active_mask) != 0:
                    continue
                new_pairs = 0
                for pair in triple_pairs[idx]:
                    if pair not in covered_pairs:
                        new_pairs += 1
                if new_pairs > best_new_pairs:
                    best_new_pairs = new_pairs
                    best_idx = idx
            
            if best_idx == -1 or best_new_pairs <= 0:
                break
            
            # Add the best triple
            active_mask |= (1 << best_idx)
            for pair in triple_pairs[best_idx]:
                covered_pairs.add(pair)
            
            # Remove this triple and its conflicts from remaining
            conflicts = conflict_bits[best_idx]
            remaining_triples = [i for i in remaining_triples if i != best_idx and not ((conflicts >> i) & 1)]
        
        # Greedy refill: add any remaining non-conflicting triples
        for idx in range(num_triples):
            if (active_mask >> idx) & 1:
                continue
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
            
            # Strategy: try swapping - remove one active triple, add one inactive triple
            active_indices = [i for i in range(num_triples) if (current_mask >> i) & 1]
            inactive_indices = [i for i in range(num_triples) if not ((current_mask >> i) & 1)]
            
            if not active_indices or not inactive_indices:
                break
            
            # Try a few random swaps
            improved = False
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                
                rem_idx = random.choice(active_indices)
                add_idx = random.choice(inactive_indices)
                
                # Check if adding add_idx is possible after removing rem_idx
                new_mask = current_mask & ~(1 << rem_idx)
                
                # Remove conflicts with add_idx
                conflicts = conflict_bits[add_idx] & new_mask
                new_mask = new_mask & ~conflicts
                new_mask |= (1 << add_idx)
                
                # Quick greedy refill
                for idx in range(num_triples):
                    if (new_mask >> idx) & 1:
                        continue
                    if (conflict_bits[idx] & new_mask) == 0:
                        new_mask |= (1 << idx)
                
                new_alpha = find_max_independent_set(new_mask)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    current_mask = new_mask
                    improved = True
                    break
                elif new_alpha == current_alpha and random.random() < 0.1:
                    # Accept with small probability for exploration
                    current_mask = new_mask
            
        return current_mask, current_alpha
    
    best_alpha = 16
    best_mask = 0
    
    # Multiple restarts with greedy initial construction
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate initial solution using greedy pair-coverage
        current_mask = generate_greedy_initial()
        current_mask, current_alpha = local_search(current_mask, max_iterations=80)
        
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
