"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute pair conflict: for each pair, list of triples containing it
    pair_to_triples = defaultdict(list)
    for idx, t in enumerate(all_triples):
        for i in range(3):
            for j in range(i+1, 3):
                pair = (min(t[i], t[j]), max(t[i], t[j]))
                pair_to_triples[pair].append(idx)
    
    # Precompute conflict mask: for each triple, bitmask of conflicting triples
    # Two triples conflict if they share a pair
    conflict_mask = [0] * num_triples
    for idx, t in enumerate(all_triples):
        for i in range(3):
            for j in range(i+1, 3):
                pair = (min(t[i], t[j]), max(t[i], t[j]))
                for other_idx in pair_to_triples[pair]:
                    if other_idx != idx:
                        conflict_mask[idx] |= (1 << other_idx)
    
    def is_linear_given_mask(selected_mask, triple_idx):
        """Check if adding triple_idx maintains linearity using conflict mask."""
        return (conflict_mask[triple_idx] & selected_mask) == 0
    
    def compute_alpha_fast(triple_indices):
        """Compute independence number using bitmask approach.
        
        A set S is independent iff no triple is fully contained in S.
        We check subsets from largest to smallest to find the maximum independent set.
        """
        if not triple_indices:
            return n
        
        # Build a lookup: for each subset mask, does it contain any triple?
        # We'll use a different approach: for each point, track which triples it covers
        # Then use inclusion-exclusion or direct checking
        
        # Convert triple indices to point masks
        triple_point_masks = []
        for idx in triple_indices:
            t = all_triples[idx]
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_point_masks.append(mask)
        
        # Check all subsets from size n down to 1
        # A subset S (as bitmask) is independent iff no triple's point mask is subset of S
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                subset_mask = 0
                for p in subset:
                    subset_mask |= (1 << p)
                
                # Check if any triple is fully contained
                is_independent = True
                for tm in triple_point_masks:
                    if (tm & subset_mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return size
        
        return 1  # At least one point is always independent
    
    def greedy_build(rng, strategy='random'):
        """Greedy build of a linear hypergraph with different strategies."""
        selected_mask = 0
        selected_indices = []
        
        if strategy == 'random':
            shuffled = list(range(num_triples))
            rng.shuffle(shuffled)
            candidates = shuffled
        elif strategy == 'balanced':
            # Prefer triples that cover uncovered points
            candidates = list(range(num_triples))
        else:
            candidates = list(range(num_triples))
            rng.shuffle(candidates)
        
        for idx in candidates:
            if is_linear_given_mask(selected_mask, idx):
                selected_mask |= (1 << idx)
                selected_indices.append(idx)
        
        return selected_indices
    
    best_alpha = n
    best_triples = None
    
    # Phase 1: Multiple greedy restarts with different strategies
    rng = random.Random(42)
    
    # First, try a few strategic builds
    for strategy in ['random', 'balanced']:
        if time.time() > deadline:
            break
        for _ in range(50):
            if time.time() > deadline:
                break
            triple_indices = greedy_build(rng, strategy)
            alpha = compute_alpha_fast(triple_indices)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triple_indices[:]
                if best_alpha <= 5:
                    break
    
    # Then many random restarts
    while time.time() < deadline - 30:
        triple_indices = greedy_build(rng, 'random')
        alpha = compute_alpha_fast(triple_indices)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triple_indices[:]
            if best_alpha <= 5:
                break
    
    # Phase 2: Local search - try to improve by adding/removing/swapping triples
    if best_triples is not None and time.time() < deadline:
        current_indices = best_triples[:]
        current_alpha = best_alpha
        current_mask = 0
        for idx in current_indices:
            current_mask |= (1 << idx)
        
        # Precompute which triples are not selected
        all_indices = list(range(num_triples))
        
        # Try local moves: add a triple, remove a triple, swap
        improvements = 0
        while time.time() < deadline - 5 and improvements < 200:
            move_made = False
            rng.shuffle(all_indices)
            
            # Try adding a triple
            for idx in all_indices:
                if not (current_mask & (1 << idx)):
                    if is_linear_given_mask(current_mask, idx):
                        new_mask = current_mask | (1 << idx)
                        new_indices = current_indices + [idx]
                        new_alpha = compute_alpha_fast(new_indices)
                        if new_alpha < current_alpha:
                            current_mask = new_mask
                            current_indices = new_indices
                            current_alpha = new_alpha
                            improvements += 1
                            move_made = True
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = current_indices[:]
                            break
            
            # Try removing a triple
            if not move_made:
                for idx in current_indices[:]:
                    new_mask = current_mask & ~(1 << idx)
                    new_indices = [i for i in current_indices if i != idx]
                    new_alpha = compute_alpha_fast(new_indices)
                    if new_alpha < current_alpha:
                        current_mask = new_mask
                        current_indices = new_indices
                        current_alpha = new_alpha
                        improvements += 1
                        move_made = True
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = current_indices[:]
                        break
            
            # Try swapping: remove one, add another
            if not move_made:
                for idx_to_remove in current_indices[:]:
                    temp_mask = current_mask & ~(1 << idx_to_remove)
                    for idx_to_add in all_indices:
                        if not (temp_mask & (1 << idx_to_add)):
                            if is_linear_given_mask(temp_mask, idx_to_add):
                                new_mask = temp_mask | (1 << idx_to_add)
                                new_indices = [i for i in current_indices if i != idx_to_remove] + [idx_to_add]
                                new_alpha = compute_alpha_fast(new_indices)
                                if new_alpha < current_alpha:
                                    current_mask = new_mask
                                    current_indices = new_indices
                                    current_alpha = new_alpha
                                    improvements += 1
                                    move_made = True
                                    if current_alpha < best_alpha:
                                        best_alpha = current_alpha
                                        best_triples = current_indices[:]
                                    break
                    if move_made:
                        break
            
            if not move_made:
                break
    
    # Return the best result
    if best_triples is not None:
        return [[all_triples[idx][0], all_triples[idx][1], all_triples[idx][2]] for idx in best_triples]
    else:
        # Fallback: return a valid construction
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
# EVOLVE-BLOCK-END
