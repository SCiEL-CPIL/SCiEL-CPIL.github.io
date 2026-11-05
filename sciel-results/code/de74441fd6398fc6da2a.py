import time
import random
import itertools
import numpy as np

def construct():
    """
    Constructs a linear triple system on 16 points minimizing the independence number.
    Uses random greedy search with local search refinement.
    """
    n = 16
    points = list(range(n))
    
    # Pre-generate all possible triples as bitmasks
    all_triples = []
    all_triple_masks = []
    for combo in itertools.combinations(points, 3):
        mask = 0
        for p in combo:
            mask |= (1 << p)
        all_triples.append(combo)
        all_triple_masks.append(mask)
    
    # Precompute which pairs are covered by each triple
    # triple_pair_indices[i] = list of pair indices covered by triple i
    all_pairs = list(itertools.combinations(points, 2))
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    triple_pairs = []
    for t in all_triples:
        pairs = []
        for i in range(3):
            for j in range(i+1, 3):
                pairs.append(pair_to_idx[tuple(sorted((points[t[i]], points[t[j]])))])
        triple_pairs.append(pairs)
    
    # Precompute for each pair, which triples cover it
    pair_to_triples = [[] for _ in range(len(all_pairs))]
    for i, t in enumerate(all_triples):
        for pi in triple_pairs[i]:
            pair_to_triples[pi].append(i)
    
    def is_valid_addition(triple_idx, existing_triple_indices):
        """Check if adding triple_idx is valid (no pair overlap with existing)."""
        for pi in triple_pairs[triple_idx]:
            for j in existing_triple_indices:
                if pi in triple_pairs[j]:
                    return False
        return True
    
    def compute_independence_number_fast(triple_indices):
        """Compute independence number using bitmask approach."""
        if not triple_indices:
            return n
        
        # Build a set of triple masks for fast lookup
        triple_masks_set = set(all_triple_masks[i] for i in triple_indices)
        
        # For each subset size k from n down to 1, check if there's an independent set
        # A subset S is independent if no triple is fully contained in S
        
        # Precompute: for each mask, does it contain any triple?
        # We'll check on the fly for efficiency
        
        for k in range(n, 0, -1):
            for combo in itertools.combinations(points, k):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                
                # Check if any triple is subset of mask
                is_ind = True
                for tm in triple_masks_set:
                    if (mask & tm) == tm:
                        is_ind = False
                        break
                if is_ind:
                    return k
        
        return 0
    
    def local_search(triple_indices, current_alpha):
        """Try to improve by swapping triples."""
        if not triple_indices:
            return [], current_alpha
        
        best = list(triple_indices)
        best_alpha = current_alpha
        
        # Try removing one triple and adding a different one
        for remove_idx in range(len(best)):
            remaining = [t for i, t in enumerate(best) if i != remove_idx]
            
            # Find all triples that can be added to remaining
            valid_additions = []
            for t_idx in range(len(all_triples)):
                if t_idx in remaining:
                    continue
                if is_valid_addition(t_idx, remaining):
                    valid_additions.append(t_idx)
            
            for add_idx in valid_additions:
                candidate = remaining + [add_idx]
                alpha = compute_independence_number_fast(candidate)
                if alpha < best_alpha:
                    best = candidate
                    best_alpha = alpha
        
        return best, best_alpha
    
    best_alpha = n
    best_triples = []
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # Seed random
    random.seed(42)
    
    # Run random restarts with local search
    while time.time() < deadline:
        # Greedy construction
        triples = []
        shuffled_indices = list(range(len(all_triples)))
        random.shuffle(shuffled_indices)
        
        for t_idx in shuffled_indices:
            if is_valid_addition(t_idx, triples):
                triples.append(t_idx)
        
        # Calculate independence number
        alpha = compute_independence_number_fast(triples)
        
        # Local search to try to improve
        triples, alpha = local_search(triples, alpha)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            # If we hit a very low alpha, we might stop early?
            if best_alpha <= 5:
                break
    
    # Convert triple indices back to triples
    return [list(all_triples[i]) for i in best_triples]

# EVOLVE-BLOCK-END
