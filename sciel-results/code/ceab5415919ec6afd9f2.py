import random
import itertools

# Precompute subset masks for all sizes 1 to 16
SUBSETS_BY_SIZE = [[] for _ in range(17)]
for k in range(1, 17):
    for subset in itertools.combinations(range(16), k):
        mask = 0
        for p in subset:
            mask |= (1 << p)
        SUBSETS_BY_SIZE[k].append(mask)

def is_independent(sub_mask, triple_masks):
    """Check if sub_mask contains no triple from triple_masks."""
    for tm in triple_masks:
        if (sub_mask & tm) == tm:
            return False
    return True

def greedy_reduce_alpha():
    """
    Construct a linear triple system by iteratively adding triples
    that minimize the independence number.
    """
    triples = []
    triple_masks = []
    covered_pairs = set()
    
    # All possible triples
    all_candidates = list(itertools.combinations(range(16), 3))
    
    # Current independence number
    alpha = 16
    # List of masks of independent sets of size 'alpha'
    # Initially, only the full set {0..15} is independent of size 16
    ind_sets = [(1 << 16) - 1]
    
    while True:
        best_t = None
        best_score = float('inf')
        best_new_ind_sets = None
        
        # We need to check all compatible candidates
        # To avoid bias, we can shuffle candidates? 
        # The directive says "maximize reduction". 
        # If multiple reduce, we pick one. 
        # If none reduce, we pick one that minimizes remaining sets.
        
        # Optimization: If we find one that reduces alpha (score 0), 
        # we might still want to check others? 
        # If we pick one, alpha drops. 
        # It's generally best to drop alpha as fast as possible.
        # So if we find a score 0, we can break?
        # Or we can pick the one that leaves the smallest "potential" for next step?
        # For simplicity, let's just pick the first one that reduces alpha.
        # If none reduces, pick the one with min len(new_ind_sets).
        
        found_reduction = False
        
        for t in all_candidates:
            # Check linearity
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            valid = True
            for p in pairs:
                if p in covered_pairs:
                    valid = False
                    break
            if not valid:
                continue
            
            t_mask = 0
            for p in t:
                t_mask |= (1 << p)
            
            # Calculate new independent sets of current size alpha
            new_ind_sets = []
            for s in ind_sets:
                if not ((s & t_mask) == t_mask):
                    new_ind_sets.append(s)
            
            if len(new_ind_sets) == 0:
                # This triple reduces alpha
                # We prioritize this
                # To be deterministic, we can just take the first one
                # Or we can store it and break
                best_t = t
                best_score = 0
                best_new_ind_sets = new_ind_sets
                found_reduction = True
                break # Take the first one that reduces alpha
                
            else:
                score = len(new_ind_sets)
                if score < best_score:
                    best_score = score
                    best_t = t
                    best_new_ind_sets = new_ind_sets
        
        if best_t is None:
            # No more compatible triples
            break
            
        # Add the best triple
        triples.append(list(best_t))
        t_mask = 0
        for p in best_t:
            t_mask |= (1 << p)
        triple_masks.append(t_mask)
        
        pairs = [(best_t[0], best_t[1]), (best_t[0], best_t[2]), (best_t[1], best_t[2])]
        for p in pairs:
            covered_pairs.add(p)
            
        if found_reduction:
            # Alpha decreases
            alpha -= 1
            # Regenerate ind_sets for new alpha
            ind_sets = []
            if alpha > 0:
                for s in SUBSETS_BY_SIZE[alpha]:
                    if is_independent(s, triple_masks):
                        ind_sets.append(s)
            else:
                ind_sets = [] # No independent sets of size 0? 
                             # Actually, empty set is independent of size 0.
                             # But we stop at alpha > 0 usually.
        else:
            ind_sets = best_new_ind_sets

    return triples

def construct():
    # Run the greedy reduction algorithm
    # Since the choice of "first" reducing triple might be arbitrary,
    # we can try multiple random shuffles of candidates to find a better system?
    # The algorithm is deterministic given the order of all_candidates.
    # Let's shuffle all_candidates a few times.
    
    best_triples = []
    best_score = 17
    
    for seed in range(50):
        random.seed(seed)
        # Shuffle candidates
        all_candidates = list(itertools.combinations(range(16), 3))
        random.shuffle(all_candidates)
        
        # We need to pass shuffled candidates to the greedy function
        # Or modify greedy_reduce_alpha to take candidates
        
        # Let's refactor to pass candidates
        
        triples = greedy_reduce_alpha_custom(all_candidates)
        
        # Calculate actual independence number to verify
        score = calculate_independence_number(triples)
        
        if score < best_score:
            best_score = score
            best_triples = triples
            
    return best_triples

def greedy_reduce_alpha_custom(all_candidates):
    triples = []
    triple_masks = []
    covered_pairs = set()
    
    alpha = 16
    ind_sets = [(1 << 16) - 1]
    
    while True:
        best_t = None
        best_score = float('inf')
        best_new_ind_sets = None
        found_reduction = False
        
        for t in all_candidates:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            valid = True
            for p in pairs:
                if p in covered_pairs:
                    valid = False
                    break
            if not valid:
                continue
            
            t_mask = 0
            for p in t:
                t_mask |= (1 << p)
            
            new_ind_sets = []
            for s in ind_sets:
                if not ((s & t_mask) == t_mask):
                    new_ind_sets.append(s)
            
            if len(new_ind_sets) == 0:
                best_t = t
                best_score = 0
                best_new_ind_sets = new_ind_sets
                found_reduction = True
                break
            else:
                score = len(new_ind_sets)
                if score < best_score:
                    best_score = score
                    best_t = t
                    best_new_ind_sets = new_ind_sets
        
        if best_t is None:
            break
            
        triples.append(list(best_t))
        t_mask = 0
        for p in best_t:
            t_mask |= (1 << p)
        triple_masks.append(t_mask)
        
        pairs = [(best_t[0], best_t[1]), (best_t[0], best_t[2]), (best_t[1], best_t[2])]
        for p in pairs:
            covered_pairs.add(p)
            
        if found_reduction:
            alpha -= 1
            ind_sets = []
            if alpha > 0:
                for s in SUBSETS_BY_SIZE[alpha]:
                    if is_independent(s, triple_masks):
                        ind_sets.append(s)
        else:
            ind_sets = best_new_ind_sets

    return triples

def calculate_independence_number(triples):
    """
    Calculate the size of the largest subset of {0..15} that contains no triple from 'triples'.
    """
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    for k in range(16, 0, -1):
        for sub_mask in SUBSETS_BY_SIZE[k]:
            is_independent = True
            for tm in triple_masks:
                if (sub_mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                return k
    return 0
