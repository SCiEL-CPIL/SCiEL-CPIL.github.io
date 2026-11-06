import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # Map pair -> triples containing it
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)
            
    # Precompute bitmask for each triple
    triple_masks = {}
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[t] = mask

    def get_alpha_and_witness(triples):
        """
        Compute the independence number and the witness set.
        """
        if not triples:
            return 16, set(points)
            
        t_masks = [triple_masks[t] for t in triples]
        
        max_alpha = 0
        best_witness = set()
        
        # Iterate k from 16 down to 1
        for k in range(16, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for t_mask in t_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                
                if is_independent:
                    return k, set(subset)
        return 1, set(points[:1])

    # Strategy:
    # 1. Generate random greedy packings.
    # 2. Local search with guidance from the witness set.
    
    best_alpha = 16
    best_system = []
    
    # Initialize with random greedy packings
    for _ in range(20):
        random.shuffle(all_triples)
        current = []
        used_pairs = set()
        for t in all_triples:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        
        alpha, _ = get_alpha_and_witness(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]
            
        if best_alpha <= 5: # Theoretical min for v=16 is likely higher, but check
            break
    
    current_system = best_system[:]
    current_alpha = best_alpha
    current_witness = set()
    
    # Update witness for current best
    if current_system:
        current_alpha, current_witness = get_alpha_and_witness(current_system)
    else:
        current_witness = set(points)

    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])

    def score_move(new_system, current_witness):
        """
        Heuristic score for a move.
        We want to reduce alpha.
        We prefer triples that intersect the witness set significantly.
        """
        if not new_system:
            return 0
        # Calculate how many points of the witness set are covered by the new triples
        # This is a proxy for "breaking" the witness set
        witness_mask = 0
        for p in current_witness:
            witness_mask |= (1 << p)
        
        covered = 0
        for t in new_system:
            if triple_masks[t] & witness_mask:
                covered += 1
                
        return covered

    while time.time() - start_time < deadline:
        # Get current witness if it changed or periodically
        # To save time, we might not recalculate witness every single step, 
        # but if alpha drops, we must.
        
        move_type = random.choice(['add', 'swap'])
        
        if move_type == 'add':
            used_pairs = set()
            for t in current_system:
                used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs):
                    candidates.append(t)
            
            if candidates:
                # Bias towards candidates that intersect the witness set
                # Weight by intersection size
                weights = []
                for t in candidates:
                    inter = len(set(t) & current_witness)
                    weights.append(inter + 1) # +1 to avoid zero weight if no intersection
                    
                # Simple weighted choice
                # To be fast, just pick random from top 10 by weight?
                # Or just random, but filter if we want?
                # Let's just pick randomly for now, but maybe sort if list is small?
                # 5000 candidates is too many to sort every time.
                # Let's just pick randomly. The guidance is weak.
                
                new_t = random.choice(candidates)
                new_system = current_system + [new_t]
                new_alpha, new_witness = get_alpha_and_witness(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    # Accept with probability 0.5
                    if random.random() < 0.5:
                        current_system = new_system
                        # Update witness? It might be different.
                        # Only update if we accept? Yes.
                        # But get_alpha is expensive.
                        # We already have new_witness.
                        current_witness = new_witness

        elif move_type == 'swap':
            if not current_system:
                continue
                
            idx = random.randint(0, len(current_system) - 1)
            t_remove = current_system[idx]
            
            # Remove it temporarily to find candidates
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs):
                    candidates.append(t)
            
            if candidates:
                # Exclude the one we removed if it's still a candidate (it is)
                # We want to replace it.
                # Filter out t_remove
                candidates = [t for t in candidates if t != t_remove]
                
                if candidates:
                    new_t = random.choice(candidates)
                    
                    new_system = current_system[:]
                    new_system[idx] = new_t
                    new_alpha, new_witness = get_alpha_and_witness(new_system)
                    
                    if new_alpha < current_alpha:
                        current_system = new_system
                        current_alpha = new_alpha
                        current_witness = new_witness
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = current_system[:]
                    elif new_alpha == current_alpha:
                        if random.random() < 0.5:
                            current_system = new_system
                            current_witness = new_witness

        if best_alpha <= 5:
            break
    
    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
