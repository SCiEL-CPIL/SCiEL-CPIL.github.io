import random
import time
import itertools
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples and their masks
    all_triples = list(itertools.combinations(range(n), 3))
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in all_triples]
    triple_to_mask = {t: m for t, m in zip(all_triples, triple_masks)}
    
    # Precompute pair masks for each triple
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset([(a,b), (a,c), (b,c)])
    
    best_alpha = n
    best_triples = []
    
    def compute_alpha(triples):
        """Compute independence number using bitmask approach."""
        if not triples:
            return n
        
        t_masks = [triple_to_mask[t] for t in triples]
        
        # Check sizes from high to low
        for size in range(n, 0, -1):
            for mask in itertools.combinations(range(n), size):
                subset_mask = 0
                for p in mask:
                    subset_mask |= (1 << p)
                # Check if any triple is contained in this subset
                independent = True
                for tm in t_masks:
                    if (tm & subset_mask) == tm:
                        independent = False
                        break
                if independent:
                    return size
        return 0
    
    def generate_greedy_solution(strategy='random'):
        """Generate initial linear triple system using different strategies."""
        triples = []
        used_pairs = set()
        
        if strategy == 'random':
            candidates = all_triples[:]
            random.shuffle(candidates)
        elif strategy == 'spread':
            # Try to spread points evenly
            candidates = all_triples[:]
            # Weight triples that use less common points
            point_counts = [0] * n
            candidates.sort(key=lambda t: sum(point_counts[p] for p in t))
            random.shuffle(candidates)
        elif strategy == 'clique':
            # Build a large clique first, then add others
            candidates = all_triples[:]
            random.shuffle(candidates)
        else:
            candidates = all_triples[:]
            random.shuffle(candidates)
        
        for t in candidates:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        
        return triples
    
    def local_search(triples, time_limit):
        """Apply local search to improve a solution."""
        current_triples = triples[:]
        current_alpha = compute_alpha(current_triples)
        
        if current_alpha < best_alpha:
            return current_triples, current_alpha
        
        # Build used pairs set
        used_pairs = set()
        for t in current_triples:
            a, b, c = t
            used_pairs.add((a,b))
            used_pairs.add((a,c))
            used_pairs.add((b,c))
        
        # Precompute for each triple which other triples it conflicts with (share a pair)
        # This helps in finding valid candidates faster
        conflict_map = {}
        for i, t1 in enumerate(all_triples):
            conflict_map[t1] = set()
        
        for i, t1 in enumerate(all_triples):
            for j, t2 in enumerate(all_triples):
                if i == j:
                    continue
                if triple_pairs[t1] & triple_pairs[t2]:
                    conflict_map[t1].add(t2)
        
        temperature = 1.0
        cooling_rate = 0.9995
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < time_limit:
            if temperature < 0.0001:
                break
            if no_improve_count > max_no_improve:
                break
                
            if not current_triples:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Move 1: Remove one triple, add a different one
                idx = random.randint(0, len(current_triples) - 1)
                t_remove = current_triples[idx]
                
                # Temporarily remove
                current_triples.pop(idx)
                a, b, c = t_remove
                used_pairs.discard((a,b))
                used_pairs.discard((a,c))
                used_pairs.discard((b,c))
                
                # Find valid candidates to add
                valid_candidates = []
                for t_new in all_triples:
                    if t_new == t_remove:
                        continue
                    pairs_new = triple_pairs[t_new]
                    if not (pairs_new & used_pairs):
                        valid_candidates.append(t_new)
                
                if not valid_candidates:
                    # Revert
                    current_triples.insert(idx, t_remove)
                    used_pairs.add((a,b))
                    used_pairs.add((a,c))
                    used_pairs.add((b,c))
                    temperature *= cooling_rate
                    continue
                
                t_new = random.choice(valid_candidates)
                
                # Add new triple
                current_triples.append(t_new)
                x, y, z = t_new
                used_pairs.add((x,y))
                used_pairs.add((x,z))
                used_pairs.add((y,z))
                
                new_alpha = compute_alpha(current_triples)
                
                # Metropolis criterion
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_alpha = new_alpha
                    no_improve_count = 0
                else:
                    # Revert
                    current_triples.pop()
                    used_pairs.discard((x,y))
                    used_pairs.discard((x,z))
                    used_pairs.discard((y,z))
                    current_triples.insert(idx, t_remove)
                    used_pairs.add((a,b))
                    used_pairs.add((a,c))
                    used_pairs.add((b,c))
                    no_improve_count += 1
                    
            elif move_type < 0.85:
                # Move 2: Swap two triples
                if len(current_triples) < 2:
                    temperature *= cooling_rate
                    continue
                    
                idx1 = random.randint(0, len(current_triples) - 1)
                idx2 = random.randint(0, len(current_triples) - 1)
                if idx1 == idx2:
                    temperature *= cooling_rate
                    continue
                    
                t1, t2 = current_triples[idx1], current_triples[idx2]
                
                # Try to find replacements for both
                # Remove both
                a1, b1, c1 = t1
                a2, b2, c2 = t2
                used_pairs.discard((a1,b1))
                used_pairs.discard((a1,c1))
                used_pairs.discard((b1,c1))
                used_pairs.discard((a2,b2))
                used_pairs.discard((a2,c2))
                used_pairs.discard((b2,c2))
                
                # Find valid replacement for t1
                valid_for_t1 = []
                for t_new in all_triples:
                    if t_new in (t1, t2):
                        continue
                    pairs_new = triple_pairs[t_new]
                    if not (pairs_new & used_pairs):
                        valid_for_t1.append(t_new)
                
                if not valid_for_t1:
                    # Revert
                    used_pairs.add((a1,b1))
                    used_pairs.add((a1,c1))
                    used_pairs.add((b1,c1))
                    used_pairs.add((a2,b2))
                    used_pairs.add((a2,c2))
                    used_pairs.add((b2,c2))
                    temperature *= cooling_rate
                    continue
                
                t_new1 = random.choice(valid_for_t1)
                x1, y1, z1 = t_new1
                used_pairs.add((x1,y1))
                used_pairs.add((x1,z1))
                used_pairs.add((y1,z1))
                
                # Find valid replacement for t2
                valid_for_t2 = []
                for t_new in all_triples:
                    if t_new in (t1, t2, t_new1):
                        continue
                    pairs_new = triple_pairs[t_new]
                    if not (pairs_new & used_pairs):
                        valid_for_t2.append(t_new)
                
                if not valid_for_t2:
                    # Revert
                    used_pairs.discard((x1,y1))
                    used_pairs.discard((x1,z1))
                    used_pairs.discard((y1,z1))
                    used_pairs.add((a1,b1))
                    used_pairs.add((a1,c1))
                    used_pairs.add((b1,c1))
                    used_pairs.add((a2,b2))
                    used_pairs.add((a2,c2))
                    used_pairs.add((b2,c2))
                    temperature *= cooling_rate
                    continue
                
                t_new2 = random.choice(valid_for_t2)
                x2, y2, z2 = t_new2
                used_pairs.add((x2,y2))
                used_pairs.add((x2,z2))
                used_pairs.add((y2,z2))
                
                # Update triples list
                current_triples[idx1] = t_new1
                current_triples[idx2] = t_new2
                
                new_alpha = compute_alpha(current_triples)
                
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_alpha = new_alpha
                    no_improve_count = 0
                else:
                    # Revert
                    current_triples[idx1] = t1
                    current_triples[idx2] = t2
                    used_pairs.discard((x1,y1))
                    used_pairs.discard((x1,z1))
                    used_pairs.discard((y1,z1))
                    used_pairs.discard((x2,y2))
                    used_pairs.discard((x2,z2))
                    used_pairs.discard((y2,z2))
                    used_pairs.add((a1,b1))
                    used_pairs.add((a1,c1))
                    used_pairs.add((b1,c1))
                    used_pairs.add((a2,b2))
                    used_pairs.add((a2,c2))
                    used_pairs.add((b2,c2))
                    no_improve_count += 1
                    
            else:
                # Move 3: Just add a triple if possible
                valid_candidates = []
                for t_new in all_triples:
                    if t_new in current_triples:
                        continue
                    pairs_new = triple_pairs[t_new]
                    if not (pairs_new & used_pairs):
                        valid_candidates.append(t_new)
                
                if not valid_candidates:
                    temperature *= cooling_rate
                    continue
                
                t_new = random.choice(valid_candidates)
                current_triples.append(t_new)
                x, y, z = t_new
                used_pairs.add((x,y))
                used_pairs.add((x,z))
                used_pairs.add((y,z))
                
                new_alpha = compute_alpha(current_triples)
                
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_alpha = new_alpha
                    no_improve_count = 0
                else:
                    # Revert
                    current_triples.pop()
                    used_pairs.discard((x,y))
                    used_pairs.discard((x,z))
                    used_pairs.discard((y,z))
                    no_improve_count += 1
            
            temperature *= cooling_rate
        
        return current_triples, current_alpha
    
    # Multi-start local search
    strategies = ['random', 'spread', 'clique']
    remaining_time = deadline - time.time()
    time_per_start = remaining_time / (len(strategies) + 2)  # +2 for extra random starts
    
    for i, strategy in enumerate(strategies):
        if time.time() >= deadline:
            break
            
        initial_triples = generate_greedy_solution(strategy)
        if not initial_triples:
            continue
            
        time_limit = min(deadline - 0.5, time.time() + time_per_start)
        improved_triples, alpha = local_search(initial_triples, time_limit)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved_triples[:]
    
    # Try a couple more random starts with longer time each
    for i in range(2):
        if time.time() >= deadline:
            break
            
        initial_triples = generate_greedy_solution('random')
        if not initial_triples:
            continue
            
        time_limit = deadline - 0.5
        improved_triples, alpha = local_search(initial_triples, time_limit)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved_triples[:]
    
    return best_triples
# EVOLVE-BLOCK-END
