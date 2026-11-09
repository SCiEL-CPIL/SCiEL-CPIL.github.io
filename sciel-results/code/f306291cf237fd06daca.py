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
    
    best_alpha = n
    best_triples = []
    
    # Precompute all possible triples and their masks
    all_triples = list(itertools.combinations(range(n), 3))
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in all_triples]
    
    # Precompute masks for all subsets of size >= 8 (since alpha is likely around 8)
    # We can check sizes from 15 down to 0.
    # To speed up alpha calculation, we precompute lists of masks for each size.
    masks_by_size = {}
    for size in range(16, -1, -1):
        masks_by_size[size] = [sum(1 << p for p in comb) for comb in itertools.combinations(range(n), size)]

    def get_alpha(triples):
        if not triples:
            return n
            
        # Create a set of masks for fast lookup
        t_masks = set()
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            t_masks.add(mask)
            
        # Check sizes from high to low
        for size in range(n, 0, -1):
            for mask in masks_by_size[size]:
                # Check if any triple is a subset of mask
                # Optimization: if we just need to know if ANY triple is contained
                # We can iterate t_masks and check (t & mask) == t
                # This is O(|triples|) per mask.
                # To speed up, we can break early if found.
                independent = True
                for t in t_masks:
                    if (t & mask) == t:
                        independent = False
                        break
                if independent:
                    return size
        return 0

    def try_greedy_packing():
        """
        Deterministic greedy approach to find a large linear 3-uniform hypergraph.
        Prioritizes triples that cover points with low degree to balance the hypergraph.
        """
        triples = []
        used_pairs = set()
        point_degree = [0] * n
        
        # Shuffle all_triples to introduce some randomness but keep the greedy logic
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
                point_degree[a] += 1
                point_degree[b] += 1
                point_degree[c] += 1
                    
        return triples

    # Initial greedy solution
    current_triples = try_greedy_packing()
    current_alpha = get_alpha(current_triples)
    best_alpha = current_alpha
    best_triples = current_triples[:]
    
    # Simulated Annealing
    # We will try to swap one triple with another, or two triples with two others.
    
    # Let's maintain a set of used pairs for the current solution.
    current_pairs = set()
    for t in current_triples:
        a, b, c = t
        current_pairs.add((a,b))
        current_pairs.add((a,c))
        current_pairs.add((b,c))
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if temperature < 0.001:
            break
            
        if not current_triples:
            break
            
        # Choose move type: 1-swap (70%) or 2-swap (30%)
        move_type = random.choice([1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        
        if move_type == 1:
            # 1-swap
            idx = random.randint(0, len(current_triples) - 1)
            t_remove = current_triples[idx]
            
            # Remove it from current state
            current_triples.pop(idx)
            
            # Remove its pairs from used_pairs
            a, b, c = t_remove
            current_pairs.discard((a,b))
            current_pairs.discard((a,c))
            current_pairs.discard((b,c))
            
            # Find a valid triple to add
            candidate_found = False
            candidates = all_triples[:]
            random.shuffle(candidates)
            
            for t_new in candidates:
                x, y, z = t_new
                pairs_new = [(x,y), (x,z), (y,z)]
                valid = True
                for p in pairs_new:
                    if p in current_pairs:
                        valid = False
                        break
                if valid:
                    # Accept move
                    current_triples.append(t_new)
                    for p in pairs_new:
                        current_pairs.add(p)
                    candidate_found = True
                    break
                    
            if not candidate_found:
                # Revert
                current_triples.insert(idx, t_remove)
                current_pairs.add((a,b))
                current_pairs.add((a,c))
                current_pairs.add((b,c))
                temperature *= cooling_rate
                continue
            
            # Evaluate new alpha
            new_alpha = get_alpha(current_triples)
            
            # Metropolis criterion
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
                    if best_alpha <= 7:
                        break
            else:
                delta = new_alpha - current_alpha
                if delta == 0:
                    if random.random() < 0.5:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        current_triples.pop()
                        for p in pairs_new:
                            current_pairs.discard(p)
                        current_triples.insert(idx, t_remove)
                        current_pairs.add((a,b))
                        current_pairs.add((a,c))
                        current_pairs.add((b,c))
                else:
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        current_triples.pop()
                        for p in pairs_new:
                            current_pairs.discard(p)
                        current_triples.insert(idx, t_remove)
                        current_pairs.add((a,b))
                        current_pairs.add((a,c))
                        current_pairs.add((b,c))
                        
        else:
            # 2-swap
            if len(current_triples) < 2:
                temperature *= cooling_rate
                continue
                
            idx1, idx2 = random.sample(range(len(current_triples)), 2)
            # Ensure idx1 < idx2 for consistent popping
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
                
            t_remove1 = current_triples[idx1]
            t_remove2 = current_triples[idx2]
            
            # Remove them
            # Pop the higher index first to keep indices valid
            current_triples.pop(idx2)
            current_triples.pop(idx1)
            
            a1, b1, c1 = t_remove1
            a2, b2, c2 = t_remove2
            
            current_pairs.discard((a1,b1))
            current_pairs.discard((a1,c1))
            current_pairs.discard((b1,c1))
            current_pairs.discard((a2,b2))
            current_pairs.discard((a2,c2))
            current_pairs.discard((b2,c2))
            
            # Find two valid triples to add
            candidates = all_triples[:]
            random.shuffle(candidates)
            
            t_new1 = None
            t_new2 = None
            pairs_new1 = []
            pairs_new2 = []
            
            # Try to find first
            for t1 in candidates:
                x, y, z = t1
                p1 = [(x,y), (x,z), (y,z)]
                valid = True
                for p in p1:
                    if p in current_pairs:
                        valid = False
                        break
                if valid:
                    t_new1 = t1
                    pairs_new1 = p1
                    # Temporarily add to check for second
                    for p in p1:
                        current_pairs.add(p)
                    break
            
            if t_new1 is None:
                # Revert
                current_triples.insert(idx1, t_remove1)
                current_triples.insert(idx2, t_remove2)
                for p in [(a1,b1), (a1,c1), (b1,c1), (a2,b2), (a2,c2), (b2,c2)]:
                    current_pairs.add(p)
                temperature *= cooling_rate
                continue
            
            # Try to find second
            for t2 in candidates:
                if t2 == t_new1:
                    continue
                x, y, z = t2
                p2 = [(x,y), (x,z), (y,z)]
                valid = True
                for p in p2:
                    if p in current_pairs:
                        valid = False
                        break
                if valid:
                    t_new2 = t2
                    pairs_new2 = p2
                    break
            
            if t_new2 is None:
                # Revert
                for p in pairs_new1:
                    current_pairs.discard(p)
                current_triples.insert(idx1, t_remove1)
                current_triples.insert(idx2, t_remove2)
                for p in [(a1,b1), (a1,c1), (b1,c1), (a2,b2), (a2,c2), (b2,c2)]:
                    current_pairs.add(p)
                temperature *= cooling_rate
                continue
            
            # Apply second
            for p in pairs_new2:
                current_pairs.add(p)
            current_triples.append(t_new1)
            current_triples.append(t_new2)
            
            new_alpha = get_alpha(current_triples)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
                    if best_alpha <= 7:
                        break
            else:
                delta = new_alpha - current_alpha
                if delta == 0:
                    if random.random() < 0.5:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        current_triples.pop()
                        current_triples.pop()
                        for p in pairs_new1 + pairs_new2:
                            current_pairs.discard(p)
                        current_triples.insert(idx1, t_remove1)
                        current_triples.insert(idx2, t_remove2)
                        for p in [(a1,b1), (a1,c1), (b1,c1), (a2,b2), (a2,c2), (b2,c2)]:
                            current_pairs.add(p)
                else:
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        current_triples.pop()
                        current_triples.pop()
                        for p in pairs_new1 + pairs_new2:
                            current_pairs.discard(p)
                        current_triples.insert(idx1, t_remove1)
                        current_triples.insert(idx2, t_remove2)
                        for p in [(a1,b1), (a1,c1), (b1,c1), (a2,b2), (a2,c2), (b2,c2)]:
                            current_pairs.add(p)
                            
        temperature *= cooling_rate

    return best_triples
# EVOLVE-BLOCK-END
