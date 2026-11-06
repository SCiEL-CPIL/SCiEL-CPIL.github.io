import random
import time

def compute_alpha(triples):
    # triples is a list of tuples (a, b, c)
    # Returns the size of the largest independent set
    n = 16
    # is_independent[mask] is True if the set of points in mask is independent
    is_ind = [True] * (1 << n)
    
    # For each triple, mark all supersets as not independent
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        # Iterate all supersets of mask
        # Supersets are formed by adding any subset of the remaining 13 bits
        # remaining_bits = ((1 << n) - 1) ^ mask
        # This loop is 2^(n-3) = 8192 iterations per triple
        # This is efficient enough
        remaining = ((1 << n) - 1) ^ mask
        sub = remaining
        while True:
            is_ind[mask | sub] = False
            if sub == 0:
                break
            sub = (sub - 1) & remaining
            
    # Find max popcount of independent sets
    max_alpha = 0
    for mask in range(1 << n):
        if is_ind[mask]:
            pop = bin(mask).count('1')
            if pop > max_alpha:
                max_alpha = pop
    return max_alpha

def random_maximal_system():
    used_pairs = set()
    triples = []
    # Generate all possible triples
    all_triples = [(i, j, k) for i in range(16) for j in range(i+1, 16) for k in range(j+1, 16)]
    random.shuffle(all_triples)
    
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
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

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_triples = None
    best_alpha = 11 # Seed value
    
    # Phase 1: Random Initialization
    # Try a few random systems
    num_initial = 20
    for _ in range(num_initial):
        if time.time() > deadline:
            break
        triples = random_maximal_system()
        alpha = compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
    
    # Phase 2: Local Search on best candidates
    # We keep the best one and search from it
    # Or keep a few? Let's just search from the best found so far.
    if best_triples:
        current_triples = best_triples
        current_alpha = best_alpha
        
        # Local search loop
        steps = 0
        while time.time() < deadline:
            steps += 1
            # Generate a move
            # Pick a random triple to remove
            idx = random.randint(0, len(current_triples) - 1)
            old_triple = current_triples[idx]
            
            # Remove it
            new_triples = current_triples[:idx] + current_triples[idx+1:]
            
            # Find a valid new triple to add
            # We need a triple that doesn't share a pair with any in new_triples
            # We can try random triples until we find one
            # Or precompute available pairs?
            # Let's just try a few random triples
            found_new = False
            for _ in range(100): # Try 100 random triples
                i = random.randint(0, 15)
                j = random.randint(0, 15)
                k = random.randint(0, 15)
                if i == j or j == k or i == k:
                    continue
                t_new = tuple(sorted((i, j, k)))
                
                pairs = [(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]
                valid = True
                # Check against new_triples
                # We need to check if any pair in t_new is in any triple in new_triples
                # This is slow if done naively.
                # Let's maintain a set of used pairs for new_triples?
                # Since we only removed one triple, we can update the set.
                pass
            
            # Optimized Local Search Move Generation
            # Let's rebuild the used_pairs set for the current state?
            # That's O(M) where M is num triples.
            # M is ~40. So it's fast.
            
            used_pairs_curr = set()
            for t in current_triples:
                used_pairs_curr.add((t[0], t[1]))
                used_pairs_curr.add((t[0], t[2]))
                used_pairs_curr.add((t[1], t[2]))
            
            # Remove old triple's pairs
            used_pairs_curr.discard((old_triple[0], old_triple[1]))
            used_pairs_curr.discard((old_triple[0], old_triple[2]))
            used_pairs_curr.discard((old_triple[1], old_triple[2]))
            
            # Find a new triple
            new_triple = None
            for _ in range(50):
                i = random.randint(0, 15)
                j = random.randint(0, 15)
                k = random.randint(0, 15)
                if i == j or j == k or i == k:
                    continue
                t_new = tuple(sorted((i, j, k)))
                pairs = [(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]
                valid = True
                for p in pairs:
                    if p in used_pairs_curr:
                        valid = False
                        break
                if valid:
                    new_triple = t_new
                    break
            
            if new_triple is None:
                continue
                
            # Construct new list
            candidate_triples = new_triples + [new_triple]
            
            # Evaluate
            candidate_alpha = compute_alpha(candidate_triples)
            
            # Acceptance criterion
            if candidate_alpha < current_alpha:
                current_triples = candidate_triples
                current_alpha = candidate_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples
            elif candidate_alpha == current_alpha:
                # Accept to explore
                if random.random() < 0.5:
                    current_triples = candidate_triples
            else:
                # Reject
                pass
                
    if best_triples is None:
        # Fallback to seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return [list(t) for t in best_triples]
