import random
import time
import math

_START = time.time()
_LIMIT = 110.0  # Safety margin

def _is_valid_triple_set(triples):
    """Check if the set of triples is linear (any two share at most 1 point)."""
    n = len(triples)
    for i in range(n):
        a, b, c = triples[i]
        for j in range(i + 1, n):
            x, y, z = triples[j]
            inter = 0
            if a == x or a == y or a == z: inter += 1
            if b == x or b == y or b == z: inter += 1
            if c == x or c == y or c == z: inter += 1
            if inter > 1:
                return False
    return True

def _compute_alpha(triples):
    """
    Compute the independence number of the hypergraph defined by triples.
    Uses bitmask counting.
    """
    # cnt[mask] = number of triples fully contained in mask
    cnt = [0] * (1 << 16)
    
    for t in triples:
        a, b, c = t
        triple_mask = (1 << a) | (1 << b) | (1 << c)
        # Iterate all supersets of triple_mask
        # Supersets are of the form triple_mask | sub_mask where sub_mask is a subset of complement
        comp_mask = ((1 << 16) - 1) ^ triple_mask
        sub = comp_mask
        while True:
            mask = triple_mask | sub
            cnt[mask] += 1
            if sub == 0:
                break
            sub = (sub - 1) & comp_mask
            
    # Find max size of mask with cnt[mask] == 0
    # Precompute popcount for all masks? Or just calculate.
    # 2^16 is small.
    max_alpha = 0
    for mask in range(1 << 16):
        if cnt[mask] == 0:
            size = bin(mask).count('1')
            if size > max_alpha:
                max_alpha = size
    return max_alpha

def construct():
    rng = random.Random(42)
    
    # Precompute all possible triples
    all_triples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                all_triples.append((i, j, k))
                
    # Precompute pair masks for fast conflict checking?
    # Or just use sets of pairs.
    
    def get_used_pairs(triples):
        pairs = set()
        for a, b, c in triples:
            pairs.add((a, b))
            pairs.add((a, c))
            pairs.add((b, c))
        return pairs

    def is_conflict(triple, used_pairs):
        a, b, c = triple
        if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
            return True
        return False

    # 1. Greedy Initialization
    def greedy_init():
        triples = []
        used_pairs = set()
        remaining = all_triples[:]
        rng.shuffle(remaining)
        
        # Bias towards covering points uniformly?
        # Simple random greedy first
        for t in remaining:
            if not is_conflict(t, used_pairs):
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                
        return triples

    # 2. Simulated Annealing / Local Search
    # We want to minimize alpha.
    # Moves:
    # - Add a triple (if valid)
    # - Remove a triple
    # - Swap: Remove T1, Add T2 (if valid)
    # - 2-Swap: Remove T1, T2, Add T3, T4 (if valid) - might be too slow/expensive to generate candidates
    
    # Let's stick to Add, Remove, Swap.
    
    def simulate_annealing(init_triples, time_budget):
        current_triples = list(init_triples)
        current_alpha = _compute_alpha(current_triples)
        best_triples = list(current_triples)
        best_alpha = current_alpha
        
        T = 1.0
        cooling = 0.999
        start_time = time.time()
        
        # Pre-compute list of all triples for random selection
        # We need to efficiently find valid adds.
        # Just pick random triples from all_triples and check validity.
        
        while (time.time() - start_time) < time_budget:
            # Generate a move
            move_type = rng.random()
            new_triples = None
            new_alpha = None
            delta = 0
            
            if move_type < 0.3:
                # Add a triple
                # Pick random triple
                t = rng.choice(all_triples)
                # Check if already in list?
                if t in current_triples:
                    continue
                # Check conflict
                used_pairs = get_used_pairs(current_triples)
                if not is_conflict(t, used_pairs):
                    new_triples = current_triples + [t]
                    new_alpha = _compute_alpha(new_triples)
                    delta = new_alpha - current_alpha
            elif move_type < 0.6:
                # Remove a triple
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                new_triples = current_triples[:idx] + current_triples[idx+1:]
                new_alpha = _compute_alpha(new_triples)
                delta = new_alpha - current_alpha
            else:
                # Swap: Remove one, Add one
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                t_remove = current_triples[idx]
                # Pick a random t_add
                t_add = rng.choice(all_triples)
                if t_add == t_remove:
                    continue
                # Check if t_add is already in the list (excluding t_remove)
                if t_add in current_triples:
                    continue
                
                # Tentatively remove t_remove
                temp_triples = current_triples[:idx] + current_triples[idx+1:]
                used_pairs = get_used_pairs(temp_triples)
                
                if not is_conflict(t_add, used_pairs):
                    new_triples = temp_triples + [t_add]
                    new_alpha = _compute_alpha(new_triples)
                    delta = new_alpha - current_alpha

            if new_triples is None:
                continue

            # Acceptance criterion
            if delta < 0:
                # Improvement
                current_triples = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            else:
                # Worse or same
                if T > 0:
                    # Accept with probability exp(-delta / T)
                    # delta >= 0
                    if delta == 0:
                        # Accept neutral moves to explore
                        if rng.random() < 0.5:
                            current_triples = new_triples
                            current_alpha = new_alpha
                    else:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            current_triples = new_triples
                            current_alpha = new_alpha
            
            T *= cooling
            if T < 0.01:
                T = 0.01 # Floor

        return best_triples, best_alpha

    # Run multiple restarts
    best_global_alpha = 16
    best_global_triples = None
    
    start_time = time.time()
    restarts = 0
    while (time.time() - start_time) < (_LIMIT - 10):
        init_triples = greedy_init()
        if not init_triples:
            continue
        
        # Budget per restart
        budget = min(10.0, max(1.0, _LIMIT - 15 - (time.time() - start_time)))
        
        opt_triples, alpha = simulate_annealing(init_triples, budget)
        
        if alpha < best_global_alpha:
            best_global_alpha = alpha
            best_global_triples = opt_triples
            
        restarts += 1
        if best_global_alpha <= 5: # Theoretical lower bound is likely 5 or 6
            break

    if best_global_triples is None:
        # Fallback
        best_global_triples = greedy_init()
        # Ensure it's valid
        if not _is_valid_triple_set(best_global_triples):
             # Very unlikely, but just in case
             best_global_triples = []
             
    return best_global_triples

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
    print(f"Alpha: {_compute_alpha(res)}")
