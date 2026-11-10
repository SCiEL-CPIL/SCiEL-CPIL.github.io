import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s

    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    
    # Precompute pair keys for each triple
    triple_pairs = {}
    for t in all_triples:
        pairs = set()
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pairs.add(key)
        triple_pairs[t] = pairs

    # For each pair, the unique triple containing it (in the full set of all triples)
    # This is just for reference, not directly used for validity in partial systems
    # Validity is checked via used_pairs set in the search.

    def get_alpha_and_set(triples):
        """
        Exact independence number and the specific independent set via DP over bitmasks.
        Returns (alpha, independent_set_mask).
        """
        if not triples:
            return n, (1 << n) - 1

        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)
        
        # DP: We want to find the mask with max popcount such that no triple is fully contained.
        # We can iterate through all masks. 2^16 = 65536 is small.
        # Optimization: Precompute popcount for all masks? Or just use bin.count('1').
        # bin.count('1') is fast enough for 65k iterations.
        
        best_alpha = 0
        best_mask = 0
        
        # We can prune: if a mask contains a triple, it's invalid.
        # We iterate all masks.
        for mask in range(1 << n):
            # Check if mask is independent
            is_independent = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                cnt = bin(mask).count('1')
                if cnt > best_alpha:
                    best_alpha = cnt
                    best_mask = mask
                    # Early exit if we found a full set? No, we want max.
                    # If best_alpha == n, we can break, but that's rare for non-empty triples.
        
        return best_alpha, best_mask

    def is_valid_addition(triple, used_pairs):
        """Check if triple can be added without violating linearity."""
        for p in triple_pairs[triple]:
            if p in used_pairs:
                return False
        return True

    def greedy_construct():
        triples = []
        used_pairs = set()
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for t in shuffled:
            if is_valid_addition(t, used_pairs):
                triples.append(t)
                used_pairs.update(triple_pairs[t])
        return triples

    def local_search(triples, deadline):
        """
        Local search with exact alpha evaluation.
        Uses the exact independent set to guide moves (targeted hits).
        """
        if not triples:
            return triples, n, (1 << n) - 1

        current_alpha, current_imask = get_alpha_and_set(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        best_imask = current_imask

        used_pairs = set()
        for t in triples:
            used_pairs.update(triple_pairs[t])

        current = triples[:]
        cur_alpha = current_alpha
        cur_imask = current_imask

        # Precompute candidates that can be added to the *initial* set
        # Note: as we swap, the set of valid candidates changes.
        # We will recompute or filter on the fly.
        
        max_steps = 2000
        temperature = 1.0
        cooling = 0.995

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if step % 100 == 0:
                temperature *= cooling

            # Identify candidates that can be added to current
            candidates = []
            for t in all_triples:
                if t in current:
                    continue
                if is_valid_addition(t, used_pairs):
                    candidates.append(t)

            if not candidates:
                break

            move_type = random.random()
            
            new_triples = None
            new_used_pairs = None

            if move_type < 0.5:
                # Move 1: Add a triple that hits the current independent set
                # This is the most likely move to reduce alpha.
                # Filter candidates by overlap with cur_imask
                targeted = []
                for t in candidates:
                    t_mask = 0
                    for p in t:
                        t_mask |= (1 << p)
                    overlap = bin(t_mask & cur_imask).count('1')
                    if overlap >= 2:
                        targeted.append(t)
                
                if targeted:
                    added = random.choice(targeted)
                else:
                    added = random.choice(candidates)
                
                new_triples = current + [added]
                new_used_pairs = used_pairs | triple_pairs[added]

            else:
                # Move 2: Swap (remove one, add one)
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                
                # Calculate used_pairs without 'removed'
                new_used_pairs = set()
                for t in current:
                    if t != removed:
                        new_used_pairs.update(triple_pairs[t])
                
                # Find valid candidates for this new state
                valid_cands = []
                for t in candidates:
                    if t == removed:
                        continue
                    if is_valid_addition(t, new_used_pairs):
                        valid_cands.append(t)
                
                if not valid_cands:
                    continue
                
                added = random.choice(valid_cands)
                new_triples = current[:idx] + current[idx+1:] + [added]
                new_used_pairs = new_used_pairs | triple_pairs[added]

            if new_triples is None:
                continue

            new_alpha, new_imask = get_alpha_and_set(new_triples)
            delta = new_alpha - cur_alpha

            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                cur_imask = new_imask
                used_pairs = new_used_pairs
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
                    best_imask = cur_imask
            else:
                # Accept worse with probability exp(-delta / temperature)
                if temperature > 0.01:
                    prob = pow(2.718281828, -delta / max(temperature, 0.01))
                    if random.random() < prob:
                        current = new_triples
                        cur_alpha = new_alpha
                        cur_imask = new_imask
                        used_pairs = new_used_pairs

            if best_alpha <= 4:
                break

        return best_triples, best_alpha, best_imask

    # Main
    best_triples = []
    best_alpha = n
    best_imask = (1 << n) - 1

    # Multiple restarts
    # Since get_alpha is very fast (exact DP), we can afford many restarts.
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        triples = greedy_construct()
        if not triples:
            continue

        triples, alpha, imask = local_search(triples, deadline - 5)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            best_imask = imask

        if best_alpha <= 4:
            break

    # Final polish: try to add triples to the best solution found
    if time.time() < deadline - 2 and best_triples:
        used_pairs = set()
        for t in best_triples:
            used_pairs.update(triple_pairs[t])
        
        candidates = []
        for t in all_triples:
            if t in best_triples:
                continue
            if is_valid_addition(t, used_pairs):
                candidates.append(t)
        
        if candidates:
            # Sort candidates by potential to reduce alpha (overlap with best_imask)
            scored_cands = []
            for t in candidates:
                t_mask = 0
                for p in t:
                    t_mask |= (1 << p)
                overlap = bin(t_mask & best_imask).count('1')
                scored_cands.append((overlap, t))
            
            scored_cands.sort(key=lambda x: -x[0])
            
            for overlap, t in scored_cands:
                if time.time() > deadline - 2:
                    break
                if overlap < 2:
                    break # If we can't hit the IS with 2 points, adding is less likely to help
                
                new_triples = best_triples + [t]
                new_alpha, new_imask = get_alpha_and_set(new_triples)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples
                    best_imask = new_imask
                    used_pairs.update(triple_pairs[t])
                    # Re-evaluate candidates? Just break and let next loop handle it if we had one, 
                    # but this is a single pass polish.
                    break

    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    used_pairs = set()
    valid = True
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print(f"INVALID: Shared pair {key} in {t}")
                    valid = False
                used_pairs.add(key)
    if valid:
        print("Valid linear system")
