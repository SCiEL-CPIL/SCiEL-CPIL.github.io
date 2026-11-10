import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    
    # Precompute all triples and their masks
    all_triples = list(combinations(points, 3))
    triple_masks = []
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    # Precompute pair keys for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = set()
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pairs.add(key)
        triple_pairs.append(pairs)
    
    # Precompute compatibility: which triples can coexist (share at most 1 point)
    # Use bitmasks for fast intersection check
    # Two triples are compatible if their intersection has size <= 1
    # i.e., popcount(mask_i & mask_j) <= 1
    
    def popcount(x):
        return bin(x).count('1')
    
    # Precompute intersection sizes between all triple pairs
    # This is O(C(16,3)^2) = O(560^2) = 313600, which is fine
    # Store as a bitset: compat[i] is a bitmask of triples j that are compatible with i
    num_triples = len(all_triples)
    compat = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        for j in range(num_triples):
            if i == j:
                continue
            mj = triple_masks[j]
            inter = mi & mj
            if popcount(inter) <= 1:
                compat[i] |= (1 << j)
    
    def calc_alpha_fast(triple_indices):
        """Compute independence number using bitmask DP.
        Returns the size of the largest subset of points containing no triple."""
        if not triple_indices:
            return n
        
        # Build set of triple masks
        t_masks = [triple_masks[i] for i in triple_indices]
        num_t = len(t_masks)
        
        # DP over all 2^16 masks
        # independent[mask] = True if mask contains no triple
        # We want max popcount(mask) where independent[mask] = True
        
        # Optimization: iterate masks in increasing order of popcount,
        # and stop early if we find a large independent set
        
        # Actually, let's just iterate all masks and check
        # 2^16 = 65536 masks, each check is O(num_t)
        
        max_ind = 0
        for mask in range(1 << n):
            # Quick check: if popcount(mask) <= max_ind, skip
            cnt = bin(mask).count('1')
            if cnt <= max_ind:
                continue
            
            # Check if mask is independent
            independent = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                max_ind = cnt
                if max_ind == n:
                    break
        
        return max_ind
    
    def calc_alpha_branch_and_bound(triple_indices, time_limit=None):
        """Branch and bound to find independence number."""
        if not triple_indices:
            return n
        
        t_masks = [triple_masks[i] for i in triple_indices]
        num_t = len(t_masks)
        
        # Sort points by degree (number of triples containing them)
        point_deg = [0] * n
        for tm in t_masks:
            for p in range(n):
                if tm & (1 << p):
                    point_deg[p] += 1
        
        # Try to find a large independent set
        # Start with all points, try to remove minimum to hit all triples
        
        best_size = 0
        # Greedy: start with all points, iteratively remove points from triples
        # But we want the LARGEST independent set, so we want to KEEP as many as possible
        
        # Alternative: branch and bound
        # State: (remaining_triples_to_hit, current_independent_set)
        # But this is complex. Let's use a simpler approach:
        # Iterate subsets in decreasing order of size, check if independent
        
        # Actually, the DP approach above is already O(2^16 * num_t) which is fast enough
        # Let's just use that
        
        return calc_alpha_fast(triple_indices)
    
    def greedy_clique_expand(start_triple_idx, deadline):
        """Greedy clique expansion: start with one triple, add compatible triples."""
        current = [start_triple_idx]
        current_mask = triple_masks[start_triple_idx]
        
        # Track which triples are still candidates
        candidates = set()
        for j in range(num_triples):
            if j != start_triple_idx and (compat[start_triple_idx] >> j) & 1:
                candidates.add(j)
        
        while candidates:
            if time.time() > deadline:
                break
            
            # Pick a random candidate
            cand = random.choice(list(candidates))
            current.append(cand)
            current_mask |= triple_masks[cand]
            candidates.discard(cand)
            
            # Remove incompatible triples from candidates
            to_remove = set()
            for j in candidates:
                if not ((compat[cand] >> j) & 1):
                    to_remove.add(j)
            candidates -= to_remove
        
        return current
    
    def local_search(triples, deadline):
        """Local search to improve independence number."""
        current = triples[:]
        cur_alpha = calc_alpha_fast(current)
        best_alpha = cur_alpha
        best_triples = current[:]
        
        temperature = 1.0
        cooling = 0.998
        
        max_steps = 2000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if step % 50 == 0:
                temperature *= cooling
            
            # Pick a random triple to remove
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            
            # Find compatible triples to add
            new_current = current[:idx] + current[idx+1:]
            
            # Find candidates compatible with all in new_current
            valid_cands = []
            for j in range(num_triples):
                if j in new_current:
                    continue
                # Check if j is compatible with all triples in new_current
                compatible = True
                for k in new_current:
                    if not ((compat[j] >> k) & 1):
                        compatible = False
                        break
                if compatible:
                    valid_cands.append(j)
            
            if not valid_cands:
                continue
            
            added = random.choice(valid_cands)
            new_triples = new_current + [added]
            new_alpha = calc_alpha_fast(new_triples)
            
            # Accept if better, or with probability exp(-delta/temperature)
            delta = new_alpha - cur_alpha
            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
            else:
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current = new_triples
                    cur_alpha = new_alpha
        
        return best_triples, best_alpha
    
    def targeted_attack(triples, deadline):
        """Targeted move: find the independent set and try to hit it."""
        current = triples[:]
        cur_alpha = calc_alpha_fast(current)
        best_alpha = cur_alpha
        best_triples = current[:]
        
        # Find a maximum independent set
        # We need to find the actual set, not just its size
        t_masks = [triple_masks[i] for i in current]
        
        # Find max independent set by iterating masks
        best_mask = 0
        best_size = 0
        for mask in range(1 << n):
            cnt = bin(mask).count('1')
            if cnt <= best_size:
                continue
            independent = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                best_size = cnt
                best_mask = mask
        
        # The independent set is best_mask
        # Try to add triples that intersect this set
        # But we need to maintain linearity
        
        # Find triples that hit the independent set (intersect it)
        # and are compatible with current system
        ind_points = [p for p in range(n) if best_mask & (1 << p)]
        
        hit_triples = []
        for j in range(num_triples):
            if j in current:
                continue
            # Check if triple j hits the independent set
            tm = triple_masks[j]
            if tm & best_mask:
                # Check compatibility with all current triples
                compatible = True
                for k in current:
                    if not ((compat[j] >> k) & 1):
                        compatible = False
                        break
                if compatible:
                    hit_triples.append(j)
        
        # Try swapping: remove a triple that doesn't hit the independent set,
        # add one that does
        if not hit_triples:
            return best_triples, best_alpha
        
        # Try multiple swap attempts
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Find a triple in current that doesn't hit the independent set
            non_hit = []
            for k in current:
                if not (triple_masks[k] & best_mask):
                    non_hit.append(k)
            
            if not non_hit:
                break
            
            remove_idx = random.choice(non_hit)
            remove_val = current[remove_idx]
            add_val = random.choice(hit_triples)
            
            new_triples = current[:remove_idx] + current[remove_idx+1:] + [add_val]
            new_alpha = calc_alpha_fast(new_triples)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = new_triples[:]
                current = new_triples[:]
                cur_alpha = new_alpha
        
        return best_triples, best_alpha
    
    # Main: multiple random restarts
    best_triples = []
    best_alpha = n
    
    # Strategy 1: Greedy clique expansion from random starts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        start_idx = random.randint(0, num_triples - 1)
        triples = greedy_clique_expand(start_idx, deadline - 5)
        
        if not triples:
            continue
        
        # Local search
        triples, alpha = local_search(triples, deadline - 5)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        if best_alpha <= 4:
            break
    
    # Strategy 2: Targeted attack on best found so far
    if best_triples and time.time() < deadline - 5:
        best_triples, best_alpha = targeted_attack(best_triples, deadline - 2)
    
    # Strategy 3: More restarts with different strategies
    while time.time() < deadline - 5:
        start_idx = random.randint(0, num_triples - 1)
        triples = greedy_clique_expand(start_idx, deadline - 2)
        
        if not triples:
            continue
        
        triples, alpha = local_search(triples, deadline - 2)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        if best_alpha <= 4:
            break
    
    # Final targeted attack
    if best_triples and time.time() < deadline - 1:
        best_triples, best_alpha = targeted_attack(best_triples, deadline - 0.5)
    
    # Convert to list of lists
    return [list(all_triples[i]) for i in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    # Verify linearity
    used_pairs = set()
    valid = True
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print("INVALID: Shared pair", t)
                    valid = False
                    break
                used_pairs.add(key)
        if not valid:
            break
    if valid:
        print("Valid linear system")
