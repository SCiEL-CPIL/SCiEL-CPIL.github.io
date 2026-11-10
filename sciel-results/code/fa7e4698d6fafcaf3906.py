"""Construct a linear triple system on 16 points minimizing independence number."""
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    rng = random.Random(12345)
    
    # Precompute all C(16,3) = 560 triples
    all_triples = []
    triple_masks = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))
                triple_masks.append((1 << a) | (1 << b) | (1 << c))
    
    n_triples = len(all_triples)  # 560
    
    # Precompute compatibility: for each triple i, bitmask of compatible triples
    # Compatible means intersection size <= 1
    # 560 triples, so we need 560-bit masks -> use Python integers
    compat = [0] * n_triples
    for i in range(n_triples):
        mi = triple_masks[i]
        # Count bits in intersection
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            inter = mi & mj
            # popcount of inter
            cnt = 0
            t = inter
            while t:
                cnt += 1
                t &= t - 1
            if cnt <= 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Fast independence number computation
    # For a set of triples, find largest subset of points with no triple fully inside
    # Use: for each subset S of points, check if any triple is contained in S
    # 2^16 = 65536 subsets, each check against up to 40 triples
    # Precompute: for each triple, which point-subsets contain it
    # Better: for each point-subset, compute if it's "triple-free"
    
    # Even faster: use the dual - a set S is independent iff no triple is subset of S
    # For a given system, precompute a boolean array bad[mask] = True if mask contains a triple
    # Then alpha = max popcount(mask) for mask where bad[mask] = False
    
    def compute_alpha(selected_indices):
        """Compute independence number for a system given by indices into all_triples."""
        if not selected_indices:
            return 16
        
        # Build bad array: bad[mask] = True if some triple is fully contained in mask
        bad = bytearray(1 << 16)
        
        for idx in selected_indices:
            tm = triple_masks[idx]
            # Mark all supersets of tm as bad
            # Supersets: tm | sub where sub is any subset of (~tm) & full_mask
            full_mask = (1 << 16) - 1
            remaining = full_mask ^ tm
            sub = remaining
            while True:
                bad[tm | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        
        # Find max popcount of non-bad mask
        best = 0
        for mask in range(1 << 16):
            if not bad[mask]:
                cnt = bin(mask).count('1')
                if cnt > best:
                    best = cnt
        return best
    
    # Even faster alpha computation using a different approach:
    # Iterate from k=16 down, check if any k-subset is triple-free
    # For a k-subset S, it's triple-free iff no triple mask is a subset of S
    # i.e., for all triples t: (t_mask & S) != t_mask
    
    def compute_alpha_fast(selected_indices):
        """Faster: iterate k from 16 down, check k-subsets."""
        if not selected_indices:
            return 16
        
        tms = [triple_masks[i] for i in selected_indices]
        n_t = len(tms)
        
        for k in range(16, 0, -1):
            # Check if any k-subset is triple-free
            # A k-subset S is triple-free iff for all t in tms: (t & S) != t
            # i.e., S does NOT contain any triple
            # Equivalent: S is not a superset of any t
            # Use: for each k-subset, check all n_t triples
            # C(16,k) subsets, each check O(n_t)
            # For k=16: 1 subset, n_t checks
            # For k=15: 16 subsets
            # For k=14: 120 subsets
            # For k=13: 560 subsets
            # For k=12: 1820 subsets
            # For k=11: 4368 subsets
            # For k=10: 8008 subsets
            # For k=9: 11440 subsets
            # For k=8: 12870 subsets
            # Total: ~40k subsets * ~40 checks = 1.6M ops, fine
            
            for S in combinations(range(16), k):
                S_mask = 0
                for p in S:
                    S_mask |= (1 << p)
                
                triple_free = True
                for tm in tms:
                    if (tm & S_mask) == tm:
                        triple_free = False
                        break
                
                if triple_free:
                    return k
        
        return 0
    
    # Use the faster method for the main search
    # Precompute all C(16,k) subsets for each k
    subsets_by_k = {}
    for k in range(1, 17):
        subsets_by_k[k] = list(combinations(range(16), k))
    
    # Convert to masks for speed
    subsets_masks_by_k = {}
    for k in range(1, 17):
        masks = []
        for S in subsets_by_k[k]:
            m = 0
            for p in S:
                m |= (1 << p)
            masks.append(m)
        subsets_masks_by_k[k] = masks
    
    def compute_alpha_v2(selected_indices):
        """Very fast: precomputed subset masks."""
        if not selected_indices:
            return 16
        
        tms = [triple_masks[i] for i in selected_indices]
        n_t = len(tms)
        
        for k in range(16, 0, -1):
            for S_mask in subsets_masks_by_k[k]:
                triple_free = True
                for tm in tms:
                    if (tm & S_mask) == tm:
                        triple_free = False
                        break
                if triple_free:
                    return k
        return 0
    
    # Simulated annealing
    # State: frozenset of triple indices
    # Energy: alpha (want to minimize)
    
    # Initial solution: greedy construction
    def greedy_initial():
        """Build a maximal linear system greedily."""
        available = (1 << n_triples) - 1
        selected = set()
        
        while available:
            # Pick random available triple
            bits = bin(available).count('1')
            if bits == 0:
                break
            idx_in_bits = rng.randrange(bits)
            tmp = available
            for _ in range(idx_in_bits):
                tmp &= tmp - 1
            bit = tmp & (-tmp)
            i = bit.bit_length() - 1
            
            selected.add(i)
            # Remove incompatible triples
            available &= compat[i]
            available &= ~(1 << i)
        
        return selected
    
    # Try multiple greedy starts and pick best
    best_initial = None
    best_initial_alpha = 16
    
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        sol = greedy_initial()
        alpha = compute_alpha_v2(list(sol))
        if alpha < best_initial_alpha:
            best_initial_alpha = alpha
            best_initial = sol
    
    if best_initial is None:
        best_initial = greedy_initial()
        best_initial_alpha = compute_alpha_v2(list(best_initial))
    
    # Simulated annealing
    current = best_initial
    current_alpha = best_initial_alpha
    best = set(current)
    best_alpha = current_alpha
    
    T_start = 5.0
    T_end = 0.1
    total_iters = 50000
    # Adjust iterations based on available time
    elapsed = time.time() - start_time
    remaining_time = deadline - elapsed - 5  # leave 5s margin
    # Estimate time per iteration
    # We'll just run until deadline
    
    # SA parameters
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(total_iters, 1))
    
    # Precompute for speed: for each triple, the set of points it uses
    triple_points = [set(t) for t in all_triples]
    
    # For move generation, we need to efficiently find:
    # 1. Triples that can be added (compatible with all current)
    # 2. Triples that can be removed
    # 3. Swaps: remove one, add one compatible with rest
    
    # Build current availability mask
    def build_available_mask(selected):
        """Bitmask of triples compatible with all in selected."""
        if not selected:
            return (1 << n_triples) - 1
        avail = (1 << n_triples) - 1
        for i in selected:
            avail &= compat[i]
        # Also remove selected triples themselves
        for i in selected:
            avail &= ~(1 << i)
        return avail
    
    # Fast move: add a triple
    def try_add(selected, avail_mask):
        """Try to add a random triple from available."""
        bits = bin(avail_mask).count('1')
        if bits == 0:
            return None
        idx_in_bits = rng.randrange(bits)
        tmp = avail_mask
        for _ in range(idx_in_bits):
            tmp &= tmp - 1
        bit = tmp & (-tmp)
        return bit.bit_length() - 1
    
    # Run SA
    sa_start = time.time()
    iteration = 0
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5:
        iteration += 1
        
        # Update temperature
        if iteration < total_iters:
            T = T_start * (cooling ** iteration)
        else:
            T = T_end
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4 and current:
            # Remove a triple
            i = rng.choice(list(current))
            new_state = current - {i}
            new_alpha = compute_alpha_v2(list(new_state))
            
            # Energy change
            dE = new_alpha - current_alpha
            
            if dE <= 0 or rng.random() < min(1.0, 2.71828 ** (-dE / max(T, 0.01))):
                current = new_state
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best = set(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.8:
            # Add a triple
            avail = build_available_mask(current)
            i = try_add(current, avail)
            if i is None:
                no_improve += 1
                continue
            new_state = current | {i}
            new_alpha = compute_alpha_v2(list(new_state))
            
            dE = new_alpha - current_alpha
            
            if dE <= 0 or rng.random() < min(1.0, 2.71828 ** (-dE / max(T, 0.01))):
                current = new_state
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best = set(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Swap: remove one, add one
            if not current:
                no_improve += 1
                continue
            i_remove = rng.choice(list(current))
            remaining = current - {i_remove}
            avail = build_available_mask(remaining)
            i_add = try_add(remaining, avail)
            if i_add is None:
                no_improve += 1
                continue
            new_state = remaining | {i_add}
            new_alpha = compute_alpha_v2(list(new_state))
            
            dE = new_alpha - current_alpha
            
            if dE <= 0 or rng.random() < min(1.0, 2.71828 ** (-dE / max(T, 0.01))):
                current = new_state
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best = set(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        # Early termination if stuck
        if no_improve > max_no_improve:
            # Restart with new random seed
            sol = greedy_initial()
            alpha = compute_alpha_v2(list(sol))
            if alpha < best_alpha:
                best_alpha = alpha
                best = set(sol)
            current = sol
            current_alpha = alpha
            no_improve = 0
            T = T_start * 0.5
    
    # Final polish: try to improve best solution
    if time.time() < deadline - 3:
        # Try removing triples from best to see if we can get lower alpha
        # Actually, removing triples increases alpha, so we want to ADD triples
        # Try to add triples to best
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            avail = build_available_mask(best)
            i = try_add(best, avail)
            if i is None:
                break
            candidate = best | {i}
            alpha = compute_alpha_v2(list(candidate))
            if alpha <= best_alpha:
                best = candidate
                best_alpha = alpha
    
    return [list(all_triples[i]) for i in sorted(best)]

# EVOLVE-BLOCK-END
