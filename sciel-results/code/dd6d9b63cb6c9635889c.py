"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 135  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # For each divisor, precompute the number of positions it covers
    # and the positions for each residue class
    # div_positions[m][r] = set of positions covered by class (m, r)
    # This is too much memory for large m, so we compute on the fly
    
    def count_uncovered(covered):
        """Count uncovered positions using bytearray."""
        return sum(1 for c in covered if c == 0)
    
    def fast_count_uncovered(covered):
        """Faster count using Python's sum over bytearray."""
        return base - sum(covered)
    
    # Adaptive randomized greedy with look-ahead
    def adaptive_greedy_init(seed_offset=0):
        """Randomized greedy: probabilistically select next modulus based on marginal gain."""
        rng = random.Random(seed_offset)
        covered = bytearray(base)
        residues = [0] * n_div
        used = [False] * n_div
        remaining = list(range(n_div))
        
        # Compute initial marginal gains for all moduli
        # For each modulus m and residue r, gain = number of uncovered positions in class (m,r)
        # We'll compute this lazily
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Compute marginal gain for each remaining modulus
            # For each remaining modulus, find the best residue and its gain
            best_idx = -1
            best_gain = -1
            best_res = 0
            # For adaptive selection, compute gains for all and pick probabilistically
            gains = []
            for idx in remaining:
                m = divisors[idx]
                # Find best residue for this modulus
                best_r = 0
                best_g = -1
                # Sample a few residues if m is large to save time
                if m <= 50:
                    rs_to_try = range(m)
                else:
                    # Sample up to 20 residues
                    rs_to_try = [0] + [rng.randint(0, m-1) for _ in range(min(19, m-1))]
                    # Remove duplicates
                    rs_to_try = list(set(rs_to_try))
                
                for r in rs_to_try:
                    # Count uncovered positions in this residue class
                    g = 0
                    for p in range(r, base, m):
                        if covered[p] == 0:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_r = r
                
                gains.append((best_g, idx, best_r))
            
            if not gains:
                break
            
            # Sort by gain descending
            gains.sort(key=lambda x: -x[0])
            
            # Adaptive selection: pick probabilistically weighted by gain
            # Use a softmax-like selection with temperature
            total_gain = sum(g for g, _, _ in gains)
            if total_gain == 0:
                # All gains are zero, pick randomly
                pick = rng.choice(gains)
            else:
                # Weighted random selection: probability proportional to gain^2 (emphasize high gain)
                weights = [float(g*g) for g, _, _ in gains]
                total_w = sum(weights)
                if total_w == 0:
                    pick = rng.choice(gains)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    pick = gains[0]
                    for item, w in zip(gains, weights):
                        cum += w
                        if r <= cum:
                            pick = item
                            break
            
            _, idx, res = pick
            m = divisors[idx]
            
            # Apply this class
            for p in range(res, base, m):
                covered[p] = 1
            residues[idx] = res
            used[idx] = True
            remaining.remove(idx)
        
        return residues, covered
    
    # Fast local search using delta computation
    def local_search(residues, covered, current_uncovered, max_iters=50000):
        """Local search: try single and double moves with delta computation."""
        best_residues = residues[:]
        best_uncovered = current_uncovered
        temp = 50.0
        cooling = 0.9995
        no_improve = 0
        rng = random.Random(42)
        
        # Precompute for each modulus the list of positions for its current residue
        # This helps with fast delta computation
        mod_positions = {}
        for i, m in enumerate(divisors):
            r = residues[i]
            mod_positions[i] = list(range(r, base, m))
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            if temp < 1.0:
                break
            
            # Pick a random modulus to change
            i = rng.randrange(n_div)
            mi = divisors[i]
            old_ri = residues[i]
            
            # Pick a new residue
            if rng.random() < 0.3:
                # Try a specific promising residue
                new_ri = rng.randint(0, mi - 1)
            else:
                new_ri = rng.randint(0, mi - 1)
            
            if new_ri == old_ri:
                continue
            
            # Compute delta: change in uncovered count
            # Old positions: old_ri, old_ri+mi, ...
            # New positions: new_ri, new_ri+mi, ...
            # Positions that change: symmetric difference
            
            # For positions in old but not new: they lose coverage from modulus i
            #   If they were only covered by i, they become uncovered (+1 to uncovered)
            # For positions in new but not old: they gain coverage from modulus i
            #   If they were uncovered, they become covered (-1 to uncovered)
            
            # Efficient computation:
            # affected = (old_positions - new_positions) ∪ (new_positions - old_positions)
            
            delta = 0
            
            # Positions in old but not new
            for p in range(old_ri, base, mi):
                # Check if p is also in new
                if (p - new_ri) % mi == 0 and p >= new_ri:
                    continue
                # p loses coverage from i
                if covered[p] == 1:
                    # Was covered; check if only by i
                    # We need to verify no other modulus covers p
                    # This is expensive; use a heuristic: assume it's covered by others
                    # Actually, we can't easily check. Let's use a different approach.
                    pass
            
            # This approach is too slow. Let's use a different strategy:
            # Just recompute for the affected positions using the covered array.
            # We'll temporarily modify covered and count the change.
            
            # Save old values
            old_covered = bytearray(covered)  # This is a full copy, ~52KB, fast enough
            
            # Remove old coverage
            for p in range(old_ri, base, mi):
                covered[p] = 0
            # Add new coverage
            for p in range(new_ri, base, mi):
                covered[p] = 1
            
            # Count new uncovered
            new_uncovered = base - sum(covered)
            delta = new_uncovered - current_uncovered
            
            if delta < 0 or (delta == 0 and rng.random() < 0.1) or (rng.random() < (2.718281828 ** (-delta / temp)) if delta > 0 else 0):
                # Accept
                residues[i] = new_ri
                current_uncovered = new_uncovered
                no_improve = 0
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues[:]
            else:
                # Reject, restore
                covered[:] = old_covered
                no_improve += 1
            
            temp *= cooling
            
            if no_improve > 2000:
                # Restart with higher temperature
                temp = 50.0
                no_improve = 0
        
        return best_residues, best_uncovered, covered
    
    # Main: try multiple adaptive greedy inits with different seeds, then local search
    best_uncovered = base  # worst case
    best_residues = [0] * n_div
    
    # Run adaptive greedy with a few different seeds
    for seed in range(3):
        if time.time() > deadline - 60:
            break
        
        residues, covered = adaptive_greedy_init(seed * 1000 + 42)
        current_uncovered = base - sum(covered)
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues[:]
        
        # Local search from this starting point
        if time.time() < deadline - 5:
            new_res, new_unc, _ = local_search(residues, covered, current_uncovered)
            if new_unc < best_uncovered:
                best_uncovered = new_unc
                best_residues = new_res[:]
    
    # Final: try to improve with more local search from the best solution
    if time.time() < deadline - 3:
        # Reconstruct covered from best_residues
        covered = bytearray(base)
        for i, m in enumerate(divisors):
            r = best_residues[i]
            for p in range(r, base, m):
                covered[p] = 1
        current_uncovered = base - sum(covered)
        
        # More local search
        new_res, new_unc, _ = local_search(best_residues, covered, current_uncovered, max_iters=100000)
        if new_unc < best_uncovered:
            best_uncovered = new_unc
            best_residues = new_res[:]
    
    # Return the best solution found
    classes = [[divisors[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
