import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    MAX_L = 50000  # Cap for coverage tracking
    
    # Precompute masks for each prime p: mask[p][r] = bitset where bit i set if i%p==r, 1<=i<=MAX_L
    # Memory: sum of p for p<=1000 is ~76127, each mask is ~50000 bits = 6250 bytes
    # Total: ~76127 * 6250 = ~476 MB — too much!
    # Instead, compute masks on the fly and cache only for primes in use.
    
    mask_cache = {}
    
    def get_mask(p, r):
        key = (p, r)
        if key in mask_cache:
            return mask_cache[key]
        mask = 0
        start = r if r > 0 else p
        if start <= MAX_L:
            for i in range(start, MAX_L + 1, p):
                mask |= (1 << i)
        mask_cache[key] = mask
        return mask
    
    def get_all_masks_for_prime(p):
        key = p
        if key in mask_cache:
            return mask_cache[key]
        masks = [get_mask(p, r) for r in range(p)]
        mask_cache[key] = masks
        return masks
    
    def compute_L(covered):
        """Find largest L such that all bits 1..L are set in covered."""
        L = 0
        for i in range(1, MAX_L + 1):
            if (covered >> i) & 1:
                L = i
            else:
                break
        return L
    
    def compute_coverage(config):
        """Compute covered bitset and L for a given config."""
        covered = 0
        for p, r in config:
            covered |= get_mask(p, r)
        return covered, compute_L(covered)
    
    def sum_reciprocals(config):
        s = Fraction(0, 1)
        for p, r in config:
            s += Fraction(1, p)
        return s
    
    # === Initial solution: greedy construction ===
    def greedy_init():
        covered = 0
        budget = Fraction(1, 1)
        used = set()
        config = []
        
        # Sort primes by size (smaller primes cover more)
        for p in sorted(primes):
            cost = Fraction(1, p)
            if budget < cost:
                continue
            if p in used:
                continue
            
            # Find best residue
            best_gain = 0
            best_r = 0
            masks = get_all_masks_for_prime(p)
            for r in range(p):
                m = masks[r]
                if m == 0:
                    continue
                new_bits = m & ~covered
                gain = new_bits.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                used.add(p)
                config.append([p, best_r])
                budget -= cost
                covered |= get_mask(p, best_r)
        
        return config
    
    # === Simulated Annealing ===
    def simulated_annealing(config, time_limit):
        # Start with greedy solution
        covered, L = compute_coverage(config)
        best_config = [list(c) for c in config]
        best_L = L
        best_covered = covered
        
        # Temperature schedule
        T0 = 500.0
        Tend = 1.0
        # We'll run until time runs out, cooling geometrically
        
        # Precompute some data
        config_set = set(p for p, r in config)
        available_primes = [p for p in primes if p not in config_set]
        
        # For efficiency, precompute masks for all primes in config
        for p, r in config:
            get_all_masks_for_prime(p)
        
        # Track budget
        current_budget = Fraction(1, 1) - sum(Fraction(1, p) for p, r in config)
        
        # SA loop
        T = T0
        cooling_rate = 0.9995  # Tune this
        
        while time.time() < time_limit:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Residue change: pick a prime in config, change its residue
                if not config:
                    T *= cooling_rate
                    continue
                idx = random.randint(0, len(config) - 1)
                p, old_r = config[idx]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    T *= cooling_rate
                    continue
                
                old_mask = get_mask(p, old_r)
                new_mask = get_mask(p, new_r)
                
                new_covered = (covered & ~old_mask) | new_mask
                new_L = compute_L(new_covered)
                
                delta_L = new_L - L
                # Accept if better, or with probability exp(delta_L / T)
                if delta_L > 0 or random.random() < (2.718281828 ** (delta_L / T)):
                    config[idx][1] = new_r
                    covered = new_covered
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_config = [list(c) for c in config]
                        best_covered = covered
                        
            elif move_type < 0.65:
                # Add a prime: pick an unused prime, find best residue
                if not available_primes:
                    T *= cooling_rate
                    continue
                p = random.choice(available_primes)
                cost = Fraction(1, p)
                if current_budget < cost:
                    T *= cooling_rate
                    continue
                
                # Find best residue for this prime
                best_gain = 0
                best_r = 0
                masks = get_all_masks_for_prime(p)
                for r in range(p):
                    m = masks[r]
                    if m == 0:
                        continue
                    new_bits = m & ~covered
                    gain = new_bits.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_gain == 0:
                    T *= cooling_rate
                    continue
                
                new_mask = get_mask(p, best_r)
                new_covered = covered | new_mask
                new_L = compute_L(new_covered)
                
                delta_L = new_L - L
                if delta_L > 0 or random.random() < (2.718281828 ** (delta_L / T)):
                    config.append([p, best_r])
                    available_primes.remove(p)
                    current_budget -= cost
                    covered = new_covered
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_config = [list(c) for c in config]
                        best_covered = covered
                        
            elif move_type < 0.85:
                # Remove a prime: pick a prime in config, remove it
                if not config:
                    T *= cooling_rate
                    continue
                idx = random.randint(0, len(config) - 1)
                p, r = config[idx]
                old_mask = get_mask(p, r)
                
                new_covered = covered & ~old_mask
                new_L = compute_L(new_covered)
                
                delta_L = new_L - L
                # Removing usually hurts, but may free budget for better primes
                if delta_L > 0 or random.random() < (2.718281828 ** (delta_L / T)):
                    config.pop(idx)
                    available_primes.append(p)
                    current_budget += Fraction(1, p)
                    covered = new_covered
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_config = [list(c) for c in config]
                        best_covered = covered
                        
            else:
                # Swap: remove one prime, add another
                if not config or not available_primes:
                    T *= cooling_rate
                    continue
                idx = random.randint(0, len(config) - 1)
                p_old, r_old = config[idx]
                p_new = random.choice(available_primes)
                
                cost_old = Fraction(1, p_old)
                cost_new = Fraction(1, p_new)
                
                if current_budget + cost_old < cost_new:
                    T *= cooling_rate
                    continue
                
                old_mask = get_mask(p_old, r_old)
                # Find best residue for new prime
                best_gain = 0
                best_r = 0
                masks = get_all_masks_for_prime(p_new)
                for r in range(p_new):
                    m = masks[r]
                    if m == 0:
                        continue
                    new_bits = m & ~covered
                    gain = new_bits.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                new_mask = get_mask(p_new, best_r)
                new_covered = (covered & ~old_mask) | new_mask
                new_L = compute_L(new_covered)
                
                delta_L = new_L - L
                if delta_L > 0 or random.random() < (2.718281828 ** (delta_L / T)):
                    config[idx] = [p_new, best_r]
                    available_primes.remove(p_new)
                    available_primes.append(p_old)
                    current_budget = current_budget + cost_old - cost_new
                    covered = new_covered
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_config = [list(c) for c in config]
                        best_covered = covered
            
            # Cool down
            T *= cooling_rate
            if T < Tend:
                T = Tend
        
        return best_config
    
    # Run SA
    config = greedy_init()
    time_left = deadline - time.time()
    if time_left > 10:
        config = simulated_annealing(config, deadline - 5)
    
    # Final cleanup: ensure budget is respected
    s = sum(Fraction(1, p) for p, r in config)
    if s > Fraction(1, 1):
        # Remove primes until budget is ok (shouldn't happen, but safety)
        while s > Fraction(1, 1) and config:
            # Remove the largest prime (smallest 1/p)
            max_idx = max(range(len(config)), key=lambda i: config[i][0])
            p = config[max_idx][0]
            config.pop(max_idx)
            s -= Fraction(1, p)
    
    return config

if __name__ == "__main__":
    res = construct()
    covered = 0
    MAX_L = 50000
    for p, r in res:
        start = r if r > 0 else p
        for i in range(start, MAX_L + 1, p):
            covered |= (1 << i)
    L = 0
    for i in range(1, MAX_L + 1):
        if (covered >> i) & 1:
            L = i
        else:
            break
    s = sum(Fraction(1, p) for p, r in res)
    print(f"Covered up to L = {L}")
    print(f"Sum 1/p = {s}")
    print(f"Num classes = {len(res)}")
