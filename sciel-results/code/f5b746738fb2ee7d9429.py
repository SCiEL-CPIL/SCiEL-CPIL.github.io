"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Deadline guard: 150s limit, use 145s for safety
    deadline = time.time() + 145.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Cap for coverage array. We want to see if we can reach 1000.
    # The score is capped at 10^5, but L >= 1000 is the target.
    # Let's set cap to 5000 to allow some buffer beyond 1000.
    cap = 5000
    
    # Initial assignment: 
    # Strategy: Start with r_p = 1 for all p? Or r_p = (i+1) % p?
    # Let's try a mix: r_p = 1 for small primes, random for others?
    # Actually, r_p = 1 means 1 is covered by all primes. 2 is covered by primes p where 2%p==1 => p=1 (not prime) or p divides 1? No.
    # x % p == 1 => x = 1 + k*p.
    # If r_p = 1, then 1 is covered by all 168 primes. Coverage(1) = 168.
    # 2 is covered by p if 2 % p == 1 => p=1 (no).
    # So r_p=1 is a very strong start for x=1, but weak for x=2.
    
    # Let's try r_p = 0? 
    # x % p == 0 => x is multiple of p.
    # 1 is not covered.
    
    # Let's try the current program's init: (i+1) % p.
    residues = [(i + 1) % p for i, p in enumerate(primes)]
    
    # Build initial coverage
    coverage = [0] * (cap + 1)
    for r, p in zip(residues, primes):
        first = r if r > 0 else p
        x = first
        while x <= cap:
            coverage[x] += 1
            x += p
            
    def get_L(coverage):
        L = 0
        for i in range(1, cap + 1):
            if coverage[i] >= 2:
                L = i
            else:
                break
        return L
        
    current_L = get_L(coverage)
    best_L = current_L
    best_residues = residues[:]
    
    rng = random.Random(42)
    
    # Helper to update coverage for a single prime change
    def apply_change(idx, new_r):
        p = primes[idx]
        old_r = residues[idx]
        
        # Remove old
        first_old = old_r if old_r > 0 else p
        x = first_old
        while x <= cap:
            coverage[x] -= 1
            x += p
            
        # Add new
        first_new = new_r if new_r > 0 else p
        x = first_new
        while x <= cap:
            coverage[x] += 1
            x += p
            
        residues[idx] = new_r
        
    def revert_change(idx, new_r):
        p = primes[idx]
        old_r = residues[idx] # This is new_r currently
        
        # Remove new
        first_new = new_r if new_r > 0 else p
        x = first_new
        while x <= cap:
            coverage[x] -= 1
            x += p
            
        # Add old
        # We need to know the old residue. 
        # This function is dangerous if not careful. 
        # Better to pass old_r.
        pass

    # Safer approach for the main loop:
    # We will manually manage the state.
    
    it = 0
    while time.time() < deadline:
        it += 1
        
        # Find bottleneck
        x_target = current_L + 1
        if x_target > cap:
            break
            
        moved = False
        
        # 1. Directed Phase: Try to fix x_target
        # For each prime, try setting r_p = x_target % p
        # We iterate in random order to avoid bias
        indices = list(range(n_primes))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            p = primes[idx]
            new_r = x_target % p
            old_r = residues[idx]
            
            if new_r == old_r:
                continue
                
            # Apply change
            # Remove old coverage
            first_old = old_r if old_r > 0 else p
            x = first_old
            while x <= cap:
                coverage[x] -= 1
                x += p
                
            # Add new coverage
            first_new = new_r if new_r > 0 else p
            x = first_new
            while x <= cap:
                coverage[x] += 1
                x += p
                
            # Calculate new L
            new_L = get_L(coverage)
            
            if new_L > current_L:
                # Accept
                residues[idx] = new_r
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                moved = True
                break # Restart search with new L
            else:
                # Revert
                # Remove new
                x = first_new
                while x <= cap:
                    coverage[x] -= 1
                    x += p
                # Add old
                x = first_old
                while x <= cap:
                    coverage[x] += 1
                    x += p
                
        if moved:
            continue
            
        # 2. Random Phase: If stuck, try random moves
        # We accept a random move if it doesn't decrease L, 
        # or if it decreases L slightly (to escape local minima), 
        # but we must track the best result.
        
        # Try a random prime
        idx = rng.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        new_r = rng.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Apply
        first_old = old_r if old_r > 0 else p
        x = first_old
        while x <= cap:
            coverage[x] -= 1
            x += p
            
        first_new = new_r if new_r > 0 else p
        x = first_new
        while x <= cap:
            coverage[x] += 1
            x += p
            
        new_L = get_L(coverage)
        
        # Acceptance criteria:
        # 1. If new_L > current_L: Accept
        # 2. If new_L == current_L: Accept (explore plateau)
        # 3. If new_L < current_L: Reject (keep it simple, avoid getting stuck in bad states)
        
        if new_L >= current_L:
            residues[idx] = new_r
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
        else:
            # Revert
            x = first_new
            while x <= cap:
                coverage[x] -= 1
                x += p
            x = first_old
            while x <= cap:
                coverage[x] += 1
                x += p
                
    return best_residues
# EVOLVE-BLOCK-END
