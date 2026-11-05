"""Rewritten program: Targeted Greedy Repair with Random Escape for residue optimization."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)  # 168
    L_max = 5000  # Search horizon
    
    # Initialize residues
    residues = [0] * n_primes
    coverage = [0] * (L_max + 1)
    
    # Helper to update coverage for a prime change
    # Returns delta_deficit
    def get_delta(i, r_new, coverage, residues, primes):
        p = primes[i]
        r_old = residues[i]
        if r_old == r_new:
            return 0
        
        # Generate positions
        # Optimization: Instead of creating lists, we can iterate.
        # But for simplicity and safety, we create sets for small p, 
        # and for large p, the lists are short.
        
        # We need to iterate over the union of old and new positions.
        # Positions are r, r+p, r+2p, ... <= L_max
        # Note: if r==0, positions are p, 2p, ...
        
        def gen_positions(r, p):
            x = r if r > 0 else p
            while x <= L_max:
                yield x
                x += p
        
        # To calculate delta, we need to know coverage[x] for all x in (Old U New).
        # We can use a dictionary to store coverage values for affected x, 
        # or just iterate and modify a temp copy? 
        # Modifying a temp copy of coverage (size 5000) is fast (5000 ops).
        # But doing it for every candidate (168 candidates) is 168 * 5000 = 840k ops.
        # This is acceptable if the number of iterations is not too high.
        
        # However, we can be smarter.
        # We only care about x where coverage changes.
        # Let's collect the set of affected x's.
        
        old_pos = set(gen_positions(r_old, p))
        new_pos = set(gen_positions(r_new, p))
        
        # Intersection has no net change
        # Lost = Old - New
        # Gained = New - Old
        
        delta = 0
        # Process Lost
        for x in old_pos:
            if x in new_pos:
                continue
            c = coverage[x]
            # c decreases by 1
            if c >= 2:
                pass # 2->1 (deficit 0->1? No, deficit is max(0, 2-c))
                # Wait. Deficit f(c) = max(0, 2-c).
                # c=2, f=0. c=1, f=1.
                # If c=2, new c=1. f(1)=1. Delta = 1 - 0 = +1.
                # If c=3, new c=2. f(2)=0. Delta = 0.
                if c == 2:
                    delta += 1
            elif c == 1:
                # c=1, f=1. new c=0, f=2. Delta = 2 - 1 = +1.
                delta += 1
            # c=0 impossible to lose.
        
        # Process Gained
        for x in new_pos:
            if x in old_pos:
                continue
            c = coverage[x]
            # c increases by 1
            if c == 0:
                # f(0)=2. new c=1, f(1)=1. Delta = 1 - 2 = -1.
                delta -= 1
            elif c == 1:
                # f(1)=1. new c=2, f(2)=0. Delta = 0 - 1 = -1.
                delta -= 1
            # c>=2: f(c)=0. new c>=3, f=0. Delta 0.
            
        return delta

    def apply_change(i, r_new, coverage, residues, primes):
        p = primes[i]
        r_old = residues[i]
        
        # Update coverage
        x = r_old if r_old > 0 else p
        while x <= L_max:
            coverage[x] -= 1
            x += p
            
        x = r_new if r_new > 0 else p
        while x <= L_max:
            coverage[x] += 1
            x += p
            
        residues[i] = r_new

    def find_L(coverage):
        L = 0
        for x in range(1, L_max + 1):
            if coverage[x] >= 2:
                L = x
            else:
                break
        return L

    # Initialization: Greedy
    # Try to cover early positions
    for i, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        # We want to maximize coverage of [1, 200]
        for r in range(p):
            gain = 0
            x = r if r > 0 else p
            while x <= 200:
                if coverage[x] < 2:
                    gain += 1
                x += p
            if gain > best_gain:
                best_gain = gain
                best_r = r
        apply_change(i, best_r, coverage, residues, primes)
        
    current_L = find_L(coverage)
    best_L = current_L
    best_residues = residues[:]
    best_coverage = coverage[:]
    
    # Main Loop
    start_time = time.time()
    stuck_count = 0
    
    while time.time() - start_time < 110:
        # Find first deficient position
        x_def = -1
        for x in range(1, L_max + 1):
            if coverage[x] < 2:
                x_def = x
                break
        
        if x_def == -1:
            # No deficient positions in range, break or check L
            # Actually if no deficient, L is L_max.
            break
            
        # Try to fix x_def
        best_delta = 0
        best_i = -1
        best_r = -1
        
        for i in range(n_primes):
            p = primes[i]
            r_target = x_def % p
            if r_target == residues[i]:
                continue # Already covers x_def, changing would break it
            
            delta = get_delta(i, r_target, coverage, residues, primes)
            
            # We prefer moves that reduce deficit.
            # If delta < best_delta, it's better.
            # Note: best_delta starts at 0. We only accept if delta < 0 (improvement).
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_r = r_target
                
        if best_i != -1 and best_delta < 0:
            apply_change(best_i, best_r, coverage, residues, primes)
            stuck_count = 0
        else:
            # Stuck. Do a random move to escape local minimum.
            # Pick random prime and random residue
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_new = random.randint(0, p - 1)
            
            if r_new == residues[i]:
                continue
                
            delta = get_delta(i, r_new, coverage, residues, primes)
            
            # Accept with some probability (Simulated Annealing style)
            # Or just accept if it doesn't make things too bad?
            # Let's use a simple temperature decay.
            # We need a global T.
            
            # To keep it simple, let's just accept if delta <= 2 (small penalty)
            # Or use random.random() < exp(-delta / T)
            
            # Let's stick to a simple heuristic: accept if delta < 5
            if delta < 5:
                apply_change(i, r_new, coverage, residues, primes)
            
            stuck_count += 1
            if stuck_count > 1000:
                # Reset to best known? Or just continue
                # Resetting might be better if we are far away
                # But resetting loses progress.
                # Let's just continue.
                pass
        
        # Periodically check L and update best
        if stuck_count % 100 == 0:
            current_L = find_L(coverage)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
                best_coverage = coverage[:]
                
    # Final check
    current_L = find_L(coverage)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues[:]
        
    return best_residues
# EVOLVE-BLOCK-END
