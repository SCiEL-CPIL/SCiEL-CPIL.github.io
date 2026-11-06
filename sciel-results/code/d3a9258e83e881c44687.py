import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 320.0

def _sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    primes = _sieve_primes(N)
    # Only primes <= N matter for the constraint on [0, N]
    # Actually, if p > N, we can choose r_p > N, so no removal.
    # So we only optimize p <= N.
    
    P = len(primes)
    # Map prime to index
    p_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Residues r[i] for primes[i]
    r = [0] * P
    
    # block_count[x] = number of primes p such that x % p == r[p]
    block_count = [0] * (N + 1)
    
    # Initialize block_count
    for i, p in enumerate(primes):
        r_p = r[i]
        for x in range(r_p, N + 1, p):
            block_count[x] += 1
            
    # surv_count[i][res] = number of x in [0, N] such that x % p_i == res and block_count[x] == 0
    surv_count = [[0] * p for p in primes]
    
    # Initialize surv_count
    for x in range(N + 1):
        if block_count[x] == 0:
            for i, p in enumerate(primes):
                surv_count[i][x % p] += 1
                
    # Precompute G_old for each p?
    # G_old[i] = count(x % p_i == r[i] and block_count[x] == 1)
    G_old = [0] * P
    for i, p in enumerate(primes):
        r_p = r[i]
        cnt = 0
        for x in range(r_p, N + 1, p):
            if block_count[x] == 1:
                cnt += 1
        G_old[i] = cnt
        
    best_size = sum(1 for x in range(N + 1) if block_count[x] == 0)
    
    deadline = start_time + DEADLINE - 5.0 # Safety margin
    
    while time.time() < deadline:
        best_move = -1
        best_gain = 0
        best_new_r = 0
        
        # Find best move
        for i, p in enumerate(primes):
            g_old = G_old[i]
            # Find r' != r[i] minimizing surv_count[i][r']
            min_surv = float('inf')
            min_r = -1
            current_r = r[i]
            for res in range(p):
                if res == current_r:
                    continue
                if surv_count[i][res] < min_surv:
                    min_surv = surv_count[i][res]
                    min_r = res
            
            if min_r == -1:
                continue
                
            gain = g_old - min_surv
            if gain > best_gain:
                best_gain = gain
                best_move = i
                best_new_r = min_r
        
        if best_move == -1:
            # No improving move found. Try random perturbation or break?
            # Let's try a random move to escape local minimum
            if random.random() < 0.5:
                i = random.randint(0, P - 1)
                p = primes[i]
                old_r = r[i]
                # Pick random new r
                new_r = random.randint(0, p - 1)
                while new_r == old_r:
                    new_r = random.randint(0, p - 1)
                best_move = i
                best_new_r = new_r
            else:
                break
        
        # Apply move
        i = best_move
        p = primes[i]
        old_r = r[i]
        new_r = best_new_r
        
        # Update block_count and surv_count
        # Affected x: x % p == old_r or x % p == new_r
        
        # 1. Unblock old_r
        for x in range(old_r, N + 1, p):
            old_bc = block_count[x]
            block_count[x] = old_bc - 1
            if old_bc == 1:
                # Becomes survivor
                # Update surv_count for all q
                for j, q in enumerate(primes):
                    surv_count[j][x % q] += 1
            elif old_bc == 2:
                # Still blocked, but G_old for this p might change?
                # G_old counts block_count == 1.
                # This x was counted in G_old[i]. Now it is 1.
                # Wait, G_old[i] is recalculated every iteration? No, I'm updating incrementally.
                # This is getting complex.
                pass
        
        # 2. Block new_r
        for x in range(new_r, N + 1, p):
            old_bc = block_count[x]
            block_count[x] = old_bc + 1
            if old_bc == 0:
                # Becomes blocked
                for j, q in enumerate(primes):
                    surv_count[j][x % q] -= 1
        
        r[i] = new_r
        
        # Recalculate G_old for all p? Or just i?
        # G_old[j] depends on block_count.
        # block_count changed for x in old_r/new_r classes of p.
        # This affects G_old[j] for all j where x % p_j == r_j and block_count[x] changed status (1<->2 or 0<->1).
        # Specifically, G_old[j] counts x with block_count[x]==1.
        # If block_count[x] went 1->0, it is no longer counted in G_old[j] (if x%q==r_q).
        # If block_count[x] went 0->1, it is now counted in G_old[j] (if x%q==r_q).
        # If block_count[x] went 2->1, it is now counted.
        # If block_count[x] went 1->2, it is no longer counted.
        
        # To be safe and simple, let's just recalculate G_old for all p every iteration.
        # Cost: Sum(N/p) ~ 5000 * 2.4 ~ 12000 ops. Very fast.
        for j, q in enumerate(primes):
            r_q = r[j]
            cnt = 0
            for x in range(r_q, N + 1, q):
                if block_count[x] == 1:
                    cnt += 1
            G_old[j] = cnt
            
        # Update best_size
        # We can track size incrementally too, or just count.
        # Count is 5000 ops. Fast.
        current_size = sum(1 for x in range(N + 1) if block_count[x] == 0)
        if current_size > best_size:
            best_size = current_size
            
        # Check time
        if time.time() > deadline:
            break
            
    # Construct final list
    result = [x for x in range(N + 1) if block_count[x] == 0]
    
    # Verify admissibility
    # For every prime p <= len(result), some class is missed.
    # By construction, for p <= 5000, we have a specific missed class r[p].
    # If len(result) < p, condition is vacuously true.
    # If len(result) >= p, condition requires missed class.
    # Our construction ensures class r[p] is missed for all p <= 5000.
    # So it is valid.
    
    return result
