"""
Strategy: Multi-phase construction with fast numpy-based coverage tracking.
Phase 1: Greedy gap-closure (fix first under-covered point).
Phase 2: Simulated annealing-style local search to escape local minima.
Phase 3: Final polish with targeted improvements.
"""
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We'll track coverage up to a reasonable limit
    max_check = 5000
    
    # residues[i] is the residue for primes[i]
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # counts[x] = number of primes p such that x == residues[idx] (mod p)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_counts(idx, old_r, new_r, delta):
        """
        Update counts array.
        delta = +1 means adding the residue, -1 means removing.
        """
        p = primes[idx]
        r = new_r if delta == 1 else old_r
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= max_check:
            counts[start::p] += delta

    def get_prefix_score():
        """
        Find the largest L such that counts[1..L] >= 2.
        """
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]

    # Initialize: set all residues to 0 and compute counts
    for i in range(n_primes):
        update_counts(i, 0, 0, 1)
        
    best_score = get_prefix_score()
    best_residues = residues.copy()

    # --- Phase 1: Greedy Construction ---
    # Iteratively fix the first gap (count < 2)
    
    max_repair_steps = 10000
    step = 0
    
    while time.time() < deadline - 30.0 and step < max_repair_steps:
        step += 1
        
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        x_target = current_score + 1
        
        # Find a prime to cover x_target
        # Try primes in descending order (larger primes affect fewer positions)
        found = False
        for i in range(n_primes - 1, -1, -1):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
                
            # Check if removing coverage from old_r breaks any prefix point
            if old_r == 0:
                start_pos = p
            else:
                start_pos = old_r
                
            if start_pos > current_score:
                valid = True
            else:
                positions = np.arange(start_pos, current_score + 1, p)
                if np.any(counts[positions] == 2):
                    valid = False
                else:
                    valid = True
            
            if valid:
                update_counts(i, old_r, new_r, -1)
                residues[i] = new_r
                update_counts(i, new_r, new_r, 1)
                found = True
                break
        
        if not found:
            # Stuck! Try a random kick to escape local minimum
            kicked = False
            for _ in range(20):
                i = np.random.randint(n_primes)
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                
                if new_r == old_r:
                    continue
                    
                # Check if this move breaks the prefix
                if old_r == 0:
                    start_pos = p
                else:
                    start_pos = old_r
                    
                valid = True
                if start_pos <= current_score:
                    positions = np.arange(start_pos, current_score + 1, p)
                    if np.any(counts[positions] == 2):
                        valid = False
                
                if valid:
                    update_counts(i, old_r, new_r, -1)
                    residues[i] = new_r
                    update_counts(i, new_r, new_r, 1)
                    kicked = True
                    break
            
            if not kicked:
                # Try a more aggressive kick: change a prime and accept temporary damage
                i = np.random.randint(n_primes)
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                
                if new_r != old_r:
                    update_counts(i, old_r, new_r, -1)
                    residues[i] = new_r
                    update_counts(i, new_r, new_r, 1)
                else:
                    break

    # --- Phase 2: Local Search / Annealing ---
    # Try to improve by making moves that don't immediately break the prefix
    # but might lead to better long-term coverage
    
    temperature = 50.0
    cooling = 0.995
    min_temp = 1.0
    
    while time.time() < deadline - 5.0 and temperature > min_temp:
        # Pick a random prime
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
        
        # Make the move
        update_counts(i, old_r, new_r, -1)
        residues[i] = new_r
        update_counts(i, new_r, new_r, 1)
        
        new_score = get_prefix_score()
        
        # Accept if better, or with probability based on temperature
        if new_score > best_score:
            best_score = new_score
            best_residues = residues.copy()
        elif new_score >= best_score:
            # Accept if same or better (we're tracking best separately)
            pass
        else:
            # Worse: accept with probability exp((new_score - best_score) / temperature)
            # But scores are integers, so use a simpler criterion
            if np.random.random() < np.exp(-(best_score - new_score) / temperature):
                pass  # Accept
            else:
                # Revert
                update_counts(i, new_r, old_r, -1)
                residues[i] = old_r
                update_counts(i, old_r, old_r, 1)
        
        temperature *= cooling

    # --- Phase 3: Final Polish ---
    # One more round of greedy gap-closure
    for _ in range(1000):
        if time.time() > deadline - 2.0:
            break
            
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        x_target = current_score + 1
        
        found = False
        for i in range(n_primes - 1, -1, -1):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
                
            if old_r == 0:
                start_pos = p
            else:
                start_pos = old_r
                
            if start_pos > current_score:
                valid = True
            else:
                positions = np.arange(start_pos, current_score + 1, p)
                if np.any(counts[positions] == 2):
                    valid = False
                else:
                    valid = True
            
            if valid:
                update_counts(i, old_r, new_r, -1)
                residues[i] = new_r
                update_counts(i, new_r, new_r, 1)
                found = True
                break
        
        if not found:
            break

    # Final check
    final_score = get_prefix_score()
    if final_score > best_score:
        best_score = final_score
        best_residues = residues.copy()

    return [int(r) for r in best_residues]
