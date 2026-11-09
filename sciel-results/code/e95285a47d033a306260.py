"""Program for dense increasing sequences with unique block products."""
# EVOLVE-BLOCK-START
import math
import random
import time
import bisect

def _count_conflicts(seq):
    """Count the number of duplicate consecutive block products."""
    prods = set()
    conflicts = 0
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in prods:
                conflicts += 1
            else:
                prods.add(p)
    return conflicts

def construct():
    """
    Return strictly increasing integers in [2,128] such that products of
    every nonempty consecutive block are distinct. Maximize length.
    
    Approach: Simulated Annealing on the set of selected integers.
    We start with the set of primes in [2, 128] which is a known valid sequence.
    We then use Simulated Annealing to try to insert additional composite numbers
    or swap numbers to increase the total length while maintaining 0 conflicts.
    """
    # Generate primes in [2, 128]
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(2, 129) if is_prime(x)]
    current_seq = primes[:]
    current_energy = -len(current_seq) + 100 * _count_conflicts(current_seq)
    
    best_seq = current_seq[:]
    best_energy = current_energy
    
    all_numbers = set(range(2, 129))
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    start_time = time.time()
    time_limit = 55.0  # Leave 5s for overhead
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate neighbor
        r = random.random()
        new_seq = current_seq[:]
        
        if r < 0.5:
            # Insert a random unused number
            unused = all_numbers - set(current_seq)
            if not unused:
                continue
            x = random.choice(list(unused))
            bisect.insort(new_seq, x)
        elif r < 0.8:
            # Swap a used number with an unused number
            if not current_seq:
                continue
            unused = all_numbers - set(current_seq)
            if not unused:
                continue
            i = random.randint(0, len(current_seq) - 1)
            x_remove = current_seq[i]
            x_add = random.choice(list(unused))
            new_seq.pop(i)
            bisect.insort(new_seq, x_add)
        else:
            # Delete a random used number
            if not current_seq:
                continue
            i = random.randint(0, len(current_seq) - 1)
            new_seq.pop(i)
            
        # Calculate new energy
        new_conflicts = _count_conflicts(new_seq)
        new_energy = -len(new_seq) + 100 * new_conflicts
        
        # Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_seq = new_seq
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_seq = current_seq[:]
                
        T *= cooling_rate
        
    # Ensure the best sequence is valid (0 conflicts)
    # If best has conflicts, we should fall back to a known valid one, 
    # but SA usually finds valid ones. 
    # To be safe, if best_seq has conflicts, we can try to repair or just return it?
    # The task requires valid sequences. 
    # We will check if best_seq is valid. If not, we return the initial primes.
    if _count_conflicts(best_seq) > 0:
        return primes
    
    return best_seq
# EVOLVE-BLOCK-END
