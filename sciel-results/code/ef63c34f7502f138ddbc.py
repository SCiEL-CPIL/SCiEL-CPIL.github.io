import math
import random
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 1000
PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)

def _calculate_coverage(classes):
    """
    Calculate the largest L such that 1..L are covered by the given classes.
    classes: list of [p, r]
    Returns L (capped at 100000).
    """
    if not classes:
        return 0
    
    # We only care up to a reasonable limit for speed during search, 
    # but the final score is capped at 10^5. 
    # However, checking up to 10^5 every step is slow.
    # Let's check up to a dynamic limit or a fixed high limit like 5000 for speed,
    # but the seed score is ~1000. 
    # To be safe and fast, we can check up to 10000. If we cover 10000, we likely cover much more.
    # Actually, the max L is the score. If we find a gap early, we stop.
    
    # Optimization: Use a boolean array for coverage up to a certain limit.
    # If we expect L to be around 1000-2000, checking up to 5000 is sufficient for search.
    # The final result will be evaluated exactly by the judge.
    # For the search, we need a fast proxy.
    
    limit = 5000 
    covered = [False] * (limit + 1)
    
    for p, r in classes:
        # Mark numbers n in [1, limit] such that n % p == r
        # First number is r if r > 0, else p (since n>=1, and 0 is not in 1..L usually? 
        # Wait, "every integer 1..L". So n starts at 1.
        # If r == 0, first is p. If r > 0, first is r (if r>=1).
        start = r if r > 0 else p
        if start < 1:
            start += p
        for n in range(start, limit + 1, p):
            covered[n] = True
            
    # Find the largest L such that 1..L are all True
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return L

def _sum_reciprocals(classes):
    """Calculate sum of 1/p for given classes."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds safety margin
    
    # Initial solution: The seed program's solution
    # The i-th prime from 13 upward covers the integer i.
    # Let's just start with a reasonable greedy or the seed itself.
    # Seed: [[p, (i+1)%p] for i, p in enumerate(primes_from_13)]
    
    primes_from_13 = [p for p in PRIMES if p >= 13]
    initial_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_from_13)]
    
    # Check if initial is valid
    if _sum_reciprocals(initial_classes) > 1:
        # Fallback to empty if invalid (shouldn't happen with seed logic usually, but safe)
        initial_classes = []
        
    current_classes = [c[:] for c in initial_classes]
    current_score = _calculate_coverage(current_classes)
    best_classes = [c[:] for c in current_classes]
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.01
    
    # We need to track available primes
    used_primes = set(p for p, r in current_classes)
    available_primes = [p for p in PRIMES if p not in used_primes and p < 1000]
    
    # Sort available primes by size? Smaller primes are more "valuable" for coverage density
    # but cost more budget.
    
    while T > min_T and time.time() < deadline:
        # Generate a neighbor
        neighbor = [c[:] for c in current_classes]
        move_type = random.randint(0, 3)
        
        if move_type == 0 and neighbor:
            # Change residue of a random existing prime
            idx = random.randint(0, len(neighbor) - 1)
            p, r = neighbor[idx]
            new_r = random.randint(0, p - 1)
            neighbor[idx] = [p, new_r]
            
        elif move_type == 1 and neighbor and available_primes:
            # Swap a prime: remove one, add another
            idx = random.randint(0, len(neighbor) - 1)
            p_old, r_old = neighbor[idx]
            
            # Pick a random available prime
            # To maintain budget, we should prefer swapping with similar or smaller primes?
            # Or just random.
            p_new = random.choice(available_primes)
            
            # Check if budget allows
            # Current sum + 1/p_new - 1/p_old <= 1
            # Approximate check for speed, exact check later
            if 1.0/p_new - 1.0/p_old < 0.1: # Rough check
                 # Remove old
                 neighbor.pop(idx)
                 # Add new with random residue
                 r_new = random.randint(0, p_new - 1)
                 neighbor.append([p_new, r_new])
                 
                 # Update used/available sets for consistency (though we re-eval from scratch mostly)
                 # For now, just modify the list. The sets are only for move generation logic.
                 # We need to be careful not to pick a prime that is now in neighbor if we don't update sets.
                 # Let's just use a simple check: is p_new already in neighbor?
                 if any(p == p_new for p, r in neighbor):
                     # Conflict, revert
                     neighbor[idx] = [p_old, r_old]
                     neighbor = [c[:] for c in current_classes] # Revert completely
                     pass
                 else:
                     # Valid move candidate
                     pass
            else:
                 # Revert
                 neighbor = [c[:] for c in current_classes]
                 
        elif move_type == 2 and available_primes:
            # Add a new prime
            # Try to add a prime that fits in budget
            current_sum = _sum_reciprocals(neighbor)
            # Find candidates that fit
            candidates = [p for p in available_primes if current_sum + Fraction(1, p) <= 1]
            if candidates:
                p_new = random.choice(candidates)
                r_new = random.randint(0, p_new - 1)
                neighbor.append([p_new, r_new])
            else:
                neighbor = [c[:] for c in current_classes]
                
        elif move_type == 3 and neighbor:
            # Remove a prime
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
            
        else:
            # No move possible, revert
            neighbor = [c[:] for c in current_classes]
            
        # Evaluate neighbor
        # Check validity
        sum_val = _sum_reciprocals(neighbor)
        if sum_val > 1:
            neighbor_score = 0
        else:
            neighbor_score = _calculate_coverage(neighbor)
            
        # Acceptance criterion
        delta = neighbor_score - current_score
        if delta > 0:
            # Accept
            current_classes = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
        else:
            # Probabilistic accept
            # Probability exp(-delta / T)
            # delta is negative
            try:
                prob = math.exp(-delta / T)
            except OverflowError:
                prob = 0.0
            if random.random() < prob:
                current_classes = neighbor
                current_score = neighbor_score
                
        # Update temperature
        T *= cooling_rate
        
        # Update available primes for move generation?
        # This is tricky because 'available_primes' list doesn't reflect current_classes changes immediately.
        # For simplicity and speed, we rely on the 'is p in neighbor' check in swap/add moves.
        # However, 'available_primes' might contain primes currently in neighbor if we added them.
        # Let's refine the move generation to be safer or just accept that we might pick a used prime and handle it.
        
        # Optimization: Periodically refresh available_primes
        if random.random() < 0.01:
            used = set(p for p, r in current_classes)
            available_primes = [p for p in PRIMES if p not in used]

    # Final verification and cleanup
    # Ensure the best solution is valid
    if _sum_reciprocals(best_classes) > 1:
        # Fallback to seed if best is invalid (shouldn't happen)
        return initial_classes
        
    return best_classes

# EVOLVE-BLOCK-END
