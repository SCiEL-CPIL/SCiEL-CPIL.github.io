import time
import random
from fractions import Fraction
import sys

def _get_primes(limit=1000):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _get_primes(1000)
    L_target = 100000 # Cap for evaluation
    
    # --- Initialization ---
    # We start with a reasonable seed: the "standard" covering using primes >= 13
    # This provides a baseline L ~ 999.
    seed_primes = [p for p in primes if p >= 13]
    # The seed maps i-th prime to cover i+1 (roughly). 
    # Actually, the standard construction for x=1000, C=1 uses primes 13..997.
    # Let's initialize with these primes and residues r = 1 (or optimized later).
    # A better init: Use the known structure if possible, or just random residues for these primes.
    # Let's use r=1 for all initially, then SA will optimize.
    # Wait, r=1 for p=13 covers 1, 14, 27...
    # r=1 for p=17 covers 1, 18...
    # This might not cover 2.
    # Let's try to cover 1..K greedily first to get a good start.
    
    # Greedy Init:
    current_classes = []
    current_budget = Fraction(1)
    
    # We want to cover 1, 2, 3...
    # Let's just use the primes >= 13 with random residues first, it's fast.
    # Or better, use the specific primes from the seed program but randomize residues?
    # The seed program's specific residues were likely tuned or just default.
    # Let's start with the set of primes {13, 17, ..., 997} and random residues.
    
    current_primes = seed_primes[:]
    current_residues = [random.randint(0, p-1) for p in current_primes]
    
    # Check budget
    sum_recip = sum(Fraction(1, p) for p in current_primes)
    if sum_recip > 1:
        # This shouldn't happen for primes >= 13, sum is ~0.92
        pass
        
    # --- Data Structures for Fast Evaluation ---
    # We maintain a count array to know how many classes cover each number.
    # This allows us to determine the "first gap" quickly and evaluate moves faster.
    # But updating counts for a move is O(L/p).
    
    # Let's implement a class to manage the state and coverage counts.
    
    class CoveringState:
        def __init__(self, primes, residues, L_cap):
            self.primes = primes
            self.residues = residues
            self.L_cap = L_cap
            self.counts = [0] * (L_cap + 1)
            self._update_all()
            
        def _update_all(self):
            # Reset counts
            self.counts = [0] * (self.L_cap + 1)
            for i, p in enumerate(self.primes):
                r = self.residues[i]
                # Mark coverage
                start = r if r >= 1 else r + p
                for n in range(start, self.L_cap + 1, p):
                    self.counts[n] += 1
                    
        def get_L(self):
            # Find first gap
            for i in range(1, self.L_cap + 1):
                if self.counts[i] == 0:
                    return i - 1
            return self.L_cap
            
        def apply_move(self, idx, new_p, new_r):
            """
            Apply a move: replace class idx with (new_p, new_r).
            Returns delta in L (new_L - old_L) or just new_L for comparison.
            To be fast, we update counts incrementally.
            """
            old_p = self.primes[idx]
            old_r = self.residues[idx]
            
            # If no change, return
            if old_p == new_p and old_r == new_r:
                return self.get_L()
            
            # 1. Remove old class contribution
            start_old = old_r if old_r >= 1 else old_r + old_p
            for n in range(start_old, self.L_cap + 1, old_p):
                self.counts[n] -= 1
                
            # 2. Add new class contribution
            start_new = new_r if new_r >= 1 else new_r + new_p
            for n in range(start_new, self.L_cap + 1, new_p):
                self.counts[n] += 1
                
            # 3. Update state
            self.primes[idx] = new_p
            self.residues[idx] = new_r
            
            # 4. Calculate new L
            # Optimization: We know the old L. The new L is either:
            # a) The old L (if the gap was filled and no new gap appeared before old L)
            # b) Smaller (if a new gap appeared before old L)
            # c) Larger (if the gap at old L+1 is filled, etc.)
            
            # Simplest robust way: scan from 1 again? No, that's O(L).
            # We can scan from the current known L?
            # Actually, if we remove a class, a gap might appear anywhere.
            # If we add a class, a gap might be filled.
            
            # Let's just scan from 1 to find the first 0.
            # Since L is typically ~1000-5000, and we have ~115s, 
            # even O(L) per move is fine if L is small.
            # But L can go up to 100,000.
            # If L is large, the scan is expensive.
            
            # Hybrid approach:
            # Maintain a set of "gaps" (indices where count is 0)?
            # Updating a set of gaps on every +/- 1 is complex.
            
            # Let's stick to scanning from 1, but break early.
            # If we are stuck at L=1000, scan is fast.
            # If we reach L=10000, scan is 10000 ops. 10000 * 1000 moves = 10M ops. Fast in Python? Maybe.
            # 10M simple loop iterations in Python takes ~0.5-1s.
            # We have 115s. So we can afford ~100,000 moves if each takes 1ms.
            # Scanning 10,000 integers takes ~1ms in Python.
            # So we can do ~100,000 moves. That's enough for SA.
            
            L = 0
            for i in range(1, self.L_cap + 1):
                if self.counts[i] > 0:
                    L = i
                else:
                    break
            return L

    # Initialize state
    state = CoveringState(current_primes, current_residues, L_target)
    current_L = state.get_L()
    
    best_L = current_L
    best_primes = current_primes[:]
    best_residues = current_residues[:]
    
    # --- Simulated Annealing ---
    # Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Pre-compute costs for budget check
    # Budget constraint: sum(1/p) <= 1
    # We check this before applying a move.
    
    # Helper to check if a set of primes is valid budget-wise
    # We maintain current_sum as a Fraction for exactness, or float for speed?
    # Fraction is exact but slow. Float is fast but risky for "at most 1".
    # Given primes are integers, sum(1/p) is a rational.
    # We can use Fraction for the final check, but for SA moves, we can use a cached sum.
    
    current_sum = sum(Fraction(1, p) for p in current_primes)
    
    # Available primes pool
    all_primes_set = set(primes)
    
    num_moves = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Change residue of an existing prime
            if not current_primes:
                continue
            idx = random.randint(0, len(current_primes) - 1)
            p = current_primes[idx]
            old_r = current_residues[idx]
            
            # Smart residue selection:
            # 1. Try residue that covers the first gap (current_L + 1)
            # 2. Try random residue
            # 3. Try residue that covers a number near current_L
            
            candidates = set()
            
            # Target the first gap
            gap = current_L + 1
            if gap <= L_target:
                # We want r such that gap % p == r
                r_gap = gap % p
                candidates.add(r_gap)
                
            # Try to cover current_L + 2, etc?
            # Just a few random ones
            for _ in range(3):
                candidates.add(random.randint(0, p - 1))
                
            # Also try old_r + 1, old_r - 1
            candidates.add((old_r + 1) % p)
            candidates.add((old_r - 1) % p)
            
            # Pick one candidate
            new_r = random.choice(list(candidates))
            
            # Apply move
            new_L = state.apply_move(idx, p, new_r)
            
            # Evaluate
            delta_L = new_L - current_L
            
            if delta_L >= 0:
                # Accept
                current_L = new_L
                no_improve_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_primes = current_primes[:]
                    best_residues = current_residues[:]
            else:
                # Reject with probability exp(delta_L / T)
                # Since delta_L is negative integer, this is very small.
                # For L jumps, we might want to accept small drops to escape.
                # But dropping L by 1 is bad.
                # Let's use a different acceptance criterion:
                # Accept if random() < exp(delta_L / T)
                # If delta_L is -1, exp(-1/T). If T=1, 0.36.
                # If T=0.1, 0.00004.
                accept_prob = pow(2.71828, delta_L / T)
                if random.random() < accept_prob:
                    current_L = new_L
                    no_improve_count += 1
                else:
                    # Revert
                    state.apply_move(idx, p, old_r)
                    no_improve_count += 1
                    
        elif move_type < 0.9:
            # Swap a prime
            if not current_primes:
                continue
            idx = random.randint(0, len(current_primes) - 1)
            p_old = current_primes[idx]
            r_old = current_residues[idx]
            
            # Pick new prime
            # Prefer primes that are not currently in the set
            # And fit in budget
            # Budget check: current_sum - 1/p_old + 1/p_new <= 1
            
            # To speed up, pick from a subset of candidates?
            # Just pick random from all primes.
            p_new = random.choice(primes)
            
            if p_new == p_old:
                continue
                
            if p_new in current_primes:
                # If p_new is already in the set, we can't have duplicates.
                # We could swap with another index, but that's a 2-swap.
                # For now, skip if duplicate.
                continue
                
            # Budget check
            new_sum = current_sum - Fraction(1, p_old) + Fraction(1, p_new)
            if new_sum > 1:
                continue
                
            # Apply move
            # Residue: keep r_old? Or adjust?
            # r_old might be >= p_new.
            r_new = r_old % p_new
            
            new_L = state.apply_move(idx, p_new, r_new)
            
            # Update budget
            old_sum = current_sum
            current_sum = new_sum
            
            # Evaluate
            delta_L = new_L - current_L
            
            if delta_L >= 0:
                current_L = new_L
                no_improve_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_primes = current_primes[:]
                    best_residues = current_residues[:]
            else:
                accept_prob = pow(2.71828, delta_L / T)
                if random.random() < accept_prob:
                    current_L = new_L
                    no_improve_count += 1
                else:
                    # Revert
                    state.apply_move(idx, p_old, r_old)
                    current_sum = old_sum
                    no_improve_count += 1
                    
        else:
            # Add a prime (if budget allows)
            # Only if we have space in budget
            # We need to find a prime p such that current_sum + 1/p <= 1
            # And p is not in current_primes
            
            # To make this efficient, we can just try a random prime and check.
            p_add = random.choice(primes)
            if p_add in current_primes:
                continue
                
            new_sum = current_sum + Fraction(1, p_add)
            if new_sum > 1:
                continue
                
            # Pick residue
            # Try to cover current_L + 1
            gap = current_L + 1
            r_add = gap % p_add
            
            # Apply move
            # We need to append to state
            # My CoveringState class doesn't support append easily without re-init.
            # Let's modify apply_move to handle append?
            # Or just re-initialize state if we add/remove?
            # Re-initialization is O(K * L/p). K~100, L/p~1000. 100k ops.
            # If we do this rarely, it's fine.
            
            # Let's just do a full re-init for add/remove to keep code simple and robust.
            # But we need to revert if rejected.
            
            # Save state
            old_primes = current_primes[:]
            old_residues = current_residues[:]
            old_sum = current_sum
            
            # Add
            current_primes.append(p_add)
            current_residues.append(r_add)
            current_sum = new_sum
            
            # Re-init state
            state = CoveringState(current_primes, current_residues, L_target)
            new_L = state.get_L()
            
            delta_L = new_L - current_L
            
            if delta_L >= 0:
                current_L = new_L
                no_improve_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_primes = current_primes[:]
                    best_residues = current_residues[:]
            else:
                accept_prob = pow(2.71828, delta_L / T)
                if random.random() < accept_prob:
                    current_L = new_L
                    no_improve_count += 1
                else:
                    # Revert
                    current_primes = old_primes
                    current_residues = old_residues
                    current_sum = old_sum
                    state = CoveringState(current_primes, current_residues, L_target)
                    current_L = state.get_L() # Should be same as before
                    no_improve_count += 1

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Restart if stuck?
        if no_improve_count > 1000:
            # Randomize a bit
            for i in range(len(current_primes)):
                current_residues[i] = random.randint(0, current_primes[i]-1)
            state = CoveringState(current_primes, current_residues, L_target)
            current_L = state.get_L()
            no_improve_count = 0
            T = 1.0 # Reheat

    # Return best found
    # Ensure validity
    final_sum = sum(Fraction(1, p) for p in best_primes)
    if final_sum > 1:
        # Fallback to seed
        return [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
        
    return [[p, r] for p, r in zip(best_primes, best_residues)]

# To ensure we don't return an empty list if something goes wrong, 
# we can wrap in a try-except in the actual construct if needed, 
# but the logic above is robust.
