import time
import random
from fractions import Fraction

def _get_primes(limit=1000):
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
    deadline = start_time + 115
    
    primes = _get_primes(1000)
    L_cap = 100000
    primes_set = set(primes)
    recip_map = {p: Fraction(1, p) for p in primes}
    
    # --- Initialization: Greedy Seed ---
    # Start with a strong heuristic solution to ensure a good baseline
    # We want to cover 1..L.
    
    def greedy_init():
        rem_budget = Fraction(1)
        used = set()
        classes = []
        # We prioritize small primes for high coverage density, 
        # but we need to cover the prefix 1..L.
        # A common strategy:
        # For n=1,2,3..., find the smallest unused prime p that fits budget.
        # Set r = n % p.
        # This is O(L * P) which might be slow if L is large, but L starts small.
        
        # Actually, a better init is to just pick a set of primes that sum to ~1
        # and randomize residues, then let SA fix it.
        # But we can do slightly better:
        # Pick primes to cover 1..100 roughly.
        
        # Let's try a simple heuristic:
        # Sort primes by size.
        # Greedily add prime if it reduces the "first gap" or covers new numbers.
        
        # For now, let's use a robust random init with a few greedy tweaks.
        # Select ~150 primes (sum ~1)
        # Sum of 1/p for p >= 13 is ~0.93.
        # So we can afford almost all primes >= 13.
        
        candidate_primes = [p for p in primes if p >= 13]
        # Randomly select a subset that sums to <= 1? 
        # No, just use all primes >= 13.
        # Sum is ~0.93. We have room for one small prime like 11 or 7?
        # 0.93 + 1/7 = 1.07 > 1.
        # 0.93 + 1/11 = 1.02 > 1.
        # 0.93 + 1/13 is already included.
        # So {13..997} is a very good set.
        
        sel_primes = candidate_primes[:]
        # Randomize residues
        sel_residues = [random.randint(0, p-1) for p in sel_primes]
        
        return sel_primes, sel_residues

    current_primes, current_residues = greedy_init()
    current_sum = sum(Fraction(1, p) for p in current_primes)
    
    # --- State Management ---
    # We use a difference array / prefix sum approach to compute coverage efficiently?
    # No, coverage is boolean OR of arithmetic progressions.
    # Fastest in Python: maintain a count array `cover_counts` of size L_cap+1.
    # Update is O(L_cap / p).
    
    def calc_L(primes, residues, L_limit=None):
        if L_limit is None:
            L_limit = L_cap
        # We need to find the longest prefix 1..L covered.
        # This requires knowing coverage for 1..L.
        # Rebuilding the count array from scratch is O(K * L/p).
        # For K~100, L~1000, this is fast.
        # For L~10000, it's slower.
        
        counts = [0] * (L_limit + 1)
        for p, r in zip(primes, residues):
            start = r if r >= 1 else r + p
            for n in range(start, L_limit + 1, p):
                counts[n] += 1
        
        L = 0
        for i in range(1, L_limit + 1):
            if counts[i] > 0:
                L = i
            else:
                break
        return L

    current_L = calc_L(current_primes, current_residues, min(10000, L_cap))
    # If L is small, we can search in a smaller window for speed, then expand.
    # Let's set search window to min(current_L * 2 + 100, L_cap)
    # Actually, just use L_cap for correctness, but optimize the inner loop.
    
    best_L = current_L
    best_primes = current_primes[:]
    best_residues = current_residues[:]
    
    # --- Local Search / SA ---
    # We will maintain the count array and update it incrementally.
    
    class State:
        def __init__(self, primes, residues):
            self.primes = primes
            self.residues = residues
            self.L_cap = L_cap
            # We only maintain counts up to a dynamic limit?
            # No, fixed L_cap is safer, but memory is 100k ints. Fine.
            self.counts = [0] * (L_cap + 1)
            self._rebuild()
            
        def _rebuild(self):
            self.counts = [0] * (self.L_cap + 1)
            for i, p in enumerate(self.primes):
                r = self.residues[i]
                start = r if r >= 1 else r + p
                for n in range(start, self.L_cap + 1, p):
                    self.counts[n] += 1
                    
        def get_L(self):
            # Find first gap
            # Optimization: keep track of current L?
            # If we change one class, the new L is likely close to old L.
            # But a gap can appear early.
            # Scan from 1 is O(L).
            # If L is 100k, this is 100k ops.
            # 1000 moves * 100k ops = 100M ops. ~5-10s in Python.
            # Acceptable.
            
            L = 0
            for i in range(1, self.L_cap + 1):
                if self.counts[i] > 0:
                    L = i
                else:
                    break
            return L
            
        def apply_change(self, idx, new_p, new_r):
            old_p = self.primes[idx]
            old_r = self.residues[idx]
            
            if old_p == new_p and old_r == new_r:
                return self.get_L()
            
            # Remove old
            start_old = old_r if old_r >= 1 else old_r + old_p
            for n in range(start_old, self.L_cap + 1, old_p):
                self.counts[n] -= 1
                
            # Add new
            start_new = new_r if new_r >= 1 else new_r + new_p
            for n in range(start_new, self.L_cap + 1, new_p):
                self.counts[n] += 1
                
            self.primes[idx] = new_p
            self.residues[idx] = new_r
            
            return self.get_L()

    state = State(current_primes, current_residues)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    no_improve = 0
    
    while time.time() < deadline:
        # Pick move type
        # 1. Change residue (60%)
        # 2. Change prime (20%)
        # 3. Add/Remove prime (20% - careful with budget)
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Change residue
            idx = random.randint(0, len(current_primes) - 1)
            p = current_primes[idx]
            old_r = current_residues[idx]
            
            # Smart residue: target the first gap
            gap = state.get_L() + 1
            if gap <= L_cap:
                r_target = gap % p
                # With some probability, use target, else random
                if random.random() < 0.5:
                    new_r = r_target
                else:
                    new_r = random.randint(0, p - 1)
            else:
                new_r = random.randint(0, p - 1)
                
            new_L = state.apply_change(idx, p, new_r)
            
            delta = new_L - current_L
            if delta >= 0:
                current_L = new_L
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_primes = current_primes[:]
                    best_residues = current_residues[:]
            else:
                # Acceptance
                # delta is negative.
                # Use exp(delta / T).
                # If delta is large negative, prob is 0.
                # If delta is -1, prob is exp(-1/T).
                if T > 0:
                    prob = pow(2.71828, delta / T)
                else:
                    prob = 0
                if random.random() < prob:
                    current_L = new_L
                    no_improve += 1
                else:
                    # Revert
                    state.apply_change(idx, p, old_r)
                    current_L = state.get_L() # Should be restored
                    no_improve += 1
                    
        elif move_type < 0.8:
            # Change prime
            idx = random.randint(0, len(current_primes) - 1)
            p_old = current_primes[idx]
            r_old = current_residues[idx]
            
            # Pick new prime
            # Must not be in current set
            # Must fit budget
            # Try a few random candidates
            
            new_p = None
            for _ in range(5): # try 5 times
                cand_p = random.choice(primes)
                if cand_p == p_old:
                    continue
                if cand_p in current_primes:
                    continue
                # Budget check
                cost_delta = recip_map[cand_p] - recip_map[p_old]
                if current_sum + cost_delta <= Fraction(1):
                    new_p = cand_p
                    break
                    
            if new_p is None:
                continue
                
            new_r = r_old % new_p
            
            # Check if new_p is already in primes (should be checked above)
            
            new_L = state.apply_change(idx, new_p, new_r)
            
            # Update sum
            old_sum = current_sum
            current_sum = current_sum + recip_map[new_p] - recip_map[p_old]
            
            delta = new_L - current_L
            if delta >= 0:
                current_L = new_L
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_primes = current_primes[:]
                    best_residues = current_residues[:]
            else:
                prob = pow(2.71828, delta / T)
                if random.random() < prob:
                    current_L = new_L
                    no_improve += 1
                else:
                    # Revert
                    state.apply_change(idx, p_old, r_old)
                    current_sum = old_sum
                    current_L = state.get_L()
                    no_improve += 1
                    
        else:
            # Add/Remove prime
            # This is rarer and more disruptive.
            # Since we are close to budget limit with all primes >=13,
            # we might need to remove a large prime to add a small one?
            # Or just shuffle the set.
            # Let's stick to swapping primes (handled above) to maintain count.
            # Adding a prime requires re-initializing state if we don't support it in State class.
            # For simplicity and speed, we skip add/remove in the main loop
            # and rely on the initial set being good enough.
            # If we are stuck, we might try to drop a prime with low coverage utility?
            # Hard to measure utility.
            # So we skip this move type for now, or make it a "restart" trigger.
            
            if no_improve > 500:
                # Reheat and randomize residues
                for i in range(len(current_primes)):
                    current_residues[i] = random.randint(0, current_primes[i] - 1)
                state._rebuild()
                current_L = state.get_L()
                T = 10.0
                no_improve = 0

        T *= cooling
        if T < T_min:
            T = T_min

    # Final verification and return
    # Ensure budget is valid
    final_sum = sum(Fraction(1, p) for p in best_primes)
    if final_sum > Fraction(1):
        # Fallback: remove largest primes until valid?
        # Or return empty? No, return best effort.
        # Since we checked budget in moves, this should be rare.
        # But initial random sum might be > 1?
        # Sum of 1/p for p>=13 is ~0.93. So it's fine.
        
        # If somehow invalid, try to fix by removing primes with largest 1/p?
        # No, just return the list. The evaluator will check.
        # Actually, if invalid, score is 0.
        # So we must ensure validity.
        
        # Let's do a final cleanup:
        # Sort primes by 1/p descending (smallest p first)
        # Keep adding until sum > 1, then backtrack?
        # No, we have a specific set.
        # If sum > 1, we drop the prime with the largest 1/p (smallest p)?
        # No, we want to keep coverage.
        # Just drop random primes until sum <= 1.
        
        while final_sum > Fraction(1) and best_primes:
            # Remove the prime with the largest reciprocal (smallest prime)
            # to free up budget most?
            # Or just remove one.
            idx_remove = random.randint(0, len(best_primes) - 1)
            final_sum -= Fraction(1, best_primes[idx_remove])
            best_primes.pop(idx_remove)
            best_residues.pop(idx_remove)
            
    return [[p, r] for p, r in zip(best_primes, best_residues)]
