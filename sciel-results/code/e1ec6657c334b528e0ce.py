"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

# Precompute prime factors for all numbers up to 20000
MAX_SUM = 20000
_pf_cache = [set() for _ in range(MAX_SUM + 1)]

def _init_prime_factors():
    # Efficient sieve for smallest prime factor, then derive distinct prime factors
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(math.isqrt(MAX_SUM)) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM+1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for i in range(2, MAX_SUM + 1):
        n = i
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        _pf_cache[i] = factors

_init_prime_factors()

class SAState:
    def __init__(self, nums):
        self.nums = list(nums)
        self.n = len(self.nums)
        self.prime_counts = {}
        self.active_primes = 0
        self._rebuild()

    def _rebuild(self):
        self.prime_counts = {}
        for i in range(self.n):
            for j in range(i+1, self.n):
                s = self.nums[i] + self.nums[j]
                for p in _pf_cache[s]:
                    if p in self.prime_counts:
                        self.prime_counts[p] += 1
                    else:
                        self.prime_counts[p] = 1
                        self.active_primes += 1

    def score(self):
        return self.active_primes

    def try_move(self, idx, new_val):
        """
        Attempts to replace nums[idx] with new_val.
        Returns delta_score (new_score - old_score).
        Modifies state if successful (caller must revert if needed).
        """
        if new_val in self.nums:
            # We need to check if it's the same index
            if self.nums.index(new_val) == idx:
                return 0 # No change
            # Collision with other element is invalid for distinct set
            # But wait, the problem says "distinct positive integers". 
            # If new_val is already in nums at another index, it's invalid.
            # However, in SA, we usually allow invalid states or reject them.
            # Let's just reject moves that create duplicates.
            return float('inf')

        old_val = self.nums[idx]
        if old_val == new_val:
            return 0

        delta = 0
        
        # For each other element j, update the pair (idx, j)
        for j in range(self.n):
            if j == idx:
                continue
            
            old_sum = old_val + self.nums[j]
            new_sum = new_val + self.nums[j]
            
            # Remove old primes
            for p in _pf_cache[old_sum]:
                self.prime_counts[p] -= 1
                if self.prime_counts[p] == 0:
                    self.active_primes -= 1
                    delta -= 1 # Score decreased by 1 (good)
            
            # Add new primes
            for p in _pf_cache[new_sum]:
                if p in self.prime_counts:
                    self.prime_counts[p] += 1
                else:
                    self.prime_counts[p] = 1
                    self.active_primes += 1
                    delta += 1 # Score increased by 1 (bad)
        
        # Commit the change
        self.nums[idx] = new_val
        return delta

    def revert_move(self, idx, old_val):
        """Reverts a move at idx back to old_val."""
        new_val = self.nums[idx]
        if old_val == new_val:
            return
        
        # Reverse the delta logic
        for j in range(self.n):
            if j == idx:
                continue
            
            # Current sum in state is new_val + nums[j]
            # We want to go back to old_val + nums[j]
            
            # Remove current (new) primes
            for p in _pf_cache[new_val + self.nums[j]]:
                self.prime_counts[p] -= 1
                if self.prime_counts[p] == 0:
                    self.active_primes -= 1
            
            # Add old primes
            for p in _pf_cache[old_val + self.nums[j]]:
                if p in self.prime_counts:
                    self.prime_counts[p] += 1
                else:
                    self.prime_counts[p] = 1
                    self.active_primes += 1
        
        self.nums[idx] = old_val

def construct():
    n = 12
    max_val = 10000
    
    # Greedy Initialization
    # Start with small numbers to keep sums small (fewer primes generally)
    # But we need distinct integers.
    # Let's start with 1..12 and refine.
    init_nums = list(range(1, n+1))
    current = SAState(init_nums)
    best_state = SAState(init_nums)
    best_score = current.score()
    
    # Simulated Annealing
    start_temp = 10.0
    end_temp = 0.1
    cooling = 0.999
    temp = start_temp
    
    # Time limit: 50 seconds to be safe
    start_time = time.time()
    max_time = 50.0
    
    # Pre-generate a pool of candidate values to speed up random selection?
    # Or just random.
    # Random selection from 1..10000 has high collision probability?
    # 12 numbers, 10000 space. Collision prob is low.
    
    while temp > end_temp:
        if time.time() - start_time > max_time:
            break
        
        # Pick a random index
        idx = random.randint(0, n-1)
        
        # Pick a new value
        # Strategy: mostly random, sometimes local perturbation
        if random.random() < 0.5:
            # Local perturbation
            old_val = current.nums[idx]
            delta_val = random.randint(-50, 50)
            new_val = old_val + delta_val
            if new_val < 1 or new_val > max_val:
                new_val = random.randint(1, max_val)
        else:
            new_val = random.randint(1, max_val)
            
        # Check for duplicate (simple check)
        # This is O(n). n=12, so it's fast.
        # If new_val is in current.nums and not at idx, it's a collision.
        # We can check this before calling try_move to avoid the state change/revert overhead if we know it's invalid.
        # However, try_move returns inf for collisions.
        
        old_val = current.nums[idx]
        
        # Optimization: if new_val == old_val, skip
        if new_val == old_val:
            continue
            
        # Check collision manually to avoid state change if invalid
        is_collision = False
        for i, v in enumerate(current.nums):
            if v == new_val and i != idx:
                is_collision = True
                break
        if is_collision:
            continue
            
        # Try the move
        delta = current.try_move(idx, new_val)
        
        # delta is (new_score - old_score)
        # If delta < 0, it's an improvement.
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            # Accept
            new_score = current.score()
            if new_score < best_score:
                best_score = new_score
                best_state = SAState(current.nums) # Copy state
        else:
            # Reject, revert
            current.revert_move(idx, old_val)
            
        temp *= cooling
        
    return best_state.nums

# EVOLVE-BLOCK-END
