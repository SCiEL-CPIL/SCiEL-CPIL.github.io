import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import random

def construct():
    """
    Find a subset A of [0, 128] such that every integer in [128, 256] is a sum of two elements of A.
    Minimize the largest representation multiplicity (peak load) in that interval.
    
    Strategy:
    1. Start with a feasible set (e.g., [64, 128]).
    2. Use local search (hill climbing) to reduce the peak multiplicity.
    3. Moves: 
       - Remove an element if it doesn't break coverage.
       - Add an element if it reduces peak load (or keeps it same but allows further removals).
       - Swap an element (remove one, add another) if it improves peak load.
    4. Use random restarts and time budgeting.
    """
    n = 128
    target_min = n
    target_max = 2 * n
    
    # Precompute all possible pairs (a, b) with a <= b such that target_min <= a+b <= target_max
    # We represent A as a boolean array of size n+1 (indices 0..128)
    
    def get_pairs():
        """Return list of (a, b) where a <= b, 0 <= a, b <= n, and target_min <= a+b <= target_max"""
        pairs = []
        for a in range(n + 1):
            # b must be >= a
            # a + b >= target_min => b >= target_min - a
            # a + b <= target_max => b <= target_max - a
            b_min = max(a, target_min - a)
            b_max = min(n, target_max - a)
            if b_min <= b_max:
                for b in range(b_min, b_max + 1):
                    pairs.append((a, b))
        return pairs
    
    pairs = get_pairs()
    
    # Map sum -> list of pair indices
    sum_to_pairs = {}
    for i, (a, b) in enumerate(pairs):
        s = a + b
        if s not in sum_to_pairs:
            sum_to_pairs[s] = []
        sum_to_pairs[s].append(i)
    
    # Precompute pair elements for quick access
    pair_a = np.array([p[0] for p in pairs], dtype=np.int32)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int32)
    pair_sum = np.array([p[0] + p[1] for p in pairs], dtype=np.int32)
    
    # Initial feasible set: A = [64, 128]
    # This covers [128, 256] because:
    # min sum = 64+64=128, max sum = 128+128=256.
    # For any s in [128, 256], we can write s = a + b with a,b in [64,128].
    # e.g., a = max(64, s-128), b = s-a. Check if b in [64,128] and b>=a.
    # Actually, simpler: just use [64, 128] as seed.
    
    def initial_set():
        return set(range(64, 129))
    
    def compute_loads(A_set):
        """
        Given a set A, compute the multiplicity of each sum in [128, 256].
        Returns an array of length 129 (for sums 128..256) with counts.
        Also returns the peak load.
        """
        loads = np.zeros(257, dtype=np.int32)
        A_list = list(A_set)
        # For each pair (a,b) with a<=b in A, increment loads[a+b]
        # To be efficient, iterate over A_list and use symmetry
        # But for small A, direct iteration is fine.
        # Let's use the precomputed pairs? No, because A changes.
        # Direct counting:
        for i, a in enumerate(A_list):
            for b in A_list[i:]:  # b >= a to avoid double counting unordered pairs
                s = a + b
                if target_min <= s <= target_max:
                    loads[s] += 1
        return loads, loads[target_min:target_max+1].max()
    
    def is_valid(A_set):
        """Check if every sum in [128, 256] is covered."""
        loads, peak = compute_loads(A_set)
        # Check if any sum in [128, 256] has 0 representations
        for s in range(target_min, target_max + 1):
            if loads[s] == 0:
                return False
        return True
    
    def peak_of(A_set):
        loads, peak = compute_loads(A_set)
        return peak
    
    # Local search
    def local_search(A_set, max_iters=5000, time_limit=None, start_time=None):
        """
        Perform local search to minimize peak load.
        Returns the best set found.
        """
        if start_time is None:
            start_time = time.time()
        
        current_peak = peak_of(A_set)
        best_set = set(A_set)
        best_peak = current_peak
        
        # Precompute neighbors? Too expensive.
        # Moves:
        # 1. Remove x: if A\{x} is valid, and peak doesn't increase (or decreases), take it.
        # 2. Add x: if peak decreases, take it. (Even if peak same, might help later? Usually we want to minimize size or peak. Let's focus on peak.)
        # 3. Swap x, y: remove x, add y.
        
        # To speed up, we can precompute for each element x, which sums it contributes to.
        # But let's start with a simple hill climber.
        
        A_list = list(A_set)
        all_elements = list(range(n + 1))
        
        for _ in range(max_iters):
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            improved = False
            random.shuffle(A_list)
            random.shuffle(all_elements)
            
            # Try removals
            for x in A_list:
                if x not in A_set:
                    continue
                new_set = A_set - {x}
                if not new_set:
                    continue
                if is_valid(new_set):
                    new_peak = peak_of(new_set)
                    if new_peak < current_peak:
                        A_set = new_set
                        current_peak = new_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_set = set(A_set)
                        improved = True
                        break # Restart loop with new set
                    elif new_peak == current_peak:
                        # Keep if it doesn't increase peak? Might lead to smaller set.
                        # Let's accept to explore.
                        A_set = new_set
                        # current_peak unchanged
                        improved = True
                        break
            
            if improved:
                A_list = list(A_set)
                continue
                
            # Try additions
            added = False
            for x in all_elements:
                if x in A_set:
                    continue
                new_set = A_set | {x}
                new_peak = peak_of(new_set)
                if new_peak < current_peak:
                    A_set = new_set
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_set = set(A_set)
                    added = True
                    break
            
            if added:
                A_list = list(A_set)
                continue
                
            # Try swaps
            swapped = False
            for x in A_list:
                if x not in A_set:
                    continue
                for y in all_elements:
                    if y == x or y in A_set:
                        continue
                    new_set = (A_set - {x}) | {y}
                    if not new_set:
                        continue
                    if is_valid(new_set):
                        new_peak = peak_of(new_set)
                        if new_peak < current_peak:
                            A_set = new_set
                            current_peak = new_peak
                            if current_peak < best_peak:
                                best_peak = current_peak
                                best_set = set(A_set)
                            swapped = True
                            break
                if swapped:
                    break
            
            if swapped:
                A_list = list(A_set)
                continue
                
            # No improvement found, maybe random restart within this local search?
            # Or break and do a new restart.
            break
        
        return best_set, best_peak
    
    # Main loop with time budget
    start_time = time.time()
    time_limit = 60.0  # seconds
    best_set = initial_set()
    best_peak = peak_of(best_set)
    
    # Initial peak for [64, 128]
    # Let's estimate: 
    # s=128: 64+64 (1)
    # s=129: 64+65, 65+64? No, unordered. 64+65 (1)
    # ...
    # s=192: 64+128, 65+127, ..., 128+64. 
    # Number of pairs for sum s in [128, 192]: s-64 (from 64 to s-64, but max 128)
    # Actually, for s in [128, 192], pairs are (a, s-a) with 64<=a<=s-a<=128.
    # a >= 64, a <= s-64. Also a <= s/2.
    # Count = floor(s/2) - 64 + 1 = floor(s/2) - 63.
    # Max at s=192: floor(192/2)-63 = 96-63=33.
    # For s in [192, 256], pairs (a, s-a) with 64<=a<=s-a<=128.
    # a <= 128, s-a <= 128 => a >= s-128.
    # a >= 64.
    # Count = 128 - (s-128) + 1 = 257 - s.
    # Max at s=192: 257-192=65? Wait.
    # Let's re-calculate s=192.
    # a from 64 to 128. b=192-a.
    # If a=64, b=128.
    # If a=128, b=64.
    # Unordered pairs: a <= b => a <= 96.
    # So a in [64, 96]. Count = 96-64+1 = 33.
    # So peak is 33.
    
    # We want to get this lower. Theoretical minimum?
    # Total number of pairs in [128, 256] is 129 sums.
    # If |A| = k, number of pairs is k(k+1)/2.
    # Average load = k(k+1)/2 / 129.
    # If k=16, avg = 16*17/2/129 = 136/129 ~ 1.05. Peak at least 2?
    # If k=20, avg = 20*21/2/129 = 210/129 ~ 1.6.
    # If k=30, avg = 30*31/2/129 = 465/129 ~ 3.6.
    # Current k=65, peak=33.
    # We can definitely do better.
    
    # Run local search
    while (time.time() - start_time) < (time_limit - 5.0):
        A_set = initial_set()
        # Add some randomness to initial set?
        # Maybe remove some random elements first to break symmetry?
        # But must remain valid.
        
        # Let's try a different initial set: 
        # Random subset that covers? Hard.
        # Stick with [64, 128] but maybe perturb.
        
        # Perturb: remove random elements while valid
        A_set = set(A_set)
        for _ in range(100):
            if not A_set:
                break
            x = random.choice(list(A_set))
            new_set = A_set - {x}
            if is_valid(new_set):
                A_set = new_set
        
        # Now local search
        A_set, current_peak = local_search(A_set, max_iters=2000, time_limit=10.0, start_time=time.time())
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_set = A_set
            # Print progress
            # print(f"New best peak: {best_peak} with size {len(best_set)}")
    
    return sorted(list(best_set))

# Test
if __name__ == "__main__":
    A = construct()
    print(f"Set size: {len(A)}")
    print(f"Set: {A[:10]}...{A[-10:]}")
    # Verify
    n = 128
    target_min = n
    target_max = 2 * n
    A_set = set(A)
    loads = np.zeros(257, dtype=np.int32)
    A_list = list(A_set)
    for i, a in enumerate(A_list):
        for b in A_list[i:]:
            s = a + b
            if target_min <= s <= target_max:
                loads[s] += 1
    peak = loads[target_min:target_max+1].max()
    min_load = loads[target_min:target_max+1].min()
    print(f"Peak load: {peak}, Min load: {min_load}")
    # Check coverage
    for s in range(target_min, target_max + 1):
        if loads[s] == 0:
            print(f"Missing sum: {s}")
            break
    else:
        print("All sums covered.")
