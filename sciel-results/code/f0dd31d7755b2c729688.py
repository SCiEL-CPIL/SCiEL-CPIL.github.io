"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from itertools import combinations

P = {'n': 256}
_DEADLINE = None

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            for k in range(j+1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 85  # 85s budget, 5s safety margin
    random.seed(42)
    
    n = P['n']
    nums = list(range(2, n + 1))
    N = len(nums)
    
    # Precompute forbidden pairs for each x
    # forbidden[x] = list of (i, j) indices such that nums[i] and nums[j] are distinct 
    # and {x, nums[i], nums[j]} is invalid.
    
    forbidden = [list() for _ in range(N)]
    
    for x_idx, x in enumerate(nums):
        for i in range(N):
            if i == x_idx: continue
            a = nums[i]
            for j in range(i+1, N):
                if j == x_idx: continue
                b = nums[j]
                # Check if {x, a, b} is invalid
                # x | a*b OR a | x*b OR b | x*a
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    forbidden[x_idx].append((i, j))
    
    # Convert to frozensets for faster membership testing
    forbidden_sets = [set(f) for f in forbidden]
    forbidden_lens = [len(f) for f in forbidden]
    
    # Precompute primes
    primes_idx = [i for i, v in enumerate(nums) if _is_prime(v)]
    primes_set = set(primes_idx)
    others_idx = [i for i in range(N) if i not in primes_set]
    
    # Fast can_add using precomputed forbidden list
    def can_add(S_indices, x_idx):
        """S_indices is a set of indices. Check if x_idx can be added."""
        for (i, j) in forbidden[x_idx]:
            if i in S_indices and j in S_indices:
                return False
        return True
    
    # Greedy search with a given order
    def greedy_search(order):
        S = set()
        for idx in order:
            if can_add(S, idx):
                S.add(idx)
        return S
    
    # Generate diverse greedy orders
    def generate_orders():
        orders = []
        
        # Deterministic orders
        safe_sorted = sorted(range(N), key=lambda i: forbidden_lens[i])
        danger_sorted = sorted(range(N), key=lambda i: -forbidden_lens[i])
        
        # Primes first, then safe
        orders.append(primes_idx + [i for i in safe_sorted if i not in primes_set])
        # Primes first, then danger
        orders.append(primes_idx + [i for i in danger_sorted if i not in primes_set])
        # All safe
        orders.append(safe_sorted)
        # All danger
        orders.append(danger_sorted)
        # Natural order
        orders.append(list(range(N)))
        # Reverse natural
        orders.append(list(range(N-1, -1, -1)))
        
        # Random orders (shuffled)
        for _ in range(20):
            shuffled = list(range(N))
            random.shuffle(shuffled)
            orders.append(shuffled)
        
        # Hybrid: primes first, then random
        for _ in range(10):
            others_shuffled = list(others_idx)
            random.shuffle(others_shuffled)
            orders.append(primes_idx + others_shuffled)
        
        return orders
    
    # Local search: 1-for-1 swaps
    def local_search_1_for_1(S, max_iter=100):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < _DEADLINE:
            improved = False
            iters += 1
            S_list = list(S)
            random.shuffle(S_list)
            for rem in S_list:
                if time.time() > _DEADLINE:
                    return S
                S2 = S - {rem}
                # Try to add any single number not in S2
                best_add = None
                best_score = forbidden_lens[rem]  # baseline
                for add_idx in range(N):
                    if add_idx in S2:
                        continue
                    if can_add(S2, add_idx):
                        # Prefer safer additions
                        if best_add is None or forbidden_lens[add_idx] < forbidden_lens[best_add]:
                            best_add = add_idx
                
                if best_add is not None:
                    S2.add(best_add)
                    if len(S2) > len(S):
                        S = S2
                        improved = True
                        break
        
        return S
    
    # Local search: 1-for-2 swaps (remove 1, add 2)
    def local_search_1_for_2(S, max_iter=50):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < _DEADLINE:
            improved = False
            iters += 1
            S_list = list(S)
            random.shuffle(S_list)
            for rem in S_list:
                if time.time() > _DEADLINE:
                    return S
                S2 = S - {rem}
                # Try to add two numbers
                found = False
                # Get candidates not in S2, sorted by safety
                cands = [i for i in range(N) if i not in S2]
                cands.sort(key=lambda i: forbidden_lens[i])
                
                # Limit candidates to speed up
                cands = cands[:min(len(cands), 100)]
                
                for i1 in cands:
                    if time.time() > _DEADLINE:
                        return S
                    if not can_add(S2, i1):
                        continue
                    S3 = S2 | {i1}
                    for i2 in cands:
                        if i2 == i1:
                            continue
                        if can_add(S3, i2):
                            S = S3 | {i2}
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        return S
    
    # Multi-start local search
    def multi_start_search(orders, time_budget=70.0):
        best_set = set()
        best_size = 0
        start_time = time.time()
        
        for order in orders:
            if time.time() - start_time > time_budget:
                break
            
            S = greedy_search(order)
            
            # Apply local search
            S = local_search_1_for_1(S, max_iter=30)
            S = local_search_1_for_2(S, max_iter=20)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        
        return best_set
    
    # Run multi-start search
    orders = generate_orders()
    best_set = multi_start_search(orders, time_budget=65.0)
    
    # Final polish: more aggressive local search on the best set
    if time.time() < _DEADLINE - 5:
        best_set = local_search_1_for_1(best_set, max_iter=50)
        best_set = local_search_1_for_2(best_set, max_iter=30)
    
    # Final verification
    S_list = sorted([nums[i] for i in best_set])
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted([nums[i] for i in primes_idx])
        
    return S_list

# EVOLVE-BLOCK-END
