import time
import random
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    # Evaluator kills after 90s. Contract says 60s.
    # We will use 55s to be safe.
    deadline = start_time + 55.0
    
    # Precompute signatures for numbers 2..128
    # Primes <= 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    P = len(primes)
    
    def get_sig(x):
        sig = [0] * P
        n = x
        for idx, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                sig[idx] += 1
                n //= p
        if n > 1:
            idx = prime_to_idx[n]
            sig[idx] += 1
        return tuple(sig)
    
    sigs = [None] * 129
    for i in range(2, 129):
        sigs[i] = get_sig(i)
        
    def add_sig(a, b):
        return tuple(x + y for x, y in zip(a, b))

    # Global best
    best_seq = []
    best_len = 0
    
    # Fast Greedy to get an initial lower bound
    def run_greedy(start_val, end_val, skip_set=None):
        if skip_set is None:
            skip_set = set()
        seq = []
        products_set = set()
        suffix_sigs = []
        
        for x in range(start_val, end_val + 1):
            if x in skip_set:
                continue
            x_sig = sigs[x]
            new_tails = [x_sig]
            current_prod = x_sig
            for s in reversed(suffix_sigs):
                current_prod = add_sig(current_prod, s)
                new_tails.append(current_prod)
                
            # Check collision
            valid = True
            for tail in new_tails:
                if tail in products_set:
                    valid = False
                    break
            
            if valid:
                seq.append(x)
                products_set.update(new_tails)
                suffix_sigs.append(x_sig)
                
        return seq

    # Initial greedy
    init_seq = run_greedy(2, 128)
    best_seq = init_seq
    best_len = len(init_seq)
    
    # Branch and Bound
    # State: (idx, current_seq, products_set, suffix_sigs)
    # To speed up, we pass lists and sets. Python recursion overhead is high, but depth is max 127.
    # We use a heuristic: try "Include" first if it looks promising, else "Exclude".
    # Actually, standard B&B:
    # 1. Upper bound check.
    # 2. Try Include.
    # 3. Try Exclude.
    
    # Optimization: Pre-compute "max possible additions" from any index?
    # No, it depends on the current state.
    
    # We will use a recursive function.
    # To avoid deep recursion issues or overhead, we can limit the depth of backtracking
    # or use an iterative approach. Given N=127, recursion is fine if we prune well.
    
    # Heuristic for ordering:
    # If we include a number, it might block others.
    # Let's try to find a sequence that is "dense".
    
    def bbs(idx, current_seq, products_set, suffix_sigs):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
            
        # Upper Bound:
        # Max possible length is len(current_seq) + (128 - idx + 1)
        # But we can do better with a greedy estimate from this state.
        # Greedy estimate:
        # We simulate adding the rest greedily from the current state.
        # This is expensive if done at every node.
        # Instead, simple bound: len(current_seq) + (128 - idx + 1) <= best_len?
        if len(current_seq) + (128 - idx + 1) <= best_len:
            return
            
        # Base case
        if idx > 128:
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = current_seq[:]
            return
            
        x = idx
        x_sig = sigs[x]
        
        # Check if x can be added
        new_tails = [x_sig]
        current_prod = x_sig
        for s in reversed(suffix_sigs):
            current_prod = add_sig(current_prod, s)
            new_tails.append(current_prod)
            
        valid = True
        for tail in new_tails:
            if tail in products_set:
                valid = False
                break
        
        if valid:
            # Try Include
            # Copy state (expensive but necessary for correctness in B&B without undo)
            # Optimization: We can modify in place and undo, but sets and lists of tuples are tricky.
            # Let's use a "persistent" style or just copy.
            # Depth is ~127. Copying a set of ~2000 items * 127 times is heavy.
            # Alternative: Use a stack of states?
            
            # Let's try a "Include" branch first.
            new_products_set = products_set.copy()
            new_products_set.update(new_tails)
            new_suffix_sigs = suffix_sigs + [x_sig]
            new_seq = current_seq + [x]
            
            bbs(idx + 1, new_seq, new_products_set, new_suffix_sigs)
            
            # Time check
            if time.time() > deadline:
                return
            
        # Try Exclude
        # Only if it's worth it.
        # If len(current_seq) + (128 - idx) <= best_len: return (already covered by upper bound? No, upper bound used +1)
        # Upper bound for exclude branch: len(current_seq) + (128 - idx)
        if len(current_seq) + (128 - idx) > best_len:
            bbs(idx + 1, current_seq, products_set, suffix_sigs)

    # Run B&B starting from empty set
    # To make it faster, we can start from the greedy solution and try to "improve" it?
    # No, B&B from scratch is safer for correctness.
    # However, B&B from scratch might be too slow if the space is large.
    # We can seed the B&B with the greedy solution's prefix?
    # Let's try running B&B on the "decisions" of the greedy solution first, then explore alternatives.
    
    # Actually, a better strategy:
    # 1. Get greedy solution G.
    # 2. Use B&B to verify if we can do better.
    # 3. If B&B is too slow, we fall back to random restarts.
    
    # Let's run B&B with a time limit.
    bbs(2, [], set(), [])
    
    # If B&B didn't improve much, try random restarts with the greedy
    if time.time() < deadline - 5.0:
        # Random restarts
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            
            # Randomly skip some small numbers to perturb the greedy path
            skip_set = set()
            for x in range(2, 20):
                if random.random() < 0.3:
                    skip_set.add(x)
            
            seq = run_greedy(2, 128, skip_set)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq
                
    return best_seq
