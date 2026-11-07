"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_kernel(n):
    """Return the squarefree kernel (radical of squarefree part) of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def _kernel_to_mask(kernel, primes):
    """Convert a squarefree kernel to a bitmask over the given prime list."""
    mask = 0
    for i, p in enumerate(primes):
        if kernel % p == 0:
            mask |= (1 << i)
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # 210 = 2 * 3 * 5 * 7
    # We need product * 210 to be a perfect square
    # So XOR of squarefree kernels (as bitmasks) of selected numbers must equal kernel of 210
    
    # Primes up to 10210
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Target mask: squarefree kernel of 210
    target_kernel = _squarefree_kernel(n)  # 210 itself (squarefree)
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << prime_to_idx[p])
    
    # Precompute kernel masks for all numbers in (210, 10210]
    num_masks = {}
    for x in range(n + 1, max_val + 1):
        k = _squarefree_kernel(x)
        m = 0
        temp_k = k
        for p in primes:
            if p * p > temp_k:
                break
            if temp_k % p == 0:
                m |= (1 << prime_to_idx[p])
                while temp_k % p == 0:
                    temp_k //= p
        if temp_k > 1:
            m |= (1 << prime_to_idx[temp_k])
        num_masks[x] = m
    
    # We want to find a subset of numbers from (210, 10210] with size 1..100
    # such that XOR of their masks == target_mask
    # Minimize max(selected)
    
    # Strategy: iterate over possible max values M from 211 upward.
    # For each M, check if there's a subset of {211, ..., M} with XOR == target_mask
    # and size <= 100.
    # 
    # For small M, the number of available numbers is small, so we can do BFS on XOR states.
    
    best_result = None
    best_max = max_val + 1
    
    # BFS approach: states are XOR masks. We want to reach target_mask.
    # From state 0, we can add any number from the available pool.
    # We track the set of numbers used to reach each state.
    # To minimize max, we process numbers in increasing order.
    
    # For a given max M, the available numbers are 211..M.
    # We do BFS: start with state 0 (empty set, mask 0).
    # For each number x in 211..M, we can transition from state s to state s^mask[x].
    # We want to reach target_mask with <= 100 numbers.
    
    # Since the number of possible masks is bounded by 2^(number of relevant primes),
    # and we only care about primes that appear in our range, let's limit to small primes.
    
    # Actually, let's use a simpler approach for small M:
    # The XOR of any subset is determined by which primes have odd count.
    # We can model this as: we need to select numbers such that for each prime,
    # the total exponent parity (including 210's contribution) is even.
    
    # Let's try: for M from 211 upward, do a BFS/DPS over XOR states.
    # State: (mask, count) -> we track minimum count to reach each mask.
    # But we also need to reconstruct the set.
    
    # Better: for each M, do BFS tracking mask -> (count, set_of_numbers)
    # But sets can be large. Instead, track mask -> count, and separately track parent pointers.
    
    # Let's limit to masks involving only small primes (up to ~30) for efficiency.
    
    small_primes = [p for p in primes if p <= 31]
    small_prime_set = set(small_primes)
    sp_to_idx = {p: i for i, p in enumerate(small_primes)}
    
    # Recompute masks using only small primes
    num_masks_small = {}
    for x in range(n + 1, max_val + 1):
        k = _squarefree_kernel(x)
        m = 0
        for i, p in enumerate(small_primes):
            if k % p == 0:
                m |= (1 << i)
        # Check if any large prime remains
        temp_k = k
        for p in small_primes:
            while temp_k % p == 0:
                temp_k //= p
        if temp_k > 1:
            # Has a large prime factor not in our small set
            # This makes the mask space larger; handle separately or skip
            m = -1  # mark as "large"
        num_masks_small[x] = m
    
    # Target mask in small prime space
    target_small = 0
    for p in [2, 3, 5, 7]:
        target_small |= (1 << sp_to_idx[p])
    
    # BFS for each M
    # State: mask -> (count, frozenset of numbers) -- but this is expensive
    # Instead: for each M, do a BFS where we track mask -> (min_count, parent_mask, number_used)
    
    for M in range(n + 1, max_val + 1):
        if time.time() > deadline:
            break
        
        # Available numbers: 211..M
        # We'll do BFS from mask=0
        # dist[mask] = (count, prev_mask, num)
        # We want to reach target_small with count <= 100
        
        # Use a dict: mask -> (count, prev_mask, num)
        from collections import deque
        dist = {0: (0, -1, -1)}
        queue = deque([0])
        found = False
        found_mask = None
        
        while queue:
            if time.time() > deadline:
                break
            cur_mask = queue.popleft()
            cur_count, _, _ = dist[cur_mask]
            
            if cur_mask == target_small and cur_count > 0:
                found = True
                found_mask = cur_mask
                break
            
            if cur_count >= max_size:
                continue
            
            # Try adding each number x in 211..M
            # Optimization: only try numbers that haven't been "used" in a way that...
            # Actually in BFS over XOR, we can use each number at most once.
            # But tracking which numbers are used is hard in this formulation.
            
            # Alternative: since we're processing M in increasing order, and for a fixed M
            # the pool is {211..M}, we can do a different approach.
            # 
            # Let's instead use: for a fixed M, the problem is to find a subset of {211..M}
            # with XOR == target. This is a linear algebra problem over GF(2) if we view
            # each number's mask as a vector.
            #
            # The set of achievable XORs is the span of the masks. We need target to be in the span,
            # and we need a representation with at most 100 vectors.
            
            # For small M, let's just do brute force with pruning.
            # But M can be up to 10210, so |pool| can be ~10000.
            
            # Let me reconsider: for a given M, if M is small (say <= 300), the pool has <= 90 elements.
            # We can do a meet-in-the-middle or just BFS on masks.
            
            # For larger M, the pool is large and it's likely that target is easily achievable
            # with a small subset.
            
            pass
        
        # The BFS above is incomplete. Let me use a different strategy.
        
        # Strategy: For a given M, collect all masks of numbers in 211..M.
        # Find a small subset whose XOR is target_small.
        # Use greedy: repeatedly pick a number that reduces the "distance" to target.
        # Or use linear algebra: find a basis and express target in terms of it.
        
        # Let's use linear algebra over GF(2).
        # Each number's mask is a vector. We want a linear combination (XOR) that equals target.
        # The minimal number of vectors in such a combination is what we want, but <= 100 is easy.
        
        # Build a list of (mask, number) for numbers in 211..M
        vectors = []
        for x in range(n + 1, M + 1):
            m = num_masks_small[x]
            if m >= 0:  # only small-prime masks
                vectors.append((m, x))
        
        if not vectors:
            continue
        
        # Gaussian elimination to find a basis and express target
        # Vector length = len(small_primes)
        dim = len(small_primes)
        
        # Build augmented matrix: [vectors | target]
        # Actually, let's do: find if target is in the span, and find a representation.
        
        # Use row reduction on the vectors, tracking which original vectors were used.
        basis = []  # list of (mask, [indices of original vectors used])
        
        for i, (m, x) in enumerate(vectors):
            cur = m
            used = [i]
            for b_mask, b_used in basis:
                if cur == 0:
                    break
                # Find highest set bit
                hb = cur.bit_length() - 1
                bb = b_mask.bit_length() - 1
                if hb == bb:
                    cur ^= b_mask
                    used = [j for j in used if j not in b_used] + [j for j in b_used if j not in used]
                    # Actually this is getting complicated. Let me use a cleaner approach.
            # This is getting messy. Let me use a standard GF(2) basis.
            pass
        
        # Simpler: just check if we can form target using at most 100 vectors.
        # For small M, do a BFS on the mask space.
        
        if M <= 400:
            # Pool size <= 190, do BFS on masks
            # State: mask -> (count, set of numbers)
            # Use a dict: mask -> (count, tuple_of_numbers) -- but this can blow up
            # Instead: mask -> count, and track parent for reconstruction
            dist = {0: (0, -1, -1)}  # mask -> (count, prev_mask, num_idx)
            q = deque([0])
            found = False
            result_set = None
            
            while q:
                if time.time() > deadline:
                    break
                cm = q.popleft()
                cc, pm, ni = dist[cm]
                
                if cm == target_small and cc > 0:
                    found = True
                    # Reconstruct
                    nums = []
                    cur = cm
                    while cur != 0:
                        _, p2, n2 = dist[cur]
                        nums.append(vectors[n2][1])
                        cur = p2
                    result_set = nums
                    break
                
                if cc >= max_size:
                    continue
                
                for j, (m2, x2) in enumerate(vectors):
                    new_m = cm ^ m2
                    if new_m not in dist:
                        dist[new_m] = (cc + 1, cm, j)
                        q.append(new_m)
            
            if found and result_set:
                max_sel = max(result_set)
                if max_sel < best_max:
                    best_max = max_sel
                    best_result = list(result_set)
        
        # For larger M, we can likely find a solution with a small subset.
        # Let's try a greedy/heuristic approach for M > 400.
        elif M <= 1000:
            # Try to find a small subset (size 1-10) whose XOR is target
            # Use random sampling or greedy
            import random
            random.seed(42)
            
            # First try single elements
            for m2, x2 in vectors:
                if m2 == target_small:
                    best_max = M
                    best_result = [x2]
                    break
            
            if best_result is None or best_max > M:
                # Try pairs
                found_pair = False
                for i in range(len(vectors)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(vectors)):
                        if vectors[i][0] ^ vectors[j][0] == target_small:
                            best_max = M
                            best_result = [vectors[i][1], vectors[j][1]]
                            found_pair = True
                            break
                    if found_pair:
                        break
            
            if best_result is None or best_max > M:
                # Try triples
                found_triple = False
                for i in range(min(len(vectors), 200)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, min(len(vectors), 200)):
                        if time.time() > deadline:
                            break
                        for k in range(j + 1, min(len(vectors), 200)):
                            if vectors[i][0] ^ vectors[j][0] ^ vectors[k][0] == target_small:
                                best_max = M
                                best_result = [vectors[i][1], vectors[j][1], vectors[k][1]]
                                found_triple = True
                                break
                        if found_triple:
                            break
                    if found_triple:
                        break
        
        if best_result is not None and best_max <= M:
            # Verify
            prod = n
            for x in best_result:
                prod *= x
            if _is_perfect_square(prod):
                break
    
    # If we found a result, return it
    if best_result is not None:
        # Ensure all are Python ints
        return [int(x) for x in best_result]
    
    # Fallback: return the trivial solution [4*210] = [840]
    # 840 * 210 = 176400 = 420^2. Yes!
    return [840]
# EVOLVE-BLOCK-END
