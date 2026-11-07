import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Primes to use for smooth numbers
    # Increasing this list increases the candidate pool
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We want k * sigma(k) <= 10^18
    # sigma(k) >= k + 1, so k^2 < 10^18 => k < 10^9
    # We can go higher if sigma(k)/k is small, but for smooth numbers sigma(k)/k is large.
    # Let's set a safe limit for k.
    MAX_K = 10**9 

    candidates = [] # List of (M, k)

    # Recursive generation of smooth numbers
    # We track the current value, the index of the last prime used (to avoid duplicates),
    # and the current sigma value.
    # sigma(k) = product (p^(a+1)-1)/(p-1)
    
    def generate_smooth(idx, current_val, current_sigma):
        if current_val > MAX_K:
            return
        
        # If current_val is 1, we haven't added any primes yet.
        # We should record valid M > 0.
        # k=1 => sigma(1)=1 => M=1.
        # The problem asks for distinct k_i. M=1 has only k=1.
        
        # Check if current_val * current_sigma <= 10^18
        # Note: current_sigma is sigma(current_val)
        
        # We add this candidate if current_val >= 1
        # Actually, we start with val=1, sigma=1.
        # 1*1=1.
        
        # To avoid recording 1 repeatedly or handling base case separately,
        # we can just record every valid node in the recursion tree.
        
        if current_val * current_sigma <= 10**18:
            candidates.append((current_val * current_sigma, current_val))
            
        # Try multiplying by primes[idx]
        if idx >= len(primes):
            return
        
        p = primes[idx]
        # Multiply current_val by p, p^2, ...
        # new_val = current_val * p^e
        # new_sigma = current_sigma * (1 + p + ... + p^e)
        
        val = current_val
        sig = current_sigma
        p_power = 1
        sigma_term = 1
        
        # Limit exponent such that val * p_power <= MAX_K
        while val * p_power * p <= MAX_K:
            p_power *= p
            val *= p
            # Update sigma term: (1 + p + ... + p^e)
            # Previous term was for p^(e-1). New term adds p^e.
            sigma_term += p_power
            
            # New sigma is current_sigma * sigma_term
            new_sigma = current_sigma * sigma_term
            
            # Recurse with next prime index to avoid permutations
            # The value is val, sigma is new_sigma
            generate_smooth(idx + 1, val, new_sigma)
            
            # Check time periodically? 
            # The recursion tree size is the number of smooth numbers.
            # For MAX_K=10^9 and primes up to 97, this might be large but likely < 1M nodes.
            # 1M recursive calls in Python might take ~1-2 seconds.
            # Let's add a time check in the outer loop or via a counter.
            
        # Note: We don't recurse for the "current" state with the SAME prime again 
        # because we handle powers in the while loop. 
        # We recurse to idx+1 to allow OTHER primes.
        # Wait, the logic above:
        # generate_smooth(idx, val, sig)
        #   adds (val*sig, val)
        #   for p in primes[idx:]:
        #      ...
        
    # The above recursive function has a flaw: it doesn't iterate over primes[idx:] in the loop.
    # It only processes primes[idx].
    # Correct structure:
    
    def gen_smooth(idx, current_val, current_sigma):
        if current_val > MAX_K:
            return
            
        if current_val * current_sigma <= 10**18:
            candidates.append((current_val * current_sigma, current_val))
            
        if idx == len(primes):
            return
            
        p = primes[idx]
        
        # Option 1: Do not use prime p (move to next prime)
        # This is handled by the caller? No, standard smooth number generation:
        # f(idx, val, sig) generates numbers using primes[idx:].
        # 1. Include val (using no more primes from idx) -> Record it.
        # 2. Multiply val by p, p^2, ... and for each, recurse with idx+1.
        
        # Actually, simpler:
        # For each prime p in primes[idx:]:
        #   We can multiply current_val by p^e.
        
        # Let's use the standard "k-way merge" or recursive backtracking.
        
        # Current implementation attempt:
        # We are at prime index idx.
        # We can either:
        # 1. Stop using primes from idx onwards (record current_val). (Done above)
        # 2. Multiply by p^e (e>=1) and then recurse for primes idx+1 onwards.
        
        # This covers all combinations.
        
        val = current_val
        p_power = 1
        sigma_term = 1 # This will be sum_{i=1..e} p^i? No.
        # sigma(p^e) = 1 + p + ... + p^e
        # If we multiply current_val (which has sigma S) by p^e,
        # new_val = current_val * p^e
        # new_sigma = S * (1 + p + ... + p^e)
        
        # Let's iterate e from 1 upwards.
        
        e = 0
        while True:
            next_val = current_val * (p ** (e+1)) # This is inefficient power calc
            # Better:
            pass

    # Let's rewrite the generator cleanly.
    
    def generate(idx, val, sig):
        """
        Generates smooth numbers using primes[idx:] multiplying 'val'.
        'sig' is sigma(val).
        """
        if val > MAX_K:
            return
        
        # Record this candidate
        # val is composed of primes[0:idx] (implicitly, or rather, it's the current base)
        # Actually, this function is called with the current state.
        # If we don't multiply by any more primes, 'val' is a valid smooth number.
        if val * sig <= 10**18:
            candidates.append((val * sig, val))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Multiply val by p, p^2, p^3...
        # new_val = val * p^e
        # new_sig = sig * (1 + p + ... + p^e)
        
        current_pow = 1
        current_sum = 1 # 1 + ... + p^e starts with 1? No, sigma(1)=1.
        # If e=0, we don't multiply. That's the 'Record' step above.
        # We want e >= 1.
        
        # Start with e=1
        val_e = val * p
        sum_e = 1 + p # sigma(p)
        
        while val_e <= MAX_K:
            # Recurse with next prime index
            generate(idx + 1, val_e, sig * sum_e)
            
            # Prepare for next exponent
            # Check if we should break based on time?
            # The number of calls is proportional to the number of smooth numbers.
            
            val_e *= p
            sum_e += val_e # Wait, val_e is val*p^(e+1). 
            # Let's trace:
            # e=1: val_e = val*p. sum_e = 1+p.
            # Call gen(idx+1, val*p, sig*(1+p))
            # Next:
            # val_e = val*p*p = val*p^2.
            # sum_e should be 1 + p + p^2.
            # Previous sum_e was 1+p. Add p^2.
            # val_e is now val*p^2. So we add val_e? No, we add p^2.
            # p^2 = (val_e / val).
            # It's safer to track p_power separately.
            
            p_power *= p
            sum_e += p_power
            
            val_e = val * p_power
            
            if val_e > MAX_K:
                break

    # Start generation
    # Initial state: val=1, sig=1, idx=0
    # Note: The 'Record' step in generate() will record 1.
    # Then it will try to multiply by 2, 4, 8...
    
    # To prevent stack overflow or excessive time, we can add a counter.
    # But 10^9 limit with 25 primes should be fine.
    
    sys.setrecursionlimit(10000)
    
    # We need to be careful with the time.
    # Let's run it.
    
    # Optimization: If we find a group with > 1000 items, we can stop early?
    # No, we need the MAX. But we can't know the max until we finish.
    # However, if we exceed the time limit, we should return best so far.
    
    # Let's wrap the generation in a try/except or time check.
    # Since it's recursive, checking time in every call is expensive.
    # We can check every 10000 calls.
    
    call_count = 0
    def generate_timed(idx, val, sig):
        nonlocal call_count
        call_count += 1
        if call_count % 10000 == 0 and time.time() > deadline - 5:
            raise TimeoutError("Generation timeout")
            
        if val > MAX_K:
            return
        
        if val * sig <= 10**18:
            candidates.append((val * sig, val))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        p_power = 1
        sum_e = 1 # sigma(1)
        
        # We want e >= 1
        # val_e = val * p
        # sum_e = 1 + p
        
        while True:
            p_power *= p
            val_e = val * p_power
            if val_e > MAX_K:
                break
            sum_e += p_power
            
            generate_timed(idx + 1, val_e, sig * sum_e)
            
    try:
        generate_timed(0, 1, 1)
    except TimeoutError:
        print(f"Timeout during generation. Processed {len(candidates)} candidates.")
    except RecursionError:
        print("Recursion limit reached.")
    except Exception as e:
        print(f"Error: {e}")
        
    if not candidates:
        # Fallback to seed
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Group by M
    groups = defaultdict(list)
    for M, k in candidates:
        groups[M].append(k)
        
    # Find max group
    best_M = 0
    best_ks = []
    max_count = 0
    
    for M, ks in groups.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_M = M
            best_ks = ks
            
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    # If we didn't find anything better than seed, return seed?
    # The seed has 3. If we find > 3, we return that.
    # If we find <= 3, we should probably return the seed to be safe, 
    # unless our found M is valid.
    # The task says "Maximize". If we find 2, and seed has 3, seed is better.
    
    if max_count <= 3:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_ks}
