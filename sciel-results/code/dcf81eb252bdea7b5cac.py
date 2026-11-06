import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)
MAX_CHECK = 100000

# Precompute masks for each prime and residue?
# That would be 168 primes * avg 500 residues * 100000 bits = too much memory/time to precompute all.
# Instead, we compute the mask on the fly or cache it.
# Mask for prime p and residue r covers indices n such that n % p == r and 1 <= n <= MAX_CHECK.
# Since we only care about the prefix 1..L, we can represent coverage as a bitmask of length MAX_CHECK.
# Python integers are arbitrary precision, so a bitmask of 100k bits is ~12.5 KB.
# Operations on 12.5 KB integers are very fast (C-level).

# We will maintain:
# - current_residues: list of length NUM_PRIMES. 0 means unused. Otherwise r in [0, p-1].
# - current_cost: Fraction sum of 1/p for used primes.
# - coverage_mask: int bitmask where bit i is 1 if number (i+1) is covered? 
#   Let's use bit i to represent number i. So bit 0 is number 0 (unused), bit 1 is number 1.
#   We want to cover 1..L. So we check bits 1..L.
#   Actually, simpler: bit i represents integer i. We want bits 1..L to be 1.

def _make_mask(p, r):
    """
    Create a bitmask where bit n is 1 if n % p == r and 1 <= n <= MAX_CHECK.
    """
    mask = 0
    # Start from the first number >= 1 that satisfies n % p == r
    # If r == 0, the numbers are p, 2p, ...
    # If r > 0, the numbers are r, r+p, ... (if r>=1)
    
    # We need to handle the range 1..MAX_CHECK
    # Find first n >= 1 such that n % p == r
    if r == 0:
        start = p
    else:
        start = r
        if start < 1:
            start += p # r is 0..p-1, so if r>0, start=r is >=1. If r=0 handled above.
            # Actually if r>0, r is in 1..p-1. So start=r is valid.
    
    if start > MAX_CHECK:
        return 0
        
    for n in range(start, MAX_CHECK + 1, p):
        mask |= (1 << n)
        
    return mask

# Precomputing all masks is too slow/memory heavy if done for all p,r.
# But we can cache them. There are only ~168 primes.
# Total residues = sum(p for p in primes) ~ 76k.
# 76k masks * 12.5KB = ~950MB. Too much.
# So we compute on the fly or use a smarter update.

# Alternative: Maintain a list of used (p, r) pairs.
# To calculate L, we can just iterate n=1, 2, ... and check if n is covered.
# But that's slow if L is large.
# Bitmask is good if we can update it fast.
# When we change residue for prime p from r_old to r_new:
# We need to remove coverage from r_old and add coverage from r_new.
# mask_old = _make_mask(p, r_old)
# mask_new = _make_mask(p, r_new)
# coverage_mask ^= mask_old  # Remove (assuming it was set)
# coverage_mask |= mask_new  # Add
# This is O(1) big-int ops.
# The cost is generating mask_old and mask_new.
# Generating a mask takes O(MAX_CHECK/p) time.
# For p=2, that's 50k iterations. For p=1000, 100 iterations.
# This is fast enough for local search if we don't do it too often?
# 50k iterations in Python is ~1-2ms. 
# If we do 1000 moves/sec, that's 1-2 sec per 1000 moves.
# We have 150s. So ~75k-150k moves. This is reasonable.

# To speed up mask generation, we can precompute masks for small primes (where p is small, iterations are many).
# For large primes, computation is fast anyway.
# Let's precompute masks for p <= 100.
_MASK_CACHE = {}
def _get_mask(p, r):
    key = (p, r)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    mask = 0
    if r == 0:
        start = p
    else:
        start = r
    
    if start <= MAX_CHECK:
        for n in range(start, MAX_CHECK + 1, p):
            mask |= (1 << n)
    _MASK_CACHE[key] = mask
    return mask

# Precompute for small primes
for p in _PRIMES:
    if p <= 100:
        for r in range(p):
            _get_mask(p, r)

def _coverage_length_from_mask(mask):
    """
    Find largest L such that bits 1..L are all 1 in mask.
    """
    # We want to find the first 0 bit starting from index 1.
    # If bit 1 is 0, L=0.
    # If bit 1 is 1, bit 2 is 0, L=1.
    
    # Efficient way:
    # Create a mask of all 1s from bit 1 to MAX_CHECK.
    # full_mask = ((1 << (MAX_CHECK + 1)) - 1) ^ 1 # Bit 0 is 0, rest 1.
    # But we only care about the prefix.
    
    # We can iterate. 
    # To speed up, we can check chunks?
    # Or just use a loop. 
    # Since L is likely < 1000, the loop will break early.
    # If L is large (e.g. 10000), the loop takes 10000 steps.
    # 10000 steps is ~1ms.
    # So _coverage_length_from_mask is fast.
    
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if (mask >> n) & 1:
            L = n
        else:
            break
    return L

def _is_valid(assign):
    c = Fraction(0)
    for i, r in enumerate(assign):
        if r != 0: # Note: r=0 is a valid residue! 
            # Wait. In my representation, how do I distinguish "unused" from "residue 0"?
            # I used 0 for unused in the previous code.
            # But residue 0 is valid.
            # I need a better representation.
            # Let's use -1 for unused.
            pass
    return False

# Let's re-evaluate the representation.
# assign[i] = -1 if unused, else r in [0, p-1].
# Cost check: sum(1/p for i where assign[i] != -1) <= 1.

def _calculate_L(assign):
    mask = 0
    for i, r in enumerate(assign):
        if r != -1:
            p = _PRIMES[i]
            mask |= _get_mask(p, r)
    return _coverage_length_from_mask(mask)

def _calculate_cost(assign):
    c = Fraction(0)
    for i, r in enumerate(assign):
        if r != -1:
            c += Fraction(1, _PRIMES[i])
    return c

def construct():
    start = time.time()
    deadline = start + 145
    
    # Initial solution: Greedy
    # Start with empty
    best_assign = [-1] * NUM_PRIMES
    best_L = 0
    best_cost = Fraction(0)
    
    # Greedy construction
    # Sort primes by size? Or just iterate?
    # Let's try to cover 1..1000.
    # We will greedily add primes that cover the most new numbers in 1..2000.
    
    # To make it faster, we'll use a simpler heuristic for the seed:
    # Use primes 2, 3, 5, ... with residues that cover small numbers.
    # For p=2, r=0 covers evens, r=1 covers odds.
    # Let's just run a few random restarts of ILS from a simple seed.
    
    # Simple Seed:
    # Try to cover 1..100 using small primes.
    # p=2, r=0 (covers 2,4,6...)
    # p=3, r=1 (covers 1,4,7...)
    # p=5, r=2 (covers 2,7,12...)
    # This is arbitrary. Let's do a proper greedy seed.
    
    def _greedy_seed():
        assign = [-1] * NUM_PRIMES
        cost = Fraction(0)
        covered = bytearray(2001) # 1..2000
        
        # We want to cover 1..1000.
        # Iteratively pick the (p, r) that maximizes gain in 1..1000.
        
        for _ in range(50): # Max 50 primes
            best_gain = 0
            best_p_idx = -1
            best_r = -1
            
            for i in range(NUM_PRIMES):
                if assign[i] != -1:
                    continue
                if cost + Fraction(1, _PRIMES[i]) > Fraction(1, 1):
                    continue
                p = _PRIMES[i]
                for r in range(p):
                    # Calculate gain
                    gain = 0
                    # Check which numbers in 1..1000 are newly covered
                    # n % p == r
                    start = r
                    if start == 0:
                        start = p
                    if start > 1000:
                        continue
                    for n in range(start, 1001, p):
                        if not covered[n]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p_idx = i
                        best_r = r
            
            if best_p_idx == -1:
                break
                
            p = _PRIMES[best_p_idx]
            r = best_r
            assign[best_p_idx] = r
            cost += Fraction(1, p)
            
            # Update covered
            start = r
            if start == 0:
                start = p
            for n in range(start, 1001, p):
                covered[n] = 1
                
        return assign, cost

    seed_assign, seed_cost = _greedy_seed()
    seed_L = _calculate_L(seed_assign)
    
    best_assign = seed_assign
    best_L = seed_L
    best_cost = seed_cost
    
    # ILS
    current_assign = best_assign[:]
    current_L = best_L
    current_cost = best_cost
    
    # Precompute masks for current state? 
    # We will maintain a global mask for speed.
    current_mask = 0
    for i, r in enumerate(current_assign):
        if r != -1:
            p = _PRIMES[i]
            current_mask |= _get_mask(p, r)
            
    current_L = _coverage_length_from_mask(current_mask)
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if T < 0.1:
            T = 1.0
            # Random restart from best?
            # Or just continue.
            
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Change residue of a used prime
            used = [i for i in range(NUM_PRIMES) if current_assign[i] != -1]
            if not used:
                continue
            i = random.choice(used)
            p = _PRIMES[i]
            old_r = current_assign[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Calculate new L
            # Remove old mask, add new mask
            old_mask = _get_mask(p, old_r)
            new_mask = _get_mask(p, new_r)
            
            # We must be careful: if the prime was covering some numbers, 
            # removing its mask might uncover them if no other prime covers them.
            # But we are just calculating the *new* mask for the whole set.
            # current_mask represents the union of all used primes.
            # To get the mask without prime i, we can't just XOR if other primes cover the same spots.
            # So we need to recalculate the mask for prime i's contribution?
            # No, the mask is a union. 
            # If we change prime i, the new total mask is:
            # (current_mask without i's old contribution) OR (i's new contribution).
            # We don't have "current_mask without i".
            
            # Solution: Recalculate the full mask from scratch?
            # That is O(NUM_PRIMES) big-int ORs. NUM_PRIMES=168.
            # 168 ORs of 12.5KB ints. Very fast (< 1ms).
            # Let's just recalculate.
            
            new_assign = current_assign[:]
            new_assign[i] = new_r
            new_mask = 0
            for j, r in enumerate(new_assign):
                if r != -1:
                    new_mask |= _get_mask(_PRIMES[j], r)
            new_L = _coverage_length_from_mask(new_mask)
            
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_assign = new_assign
                current_mask = new_mask
                current_L = new_L
                if current_L > best_L:
                    best_assign = current_assign[:]
                    best_L = current_L
                    best_cost = current_cost
        else:
            # Add or Remove prime
            if random.random() < 0.5:
                # Add
                unused = [i for i in range(NUM_PRIMES) if current_assign[i] == -1]
                if not unused:
                    continue
                i = random.choice(unused)
                if current_cost + Fraction(1, _PRIMES[i]) > Fraction(1, 1):
                    continue
                r = random.randint(0, _PRIMES[i] - 1)
                
                new_assign = current_assign[:]
                new_assign[i] = r
                new_mask = 0
                for j, rr in enumerate(new_assign):
                    if rr != -1:
                        new_mask |= _get_mask(_PRIMES[j], rr)
                new_L = _coverage_length_from_mask(new_mask)
                new_cost = current_cost + Fraction(1, _PRIMES[i])
                
                delta_L = new_L - current_L
                if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                    current_assign = new_assign
                    current_mask = new_mask
                    current_L = new_L
                    current_cost = new_cost
                    if current_L > best_L:
                        best_assign = current_assign[:]
                        best_L = current_L
                        best_cost = current_cost
            else:
                # Remove
                used = [i for i in range(NUM_PRIMES) if current_assign[i] != -1]
                if not used:
                    continue
                i = random.choice(used)
                p = _PRIMES[i]
                
                new_assign = current_assign[:]
                new_assign[i] = -1
                new_mask = 0
                for j, rr in enumerate(new_assign):
                    if rr != -1:
                        new_mask |= _get_mask(_PRIMES[j], rr)
                new_L = _coverage_length_from_mask(new_mask)
                new_cost = current_cost - Fraction(1, p)
                
                delta_L = new_L - current_L
                if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                    current_assign = new_assign
                    current_mask = new_mask
                    current_L = new_L
                    current_cost = new_cost
                    if current_L > best_L:
                        best_assign = current_assign[:]
                        best_L = current_L
                        best_cost = current_cost
                    
        T *= cooling
        
    # Final validation
    if _calculate_cost(best_assign) <= Fraction(1, 1):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != -1]
    else:
        # Fallback
        return []

