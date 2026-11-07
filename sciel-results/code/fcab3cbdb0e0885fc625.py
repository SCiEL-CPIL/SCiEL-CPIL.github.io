# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _compute_sigs(limit):
    """Compute squarefree signatures for all numbers 1..limit using sieve."""
    sigs = [frozenset() for _ in range(limit + 1)]
    # For each prime p, toggle p in the signature of all multiples of p^odd
    # More efficiently: for each number, factor it
    # Sieve approach: for each prime p, for each multiple m of p, 
    # count exponent of p in m, if odd, add p to sig[m]
    
    # First find primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    # For each prime, determine which numbers have odd exponent
    # For prime p, a number n has odd exponent of p if p^odd divides n but p^(odd+1) doesn't
    # i.e., v_p(n) is odd
    for p in primes:
        p2 = p * p
        # Numbers with v_p(n) >= 1: multiples of p
        # Numbers with v_p(n) >= 2: multiples of p^2
        # v_p(n) is odd iff n is divisible by p but not p^3, or divisible by p^3 but not p^5, etc.
        # More simply: v_p(n) is odd iff (n // p^v) where v = v_p(n) has v odd
        # Alternative: for each multiple m of p, compute v_p(m) and check if odd
        # This is O(limit * num_primes) which might be too slow for limit=10210
        
        # Better: iterate through multiples
        # For p, the numbers with odd exponent of p are:
        # p, 2p, 3p, ... but excluding those with even exponent
        # Actually, let's just factor each number individually for clarity
        pass
    
    # For limit ~10210, factoring each number is fine
    sigs = [frozenset() for _ in range(limit + 1)]
    for n in range(2, limit + 1):
        m = n
        sig = set()
        d = 2
        while d * d <= m:
            cnt = 0
            while m % d == 0:
                m //= d
                cnt ^= 1
            if cnt:
                sig.add(d)
            d += 1 if d == 2 else 2  # skip even numbers after 2
        if m > 1:
            sig.add(m)
        sigs[n] = frozenset(sig)
    
    return sigs

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    MAX_VAL = 10210
    N = 210
    sig_210 = frozenset({2, 3, 5, 7})
    
    # Precompute signatures
    sigs = _compute_sigs(MAX_VAL)
    
    # Build lookup: signature -> list of numbers with that signature
    sig_to_nums = defaultdict(list)
    for i in range(211, MAX_VAL + 1):
        sig_to_nums[sigs[i]].append(i)
    
    best_result = None
    best_max = MAX_VAL + 1
    
    # Search for solutions with max = M, increasing M
    # For each M, check subsets of {211..M} with max element M (or not, but we want min max)
    # Actually, we just need ANY subset of {211..M} whose product*210 is square
    # and we want to minimize M
    
    # Strategy: for each M from 211 to MAX_VAL, check if there's a valid subset
    # with all elements <= M. Once we find one, that's our answer (since we're going in order).
    
    # For efficiency, check small subsets first (1, 2, 3, 4 elements)
    
    for M in range(211, min(1000, MAX_VAL + 1)):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        # Check 1-element: sig(M) == sig_210
        if sigs[M] == sig_210:
            best_max = M
            best_result = [M]
            break
        
        # Check 2-element: need a in (210, M) with sigs[a] XOR sigs[M] == sig_210
        target = sigs[M] ^ sig_210
        if target in sig_to_nums:
            for a in sig_to_nums[target]:
                if 210 < a < M:
                    best_max = M
                    best_result = [a, M]
                    break
        if best_result is not None and best_max == M:
            break
        
        # Check 3-element: need a, b in (210, M), a < b, with sigs[a] XOR sigs[b] XOR sigs[M] == sig_210
        # i.e., sigs[a] XOR sigs[b] == sigs[M] XOR sig_210
        target3 = sigs[M] ^ sig_210
        seen = {}
        found = False
        for a in range(211, M):
            if time.time() > deadline:
                break
            need = sigs[a] ^ target3
            if need in seen:
                b = seen[need]
                if b != a:
                    best_max = M
                    best_result = [a, b, M]
                    found = True
                    break
            else:
                seen[sigs[a]] = a
        if found:
            break
        
        # Check 4-element: need a, b, c in (210, M) with sigs[a]^sigs[b]^sigs[c]^sigs[M] == sig_210
        # i.e., sigs[a]^sigs[b]^sigs[c] == sigs[M]^sig_210
        # This is harder. For now, skip and rely on 1-3 element searches.
        # If M gets large enough, 3-element should find something.
    
    # If we didn't find anything up to 1000, extend the search
    if best_result is None:
        for M in range(1001, min(3000, MAX_VAL + 1)):
            if time.time() > deadline:
                break
            if M >= best_max:
                break
            
            # Check 2-element
            target = sigs[M] ^ sig_210
            if target in sig_to_nums:
                for a in sig_to_nums[target]:
                    if 210 < a < M:
                        best_max = M
                        best_result = [a, M]
                        break
            if best_result is not None and best_max == M:
                break
            
            # Check 3-element
            target3 = sigs[M] ^ sig_210
            seen = {}
            found = False
            for a in range(211, M):
                if time.time() > deadline:
                    break
                need = sigs[a] ^ target3
                if need in seen:
                    b = seen[need]
                    if b != a:
                        best_max = M
                        best_result = [a, b, M]
                        found = True
                        break
                else:
                    seen[sigs[a]] = a
            if found:
                break
    
    # If still nothing, try a more exhaustive approach for small M
    if best_result is None:
        # Try all pairs (a, b) with a < b, check if a*b*210 is square
        # Or a*b*c*210 is square
        # For pairs: a*b*210 = square
        for a in range(211, 500):
            if time.time() > deadline:
                break
            for b in range(a + 1, 500):
                if _is_perfect_square(a * b * 210):
                    best_max = max(a, b)
                    best_result = [a, b]
                    break
            if best_result is not None:
                break
    
    # Fallback: use known solution
    if best_result is None:
        # 840 = 210 * 4, so 840 * 210 = 210^2 * 4 = (210*2)^2 = 420^2
        best_result = [840]
        best_max = 840
    
    # Verify and return
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
