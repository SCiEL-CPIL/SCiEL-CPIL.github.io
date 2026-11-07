"""Combined program: fast sieve-based SFP precomputation + systematic pair/triple search with deadline guard."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _sieve_sfp(limit):
    """Precompute squarefree part for all numbers 1..limit using a sieve."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    sfp = [0] * (limit + 1)
    sfp[1] = 1
    for x in range(2, limit + 1):
        v = x
        r = 1
        while v > 1:
            p = spf[v]
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt:
                r *= p
        sfp[x] = r
    return sfp

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill
    LO = 210
    MAXV = 10210
    MAX_SIZE = 100

    # Precompute squarefree parts for all numbers up to MAXV
    sfp = _sieve_sfp(MAXV)

    # Target: squarefree_kernel(product) must equal squarefree_kernel(210) = 210
    # i.e., for each selected number x, sfp(x) contributes to the multiplicative XOR.
    # The product's squarefree part is the "multiplicative XOR" (product mod squares) of all sfp values.
    # We need: sfp(210 * product) = 1, i.e., sfp(product) = sfp(210) = 210.
    # Since sfp is multiplicative up to squares: sfp(a*b) = sfp(sfp(a)*sfp(b))
    # So we need the squarefree part of the product of selected sfp values to be 210.
    
    target_sfp = sfp[210]  # should be 210
    
    # Strategy: try increasing M (max allowed value), and for each M try to find
    # a subset of {211, ..., M} with at most MAX_SIZE elements whose sfp-product
    # has squarefree part equal to target_sfp.
    
    # Key insight: if we pick numbers whose sfp values multiply to something with
    # squarefree part 210, we're done.
    
    # For a single number: need sfp(x) = 210. Smallest such x > 210 is 840 (210*4).
    # For two numbers: need sfp(x) * sfp(y) to have squarefree part 210.
    #   i.e., sfp(sfp(x) * sfp(y)) = 210
    #   This means sfp(x) * sfp(y) = 210 * k^2 for some integer k.
    #   Equivalently, sfp(x) XOR sfp(y) (in the squarefree sense) = 210.
    #   Two squarefree numbers a, b: sfp(a*b) = a*b / gcd(a,b)^2 ... 
    #   Actually for squarefree a and b: sfp(a*b) = (a*b) / (gcd(a,b))^2 * ... 
    #   No: if a and b are squarefree, then a*b may not be squarefree.
    #   sfp(a*b) = product of primes p where v_p(a*b) is odd.
    #   v_p(a*b) = v_p(a) + v_p(b), which is odd iff exactly one of v_p(a), v_p(b) is odd.
    #   Since a,b are squarefree, v_p(a) ∈ {0,1}, v_p(b) ∈ {0,1}.
    #   So v_p(a*b) is odd iff p divides exactly one of a, b.
    #   Thus sfp(a*b) = (a * b) / gcd(a,b)^2 ... no.
    #   sfp(a*b) = (a/gcd(a,b)) * (b/gcd(a,b)) = a*b / gcd(a,b)^2
    #   Wait, that's not right either. Let me think again.
    #   If a = p1*p2, b = p2*p3, then a*b = p1*p2^2*p3, sfp = p1*p3.
    #   gcd(a,b) = p2. a*b/gcd^2 = p1*p2^2*p3/p2^2 = p1*p3. Yes!
    #   So sfp(a*b) = a*b / gcd(a,b)^2 when a,b are squarefree.
    #   More generally, for squarefree a,b: sfp(a*b) = (a//g)*(b//g) where g = gcd(a,b).
    
    # So for two numbers: need (sfp(x)//g) * (sfp(y)//g) = 210 where g = gcd(sfp(x), sfp(y)).
    
    # Let's build an index: for each squarefree value s that appears as sfp(x) for some x in range,
    # store the smallest x with that sfp.
    
    # Actually, let's just do a direct search.
    
    # Precompute sfp for all candidates
    # sfp is already computed for all up to MAXV.
    
    # Try single element
    for x in range(LO + 1, MAXV + 1):
        if sfp[x] == target_sfp:
            # Verify
            prod = LO * x
            if _is_square(prod):
                return [x]
    
    # Try pairs: for each x, we need y such that sfp(sfp[x] * sfp[y]) == target_sfp
    # i.e., (sfp[x] * sfp[y]) // gcd(sfp[x], sfp[y])^2 == target_sfp
    # Let a = sfp[x], b = sfp[y], g = gcd(a,b)
    # Need: a*b // g^2 == target_sfp
    # i.e., a*b == target_sfp * g^2
    # Since g divides both a and b, let a = g*a', b = g*b' with gcd(a',b')=1
    # Then g^2 * a' * b' = target_sfp * g^2 => a' * b' = target_sfp
    # So we need: gcd(a', b') = 1 and a' * b' = target_sfp, where a' = a/g, b' = b/g.
    # Equivalently: b = target_sfp * a / gcd(a, target_sfp * a / gcd(a,b))... this is circular.
    
    # Simpler: for a given a = sfp[x], find b = sfp[y] such that sfp(a*b) = target_sfp.
    # sfp(a*b) = a*b // gcd(a,b)^2 (for squarefree a,b)
    # Let g = gcd(a,b). Then a*b/g^2 = target_sfp.
    # b = target_sfp * g^2 / a. Since b must be an integer and squarefree, and g | a, g | b.
    # Let a = g * a1 where gcd(a1, g) may not be 1... actually g = gcd(a,b) so gcd(a/g, b/g) = 1.
    # So b = target_sfp * g^2 / a = target_sfp * g / a1 where a1 = a/g.
    # For b to be integer: a1 must divide target_sfp * g.
    # This is getting complex. Let's just use a lookup table.
    
    # Build mapping: sfp_value -> list of numbers having that sfp
    from collections import defaultdict
    sfp_to_nums = defaultdict(list)
    for x in range(LO + 1, MAXV + 1):
        sfp_to_nums[sfp[x]].append(x)
    
    # For pairs: iterate over distinct sfp values
    sfp_values = sorted(sfp_to_nums.keys())
    found = None
    
    for i in range(len(sfp_values)):
        if time.time() > deadline - 5:
            break
        a = sfp_values[i]
        nums_a = sfp_to_nums[a]
        for j in range(i, len(sfp_values)):
            b = sfp_values[j]
            nums_b = sfp_to_nums[b]
            # Check if sfp(a*b) == target_sfp
            # Compute sfp(a*b) for squarefree a, b
            g = 1
            # Compute gcd
            aa, bb = a, b
            while bb:
                aa, bb = bb, aa % bb
            g = aa
            # sfp(a*b) = a*b // g^2
            prod_sfp = (a * b) // (g * g)
            if prod_sfp == target_sfp:
                # Find smallest x in nums_a and smallest y in nums_b (x != y if a==b)
                if a == b:
                    if len(nums_a) >= 2:
                        x = nums_a[0]
                        y = nums_a[1]
                    else:
                        continue
                else:
                    x = nums_a[0]
                    y = nums_b[0]
                # Verify
                prod = LO * x * y
                if _is_square(prod):
                    mx = max(x, y)
                    if found is None or mx < found[0]:
                        found = (mx, [min(x,y), max(x,y)])
    
    if found is not None:
        return found[1]
    
    # Try triples
    # For three numbers with sfp values a, b, c: sfp(a*b*c) = target_sfp
    # This is more complex. Let's try a different approach.
    
    # Alternative: use the parity vector / linear algebra approach.
    # Represent each sfp value as a vector over GF(2) where each prime is a dimension.
    # The XOR (symmetric difference) of the prime sets must equal the prime set of target_sfp.
    
    # Collect all primes up to MAXV
    primes = []
    is_prime = [True] * (MAXV + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, MAXV + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, MAXV + 1, i):
                is_prime[j] = False
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Target vector: primes in target_sfp
    target_vec = [0] * num_primes
    tv = target_sfp
    for p in primes:
        if tv % p == 0:
            target_vec[prime_to_idx[p]] = 1
            while tv % p == 0:
                tv //= p
    
    # For each number, its parity vector
    def sfp_to_vec(s):
        vec = [0] * num_primes
        v = s
        for p in primes:
            if p * p > v:
                break
            if v % p == 0:
                vec[prime_to_idx[p]] = 1
                v //= p
                while v % p == 0:
                    v //= p
        if v > 1:
            idx = prime_to_idx.get(v)
            if idx is not None:
                vec[idx] = 1
        return vec
    
    # Precompute vectors for all sfp values
    sfp_vecs = {}
    for sv in sfp_values:
        sfp_vecs[sv] = sfp_to_vec(sv)
    
    # For triples: try to find a, b, c such that vec_a XOR vec_b XOR vec_c = target_vec
    # i.e., vec_c = target_vec XOR vec_a XOR vec_b
    
    # Build lookup: vector (as tuple) -> list of sfp values with that vector
    # Actually, each sfp value has a unique vector (since sfp is squarefree, the vector determines it)
    # So vec -> sfp is a bijection for valid sfp values.
    
    vec_to_sfp = {}
    for sv in sfp_values:
        v = sfp_vecs[sv]
        vec_to_sfp[tuple(v)] = sv
    
    # For triples: iterate over pairs of sfp values, compute needed third vector, look up
    triple_found = None
    for i in range(len(sfp_values)):
        if time.time() > deadline - 5:
            break
        a = sfp_values[i]
        va = sfp_vecs[a]
        for j in range(i, len(sfp_values)):
            b = sfp_values[j]
            vb = sfp_vecs[b]
            # Need vc = target_vec XOR va XOR vb
            vc = [target_vec[k] ^ va[k] ^ vb[k] for k in range(num_primes)]
            vt = tuple(vc)
            c = vec_to_sfp.get(vt)
            if c is None:
                continue
            # Now we have sfp values a, b, c. Find actual numbers.
            # Need distinct numbers x, y, z with sfp[x]=a, sfp[y]=b, sfp[z]=c
            # and all distinct.
            if a == b == c:
                if len(sfp_to_nums[a]) >= 3:
                    x, y, z = sfp_to_nums[a][0], sfp_to_nums[a][1], sfp_to_nums[a][2]
                else:
                    continue
            elif a == b:
                if len(sfp_to_nums[a]) >= 2 and len(sfp_to_nums[c]) >= 1:
                    x, y = sfp_to_nums[a][0], sfp_to_nums[a][1]
                    z = sfp_to_nums[c][0]
                else:
                    continue
            elif a == c:
                if len(sfp_to_nums[a]) >= 2 and len(sfp_to_nums[b]) >= 1:
                    x, z = sfp_to_nums[a][0], sfp_to_nums[a][1]
                    y = sfp_to_nums[b][0]
                else:
                    continue
            elif b == c:
                if len(sfp_to_nums[b]) >= 2 and len(sfp_to_nums[a]) >= 1:
                    y, z = sfp_to_nums[b][0], sfp_to_nums[b][1]
                    x = sfp_to_nums[a][0]
                else:
                    continue
            else:
                x = sfp_to_nums[a][0]
                y = sfp_to_nums[b][0]
                z = sfp_to_nums[c][0]
            
            # Verify
            prod = LO * x * y * z
            if _is_square(prod):
                mx = max(x, y, z)
                if triple_found is None or mx < triple_found[0]:
                    triple_found = (mx, sorted([x, y, z]))
    
    if triple_found is not None:
        return triple_found[1]
    
    # Fallback: try larger sets using a greedy approach
    # Or just return the known valid [840]
    return [840]
# EVOLVE-BLOCK-END
