import math
import time

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _parity_vectors(maxM):
    primes = _primes_upto(maxM)
    pindex = {p: i for i, p in enumerate(primes)}
    vecs = []
    for n in range(211, maxM+1):
        x = n
        v = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            c = 0
            while x % p == 0:
                x //= p
                c ^= 1
            if c:
                v |= (1 << i)
        if x > 1:
            j = pindex[x]
            v |= (1 << j)
        vecs.append(v)
    return vecs, primes

def _target_parity(primes):
    t = 210
    v = 0
    for i, p in enumerate(primes):
        c = 0
        while t % p == 0:
            t //= p
            c ^= 1
        if c:
            v |= (1 << i)
    return v

def _span_combo(vectors, target, deadline):
    basis = {}
    for pos, v in enumerate(vectors):
        if time.time() > deadline:
            return None
        vec = v
        combo = 1 << pos
        while vec:
            pivot = vec.bit_length() - 1
            if pivot in basis:
                bvec, bcombo = basis[pivot]
                vec ^= bvec
                combo ^= bcombo
            else:
                basis[pivot] = (vec, combo)
                break
    vec = target
    combo = 0
    while vec:
        pivot = vec.bit_length() - 1
        if pivot in basis:
            bvec, bcombo = basis[pivot]
            vec ^= bvec
            combo ^= bcombo
        else:
            return None
    return combo

def _reduce_size(combo, vectors, target, deadline):
    while combo.bit_count() > 100 and time.time() < deadline:
        sel = [pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
        basis2 = {}
        zero_combo = None
        for pos in sel:
            vec = vectors[pos]
            combo2 = 1 << pos
            while vec:
                pivot = vec.bit_length() - 1
                if pivot in basis2:
                    bv, bc = basis2[pivot]
                    vec ^= bv
                    combo2 ^= bc
                else:
                    basis2[pivot] = (vec, combo2)
                    break
            if vec == 0 and combo2 != 0:
                zero_combo = combo2
                break
        if zero_combo is None or zero_combo == combo:
            break
        combo ^= zero_combo
    return combo

def _find_minimal_window_incremental(vecs, target, deadline):
    """Incrementally build basis and check when target becomes expressible."""
    basis = {}
    n = len(vecs)
    for pos in range(n):
        if time.time() > deadline:
            return None, None
        v = vecs[pos]
        vec = v
        combo = 1 << pos
        while vec:
            pivot = vec.bit_length() - 1
            if pivot in basis:
                bvec, bcombo = basis[pivot]
                vec ^= bvec
                combo ^= bcombo
            else:
                basis[pivot] = (vec, combo)
                break
        # Check if target is now expressible with current basis
        tvec = target
        tcombo = 0
        expressible = True
        while tvec:
            pivot = tvec.bit_length() - 1
            if pivot in basis:
                bvec, bcombo = basis[pivot]
                tvec ^= bvec
                tcombo ^= bcombo
            else:
                expressible = False
                break
        if expressible:
            return pos + 1, tcombo
    return None, None

def construct():
    start = time.time()
    deadline = start + 75  # safety margin under 90s
    maxM = 10210
    vecs, primes = _parity_vectors(maxM)
    target = _target_parity(primes)
    
    # Use incremental approach to find minimal M
    M, combo = _find_minimal_window_incremental(vecs, target, deadline)
    
    if M is None:
        return [840]
    
    # Try to reduce size
    combo = _reduce_size(combo, vecs, target, deadline)
    selected = [211 + pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
    
    # Verify
    prod = 210
    for s in selected:
        prod *= s
    r = math.isqrt(prod)
    if (r * r != prod or len(selected) > 100 or len(set(selected)) != len(selected) 
            or any(s <= 210 or s > 10210 for s in selected) or len(selected) < 1):
        # Try slightly larger M values
        for try_M in range(M + 1, min(M + 500, len(vecs) + 1)):
            if time.time() > deadline:
                break
            sub = vecs[:try_M]
            c2 = _span_combo(sub, target, deadline)
            if c2 is not None:
                c2 = _reduce_size(c2, vecs, target, deadline)
                sel2 = [211 + pos for pos in range(c2.bit_length()) if (c2 >> pos) & 1]
                if len(sel2) <= 100 and len(set(sel2)) == len(sel2) and all(211 <= s <= 10210 for s in sel2):
                    prod2 = 210
                    for s in sel2:
                        prod2 *= s
                    r2 = math.isqrt(prod2)
                    if r2 * r2 == prod2:
                        return list(sel2)
        return [840]
    
    return list(selected)
