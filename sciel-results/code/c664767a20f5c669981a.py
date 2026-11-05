"""
Structural approach: formulate as maximum independent set on a *precise* conflict graph.

Key insight: two values a < b conflict iff they cannot coexist in ANY valid
strictly-increasing sequence within [2,128]. We can compute this precisely:
- a and b are "independent" if there exists a valid sequence containing both.
- The simplest check: {a, b} itself is valid (two-element sequence) — trivially
  no collisions since only products are a, b, ab and all distinct as long as a≠b.
  
So the true conflict graph is empty? No — the issue is that including both may
force collisions with OTHER elements. But for a *pairwise* conflict, we only need:
  a and b cannot both appear in ANY valid sequence.

Since {a, b} is always valid (products: a, b, ab — all distinct for a≠b, a,b≥2),
there is NO pairwise conflict. The conflicts are higher-order.

Therefore, the right model is NOT a pairwise conflict graph. Instead, we use:

**Direct greedy with optimal ordering + local search with product-collision tracking.**

The greedy algorithm: iterate over candidates in a smart order; add x if it
doesn't create a duplicate consecutive-block product with the current sequence.

The critical optimization: precompute prime-exponent vectors for each number.
A block product's identity is determined by its prime-exponent vector. Two blocks
collide iff their exponent vectors are identical.

We maintain the set of all block-product exponent vectors. When considering adding x,
we check if any new block product (formed by extending existing suffixes with x)
already exists.

For local search: try inserting unused elements at every position, and try
removing/replacing elements to free up product "slots".
"""
import time
from collections import Counter

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin

    # Precompute prime exponent vectors for 2..128
    # Primes up to 128: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]
    prime_idx = {p: i for i, p in enumerate(primes)}
    NP = len(primes)

    def prime_vec(n):
        v = [0] * NP
        for p, idx in prime_idx.items():
            while n % p == 0:
                v[idx] += 1
                n //= p
        return tuple(v)

    # Precompute for all candidates
    cand_range = range(2, 129)
    vecs = {x: prime_vec(x) for x in cand_range}

    def add_vec(a, b):
        return tuple(a[i] + b[i] for i in range(NP))

    def check_valid(seq):
        """Check if seq is strictly increasing and all block products distinct."""
        n = len(seq)
        if n == 0:
            return True
        for i in range(n-1):
            if seq[i] >= seq[i+1]:
                return False
        products = set()
        for i in range(n):
            pv = [0]*NP
            for j in range(i, n):
                for k in range(NP):
                    pv[k] += vecs[seq[j]][k]
                pv_t = tuple(pv)
                if pv_t in products:
                    return False
                products.add(pv_t)
        return True

    def greedy(order):
        """Greedy construction in given order."""
        seq = []
        # Maintain all block product vectors
        # block_products[i] = set of product vectors for blocks ending at position i
        # Actually, we need ALL block products. Let's maintain a global set.
        # When adding x to the end, new blocks are: [x], [seq[-1], x], [seq[-2], seq[-1], x], ...
        all_prods = set()
        suffix_prods = []  # product vectors of suffixes ending at last element
        for x in order:
            if x < 2 or x > 128:
                continue
            # Check if x is already in seq
            if x in seq:
                continue
            # Compute new block product vectors
            new_prods = [vecs[x]]
            cur = vecs[x]
            # Extend backwards through existing seq
            for a in reversed(seq):
                cur = add_vec(cur, vecs[a])
                new_prods.append(cur)
            # Check for conflicts
            conflict = False
            for p in new_prods:
                if p in all_prods:
                    conflict = True
                    break
            if not conflict:
                seq.append(x)
                all_prods.update(new_prods)
                # Update suffix products for next iteration
                # suffix_prods should be product vectors of suffixes of seq ending at last element
                new_suffix = [vecs[x]]
                cur = vecs[x]
                for a in reversed(seq[:-1]):
                    cur = add_vec(cur, vecs[a])
                    new_suffix.append(cur)
                suffix_prods = new_suffix
        return seq, all_prods

    # Try multiple orderings
    candidates = list(range(2, 129))
    
    # Ordering 1: natural
    orders = [candidates]
    
    # Ordering 2: by number of distinct prime factors (ascending — "simple" numbers first)
    def num_distinct_pf(n):
        cnt = 0
        for p in primes:
            if p > n: break
            if n % p == 0:
                cnt += 1
        return cnt
    orders.append(sorted(candidates, key=num_distinct_pf))
    orders.append(sorted(candidates, key=lambda x: -num_distinct_pf(x)))
    
    # Ordering 3: by total prime factors (with multiplicity)
    def total_pf(n):
        cnt = 0
        for p in primes:
            while n % p == 0:
                cnt += 1
                n //= p
            if n == 1: break
        return cnt
    orders.append(sorted(candidates, key=total_pf))
    orders.append(sorted(candidates, key=lambda x: -total_pf(x)))
    
    # Ordering 4: primes first, then composites
    def is_prime(n):
        for p in primes:
            if p*p > n: break
            if n % p == 0:
                return p == n
        return n > 1
    orders.append(sorted(candidates, key=lambda x: (0 if is_prime(x) else 1, x)))
    
    # Ordering 5: reverse natural
    orders.append(list(reversed(candidates)))
    
    # Ordering 6: by value mod small primes (spread out)
    for mod in [5, 7, 11]:
        orders.append(sorted(candidates, key=lambda x: x % mod))
    
    best_seq = []
    best_len = 0
    best_all_prods = set()
    
    for order in orders:
        if time.time() - t0 > DEADLINE:
            break
        seq, all_prods = greedy(order)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            best_all_prods = all_prods
    
    # Local search: try to insert unused elements
    used = set(best_seq)
    remaining = [x for x in candidates if x not in used]
    
    improved = True
    while improved and time.time() - t0 < DEADLINE:
        improved = False
        # Try inserting each remaining element at each position
        for new_x in remaining[:]:
            if time.time() - t0 > DEADLINE:
                break
            # Rebuild: try inserting new_x at each valid position
            seq = list(best_seq)
            for pos in range(len(seq) + 1):
                if pos > 0 and seq[pos-1] >= new_x:
                    continue
                if pos < len(seq) and seq[pos] <= new_x:
                    continue
                # Check product conflicts
                # New block products involving new_x:
                # Blocks that include new_x: [new_x], [.., new_x], [new_x, ..], [.., new_x, ..]
                # This is complex. Let's just rebuild and check.
                candidate_seq = seq[:pos] + [new_x] + seq[pos:]
                if check_valid(candidate_seq):
                    best_seq = candidate_seq
                    used.add(new_x)
                    remaining.remove(new_x)
                    improved = True
                    break
    
    # Final validation
    if not check_valid(best_seq):
        # Fallback: simple greedy
        best_seq, _ = greedy(candidates)
    
    if not check_valid(best_seq):
        # Absolute fallback
        best_seq = [2, 3, 5, 7, 11, 13]
    
    return best_seq
