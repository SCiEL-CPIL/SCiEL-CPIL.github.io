import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Precompute prime factorization vectors for numbers 2..128
    # Primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, 129) if is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Exponent vector for each number 2..128
    exp_vecs = {}
    for n in range(2, 129):
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)
    
    def check_valid(seq):
        """Check if sequence has all distinct consecutive block products using prefix sums."""
        n = len(seq)
        if n == 0:
            return True
        # Build prefix sum vectors: prefix[i] = sum of exp_vecs[seq[0:i]]
        prefix = [[0] * num_primes]
        for i in range(n):
            ej = exp_vecs[seq[i]]
            prev = prefix[-1]
            prefix.append([prev[k] + ej[k] for k in range(num_primes)])
        
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                # Block product from i to j-1 = prefix[j] - prefix[i]
                vec = tuple(prefix[j][k] - prefix[i][k] for k in range(num_primes))
                if vec in prods:
                    return False
                prods.add(vec)
        return True

    def check_append(seq, c):
        """Check if appending c to seq keeps all block products distinct.
        Only new blocks ending at position len(seq) need checking."""
        n = len(seq)
        if n == 0:
            return True
        # Compute prefix sums
        prefix = [[0] * num_primes]
        for i in range(n):
            ej = exp_vecs[seq[i]]
            prev = prefix[-1]
            prefix.append([prev[k] + ej[k] for k in range(num_primes)])
        
        # New element
        ej = exp_vecs[c]
        new_prefix = [prefix[-1][k] + ej[k] for k in range(num_primes)]
        
        # All existing block products
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                vec = tuple(prefix[j][k] - prefix[i][k] for k in range(num_primes))
                prods.add(vec)
        
        # New blocks: those ending at the new element (positions i..n-1, i from 0 to n)
        for i in range(n + 1):
            vec = tuple(new_prefix[k] - prefix[i][k] for k in range(num_primes))
            if vec in prods:
                return False
        return True
    
    def try_append(seq, c):
        """Try to append c to seq (c must be > seq[-1]). Return new seq or None."""
        if seq and c <= seq[-1]:
            return None
        if check_append(seq, c):
            return seq + [c]
        return None
    
    # Greedy: iterate through 2..128, add if possible
    seq = []
    for n in range(2, 129):
        if time.time() > deadline:
            break
        new_seq = try_append(seq, n)
        if new_seq is not None:
            seq = new_seq
    
    # Local search: try to improve by inserting, swapping, or 2-swap moves
    def local_search(seq, max_iter=2000):
        current = seq[:]
        used = set(current)
        remaining = [x for x in range(2, 129) if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try insertions
            for c in remaining[:]:
                # Try inserting c at each valid position
                new_seq = None
                for i in range(len(current) + 1):
                    if i > 0 and current[i-1] >= c:
                        continue
                    if i < len(current) and c >= current[i]:
                        continue
                    candidate = current[:i] + [c] + current[i:]
                    if check_valid(candidate):
                        new_seq = candidate
                        break
                if new_seq is not None:
                    current = new_seq
                    used = set(current)
                    remaining = [x for x in range(2, 129) if x not in used]
                    improved = True
                    break
            
            # Try swaps: remove one, add another
            if not improved:
                for i in range(len(current)):
                    removed = current[i]
                    base = current[:i] + current[i+1:]
                    for c in remaining:
                        if c <= removed:
                            continue
                        # Try inserting c into base
                        for j in range(len(base) + 1):
                            if j > 0 and base[j-1] >= c:
                                continue
                            if j < len(base) and c >= base[j]:
                                continue
                            candidate = base[:j] + [c] + base[j:]
                            if check_valid(candidate) and len(candidate) >= len(current):
                                current = candidate
                                used = set(current)
                                remaining = [x for x in range(2, 129) if x not in used]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try 2-swap: remove two elements, add two from remaining
            if not improved and len(current) >= 2:
                n_cur = len(current)
                for i in range(n_cur):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n_cur):
                        if time.time() > deadline:
                            break
                        base = current[:i] + current[i+1:j] + current[j+1:]
                        # Try adding two elements from remaining
                        for idx_a in range(len(remaining)):
                            if time.time() > deadline:
                                break
                            ca = remaining[idx_a]
                            for idx_b in range(idx_a + 1, len(remaining)):
                                cb = remaining[idx_b]
                                # Try all valid insertion positions for ca and cb
                                for pos_a in range(len(base) + 1):
                                    if pos_a > 0 and base[pos_a-1] >= ca:
                                        continue
                                    if pos_a < len(base) and ca >= base[pos_a]:
                                        continue
                                    temp = base[:pos_a] + [ca] + base[pos_a:]
                                    for pos_b in range(len(temp) + 1):
                                        if pos_b > 0 and temp[pos_b-1] >= cb:
                                            continue
                                        if pos_b < len(temp) and cb >= temp[pos_b]:
                                            continue
                                        candidate = temp[:pos_b] + [cb] + temp[pos_b:]
                                        if check_valid(candidate) and len(candidate) > len(current):
                                            current = candidate
                                            used = set(current)
                                            remaining = [x for x in range(2, 129) if x not in used]
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current
    
    # Improve with local search
    improved_seq = local_search(seq)
    if len(improved_seq) > len(seq):
        seq = improved_seq
    
    # Try multiple greedy variants with different orderings
    for _ in range(30):
        if time.time() > deadline:
            break
        # Try greedy with slight randomization
        import random
        rng = random.Random()
        candidates = list(range(2, 129))
        # Bias towards primes
        rng.shuffle(candidates)
        # Put primes first with some probability
        primes_first = [x for x in candidates if is_prime(x)]
        composites_first = [x for x in candidates if not is_prime(x)]
        if rng.random() < 0.5:
            order = primes_first + composites_first
        else:
            order = candidates
        rng.shuffle(order)
        
        s2 = []
        for n in order:
            if time.time() > deadline:
                break
            ns = try_append(s2, n)
            if ns is not None:
                s2 = ns
        
        s2 = local_search(s2, max_iter=500)
        if len(s2) > len(seq):
            seq = s2
    
    # Final polish: run local search on the best sequence found
    if time.time() < deadline - 5:
        final_seq = local_search(seq, max_iter=3000)
        if len(final_seq) > len(seq):
            seq = final_seq
    
    return seq
