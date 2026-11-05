import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    # Precompute prime factorizations for all numbers 2..128
    prime_factors = {}
    for n in range(2, 129):
        factors = {}
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        prime_factors[n] = factors

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]

    def vec_key(vec):
        """Convert dict to sorted tuple for hashing."""
        return tuple(sorted(vec.items()))

    def is_valid_fast(s):
        """
        Check if all consecutive block products are distinct using prime exponent vectors.
        """
        n = len(s)
        if n == 0:
            return True
        
        prods = set()
        for i in range(n):
            vec = {}
            for j in range(i, n):
                pf = prime_factors[s[j]]
                for p, e in pf.items():
                    vec[p] = vec.get(p, 0) + e
                key = vec_key(vec)
                if key in prods:
                    return False
                prods.add(key)
        return True

    def compute_all_prods(seq):
        """Compute set of all block product keys for a sequence."""
        prods = set()
        n = len(seq)
        for i in range(n):
            vec = {}
            for j in range(i, n):
                pf = prime_factors[seq[j]]
                for p, e in pf.items():
                    vec[p] = vec.get(p, 0) + e
                prods.add(vec_key(vec))
        return prods

    def find_collisions(seq):
        """Find all pairs of blocks with the same product."""
        n = len(seq)
        prod_map = defaultdict(list)
        for i in range(n):
            vec = {}
            for j in range(i, n):
                pf = prime_factors[seq[j]]
                for p, e in pf.items():
                    vec[p] = vec.get(p, 0) + e
                key = vec_key(vec)
                prod_map[key].append((i, j))
        
        collisions = []
        for key, blocks in prod_map.items():
            if len(blocks) > 1:
                for a in range(len(blocks)):
                    for b in range(a + 1, len(blocks)):
                        collisions.append((blocks[a], blocks[b]))
        return collisions

    def is_valid_incremental_check(seq, new_val, pos):
        """
        Check if inserting new_val at position pos in seq keeps all block products distinct.
        Only checks blocks that include the new element.
        """
        n = len(seq)
        new_n = n + 1
        
        # Compute old products
        old_prods = compute_all_prods(seq)
        
        # Check new blocks that include position pos
        for a in range(pos + 1):
            vec = {}
            # Add elements from a to pos-1 in seq
            for j in range(a, pos):
                pf = prime_factors[seq[j]]
                for p, e in pf.items():
                    vec[p] = vec.get(p, 0) + e
            # Add new_val
            pf = prime_factors[new_val]
            for p, e in pf.items():
                vec[p] = vec.get(p, 0) + e
            # Add elements from pos to end in seq
            for j in range(pos, n):
                pf = prime_factors[seq[j]]
                for p, e in pf.items():
                    vec[p] = vec.get(p, 0) + e
                key = vec_key(vec)
                if key in old_prods:
                    return False
                old_prods.add(key)
            # Also check the block ending at new_val (a to pos)
            vec2 = {}
            for j in range(a, pos):
                pf = prime_factors[seq[j]]
                for p, e in pf.items():
                    vec2[p] = vec2.get(p, 0) + e
            pf = prime_factors[new_val]
            for p, e in pf.items():
                vec2[p] = vec2.get(p, 0) + e
            key2 = vec_key(vec2)
            if key2 in old_prods:
                return False
            old_prods.add(key2)
        
        return True

    def repair_collisions(seq, deadline):
        """
        Targeted local repair: find collisions and try to fix them by removing/adding elements.
        """
        seq = sorted(seq)
        current_best = seq[:]
        current_len = len(seq)
        
        max_iterations = 50
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            collisions = find_collisions(current_best)
            if not collisions:
                break
            
            # For each collision, try to fix it
            fixed = False
            for (i1, j1), (i2, j2) in collisions[:5]:  # Limit to first 5 collisions
                if time.time() > deadline - 5:
                    break
                
                # Try removing one element from each colliding block
                candidates_to_remove = set()
                for idx in range(i1, j1 + 1):
                    candidates_to_remove.add(idx)
                for idx in range(i2, j2 + 1):
                    candidates_to_remove.add(idx)
                
                for rem_idx in candidates_to_remove:
                    if time.time() > deadline - 5:
                        break
                    
                    new_seq = current_best[:rem_idx] + current_best[rem_idx+1:]
                    if not is_valid_fast(new_seq):
                        continue
                    
                    # Try to add back a different element
                    removed_val = current_best[rem_idx]
                    missing = [x for x in all_numbers if x not in set(new_seq)]
                    
                    for new_val in missing:
                        # Find insertion position
                        pos = 0
                        while pos < len(new_seq) and new_seq[pos] < new_val:
                            pos += 1
                        
                        test_seq = new_seq[:pos] + [new_val] + new_seq[pos:]
                        if is_valid_fast(test_seq):
                            current_best = test_seq
                            current_len = len(current_best)
                            fixed = True
                            break
                    
                    if fixed:
                        break
                
                if fixed:
                    break
            
            if not fixed:
                # Try single element swaps
                improved = False
                for idx in range(len(current_best)):
                    if time.time() > deadline - 5:
                        break
                    for new_val in all_numbers:
                        if new_val == current_best[idx]:
                            continue
                        if new_val in current_best:
                            continue
                        
                        candidate = current_best[:idx] + [new_val] + current_best[idx+1:]
                        candidate.sort()
                        if len(set(candidate)) != len(candidate):
                            continue
                        
                        if is_valid_fast(candidate):
                            current_best = candidate
                            current_len = len(candidate)
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
        
        return current_best, current_len

    def beam_search(seed, beam_width, deadline):
        """
        Beam search over insertion decisions.
        """
        seed = sorted(seed)
        if not is_valid_fast(seed):
            return seed, len(seed)
        
        current_set = set(seed)
        remaining = [x for x in all_numbers if x not in current_set]
        
        # Beam: list of (sequence, length)
        beam = [(seed[:], len(seed))]
        
        for c in remaining:
            if time.time() > deadline - 10:
                break
            
            new_beam = []
            for seq, length in beam:
                # Find insertion position
                pos = 0
                while pos < len(seq) and seq[pos] < c:
                    pos += 1
                
                new_seq = seq[:pos] + [c] + seq[pos:]
                
                if is_valid_fast(new_seq):
                    new_beam.append((new_seq, length + 1))
            
            # Also keep the option of not adding c
            for seq, length in beam:
                new_beam.append((seq, length))
            
            # Sort by length descending, keep top beam_width
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
            
            # Early exit if we've reached all numbers
            if len(beam[0][0]) == 127:
                break
        
        best = max(beam, key=lambda x: x[1])
        return best[0], best[1]

    def greedy_insert(seed, check_order_strategy, deadline):
        """
        Given a valid seed, try to insert remaining numbers.
        """
        seed = sorted(seed)
        if not is_valid_fast(seed):
            return seed, len(seed)
            
        current_seq = seed[:]
        current_len = len(current_seq)
        current_set = set(current_seq)
        
        remaining = [x for x in all_numbers if x not in current_set]
        
        if check_order_strategy == 'ascending':
            check_order = remaining[:]
        elif check_order_strategy == 'descending':
            check_order = remaining[::-1]
        elif check_order_strategy == 'random':
            check_order = remaining[:]
            random.shuffle(check_order)
        elif check_order_strategy == 'middle_out':
            mid = len(remaining) // 2
            check_order = []
            for d in range(mid + 1):
                if mid - d >= 0:
                    check_order.append(remaining[mid - d])
                if d > 0 and mid + d < len(remaining):
                    check_order.append(remaining[mid + d])
        else:
            check_order = remaining[:]
        
        for c in check_order:
            if time.time() > deadline - 5:
                break
                
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < c:
                pos += 1
            
            if pos < len(current_seq) and current_seq[pos] == c:
                continue
                
            new_seq = current_seq[:pos] + [c] + current_seq[pos:]
            
            if is_valid_fast(new_seq):
                current_seq = new_seq
                current_set.add(c)
                current_len += 1
        
        return current_seq, current_len

    best_seq = primes[:]
    best_len = len(primes)
    
    random.seed(42)
    
    # Phase 1: Greedy insertion with multiple strategies
    strategies = [
        ('primes_asc', primes[:], 'ascending'),
        ('primes_desc', primes[:], 'descending'),
        ('primes_rand', primes[:], 'random'),
        ('small_asc', all_numbers[:20], 'ascending'),
        ('small_desc', all_numbers[:20], 'descending'),
        ('small_rand', all_numbers[:20], 'random'),
        ('small_mid', all_numbers[:20], 'middle_out'),
    ]
    
    # Add a random seed
    rand_seed = []
    for x in sorted(random.sample(all_numbers, 15)):
        test = rand_seed + [x]
        pos = 0
        while pos < len(test) and test[pos] < x:
            pos += 1
        test.insert(pos, x)
        if is_valid_fast(test):
            rand_seed = test
        else:
            test.pop(pos)
    if rand_seed:
        strategies.append(('rand_asc', rand_seed[:], 'ascending'))
        strategies.append(('rand_desc', rand_seed[:], 'descending'))
        strategies.append(('rand_rand', rand_seed[:], 'random'))
    
    # Primes + squarefree composites
    sq_free_composites = [x for x in composites if all(prime_factors[x].get(p, 0) == 1 for p in prime_factors[x])]
    seed_pf = primes[:] + sq_free_composites[:10]
    seed_pf.sort()
    if is_valid_fast(seed_pf):
        strategies.append(('pf_asc', seed_pf[:], 'ascending'))
        strategies.append(('pf_desc', seed_pf[:], 'descending'))
    
    for name, seed, strategy in strategies:
        if time.time() > deadline - 20:
            break
            
        result, result_len = greedy_insert(seed, strategy, deadline)
        
        if result_len > best_len:
            best_len = result_len
            best_seq = result[:]
    
    # Phase 2: Beam search from best sequence
    if time.time() < deadline - 20:
        beam_result, beam_len = beam_search(best_seq, beam_width=3, deadline=deadline)
        if beam_len > best_len:
            best_len = beam_len
            best_seq = beam_result[:]
    
    # Phase 3: Targeted collision repair
    if time.time() < deadline - 10:
        repaired, repaired_len = repair_collisions(best_seq, deadline)
        if repaired_len > best_len:
            best_len = repaired_len
            best_seq = repaired[:]
    
    # Phase 4: Final polish with greedy on the repaired sequence
    if time.time() < deadline - 5:
        final_result, final_len = greedy_insert(best_seq, 'ascending', deadline)
        if final_len > best_len:
            best_len = final_len
            best_seq = final_result[:]
    
    return best_seq
