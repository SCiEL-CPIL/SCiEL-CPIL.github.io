import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    # Generate all 5-smooth numbers (2^a * 3^b * 5^c) <= max_val
    # Store as (value, (a, b, c))
    smooth_numbers = []
    for a in range(15):  # 2^14 = 16384 > 10000
        for b in range(10):  # 3^9 = 19683 > 10000
            for c in range(7):  # 5^6 = 15625 > 10000
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= max_val:
                    smooth_numbers.append((val, (a, b, c)))
    
    # Sort by value
    smooth_numbers.sort()
    # Extract just the values and exponents
    smooth_vals = [s[0] for s in smooth_numbers]
    smooth_exps = [s[1] for s in smooth_numbers]
    val_to_idx = {v: i for i, v in enumerate(smooth_vals)}
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using sets (faster for small n)."""
        sumset = set()
        prodset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))

    def evaluate_from_exps(exp_triples):
        """Evaluate score given exponent triples (a,b,c) for 5-smooth numbers."""
        # Convert to values
        vals = [(2**a) * (3**b) * (5**c) for (a, b, c) in exp_triples]
        return evaluate(vals)

    def get_smooth_numbers(k, limit, count):
        """Generate smallest 'count' k-smooth numbers <= limit."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        primes = [p for p in primes if p <= k]
        smooth_list = []
        for x in range(1, limit + 1):
            t = x
            for p in primes:
                while t % p == 0:
                    t //= p
            if t == 1:
                smooth_list.append(x)
                if len(smooth_list) == count:
                    return smooth_list
        return smooth_list

    # 1. Generate Initial Candidates
    
    candidates = []
    
    # Candidate 1: Smallest 20 5-smooth numbers
    cands_5 = smooth_vals[:n]
    candidates.append(cands_5)
        
    # Candidate 2: Smallest 20 7-smooth numbers
    cands_7 = get_smooth_numbers(7, max_val, n)
    if len(cands_7) == n:
        candidates.append(cands_7)

    # Candidate 3: Smallest 20 11-smooth numbers
    cands_11 = get_smooth_numbers(11, max_val, n)
    if len(cands_11) == n:
        candidates.append(cands_11)
        
    # Candidate 4: Arithmetic Progression
    candidates.append(list(range(1, 21)))
    
    # Candidate 5: Random 5-smooth numbers
    for _ in range(5):
        random_indices = random.sample(range(len(smooth_vals)), n)
        cand = sorted([smooth_vals[i] for i in random_indices])
        candidates.append(cand)
    
    # Candidate 6: Geometric progression-like (powers of 2)
    candidates.append([2**i for i in range(14)] + [10000, 9999, 9998, 9997, 9996])
    
    # Candidate 7: Mix of small smooth numbers and larger ones
    cands_mixed = sorted(smooth_vals[:10] + [v for v in smooth_vals[10:30] if v <= max_val][:10])
    if len(cands_mixed) == n:
        candidates.append(cands_mixed)
    
    # Candidate 8: Densely packed small numbers
    candidates.append(list(range(1, 21)))

    # 2. Lattice-based Local Search in Exponent Space
    
    def lattice_local_search(start_set, max_iter=50000):
        """Local search in the 5-smooth exponent lattice with swap moves."""
        # Convert start_set to exponent triples
        current_exps = []
        for val in start_set:
            if val in val_to_idx:
                current_exps.append(smooth_exps[val_to_idx[val]])
            else:
                # Not 5-smooth, use nearest 5-smooth approximation
                best_idx = min(range(len(smooth_vals)), key=lambda i: abs(smooth_vals[i] - val))
                current_exps.append(smooth_exps[best_idx])
        
        # Deduplicate: if two exponents map to same value, adjust one
        seen_vals = set()
        unique_exps = []
        for exp in current_exps:
            val = (2**exp[0]) * (3**exp[1]) * (5**exp[2])
            if val not in seen_vals:
                unique_exps.append(exp)
                seen_vals.add(val)
        
        # If we don't have enough, fill with random 5-smooth numbers
        while len(unique_exps) < n:
            idx = random.randint(0, len(smooth_vals) - 1)
            exp = smooth_exps[idx]
            val = (2**exp[0]) * (3**exp[1]) * (5**exp[2])
            if val not in seen_vals:
                unique_exps.append(exp)
                seen_vals.add(val)
        
        current_exps = unique_exps[:n]
        current_vals = [(2**a) * (3**b) * (5**c) for (a, b, c) in current_exps]
        current_score = evaluate(current_vals)
        
        local_best_exps = current_exps[:]
        local_best_score = current_score
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Two types of moves: single-element change or swap
            if random.random() < 0.5:
                # Single element change
                idx = random.randint(0, n - 1)
                a, b, c = current_exps[idx]
                
                new_a = a + random.randint(-2, 2)
                new_b = b + random.randint(-2, 2)
                new_c = c + random.randint(-2, 2)
                
                new_a = max(0, min(14, new_a))
                new_b = max(0, min(9, new_b))
                new_c = max(0, min(6, new_c))
                
                new_val = (2**new_a) * (3**new_b) * (5**new_c)
                
                if new_val in current_vals or new_val < 1 or new_val > max_val:
                    continue
                
                new_exps = current_exps[:]
                new_exps[idx] = (new_a, new_b, new_c)
                new_vals = [(2**a2) * (3**b2) * (5**c2) for (a2, b2, c2) in new_exps]
                
                if len(set(new_vals)) != n:
                    continue
            else:
                # Swap two elements
                idx1, idx2 = random.sample(range(n), 2)
                a1, b1, c1 = current_exps[idx1]
                a2, b2, c2 = current_exps[idx2]
                
                # Try swapping exponents
                new_exps = current_exps[:]
                new_exps[idx1] = (a2, b2, c2)
                new_exps[idx2] = (a1, b1, c1)
                new_vals = [(2**a3) * (3**b3) * (5**c3) for (a3, b3, c3) in new_exps]
                
                if len(set(new_vals)) != n:
                    continue
            
            new_score = evaluate(new_vals)
            
            if new_score < current_score:
                current_exps = new_exps
                current_vals = new_vals
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_exps = current_exps[:]
            else:
                # Simulated annealing with higher acceptance
                if random.random() < 0.1:
                    if new_score < current_score * 1.2:
                        current_exps = new_exps
                        current_vals = new_vals
                        current_score = new_score
        
        # Convert best exponents back to values
        best_vals = sorted([(2**a) * (3**b) * (5**c) for (a, b, c) in local_best_exps])
        return best_vals, local_best_score

    # Run lattice local search from each candidate
    for cand in candidates:
        if time.time() > deadline:
            break
            
        if len(set(cand)) != n or max(cand) > max_val or min(cand) < 1:
            continue
            
        res_set, res_score = lattice_local_search(cand)
        
        if res_score < best_score:
            best_score = res_score
            best_set = res_set[:]

    # 3. Final polish: standard local search on best set found with swap moves
    if best_set is not None and time.time() < deadline - 5.0:
        current = best_set[:]
        current_score = best_score
        
        for i in range(30000):
            if time.time() > deadline:
                break
            
            # Two types of moves: single replace or swap
            if random.random() < 0.6:
                idx = random.randint(0, n - 1)
                
                # Try replacing with a 5-smooth number
                if random.random() < 0.8:
                    new_val = smooth_vals[random.randint(0, len(smooth_vals) - 1)]
                else:
                    new_val = random.randint(1, max_val)
                
                if new_val in current or new_val < 1 or new_val > max_val:
                    continue
                
                temp_set = current[:]
                temp_set[idx] = new_val
            else:
                # Swap two elements with random new values
                idx1, idx2 = random.sample(range(n), 2)
                
                new_val1 = smooth_vals[random.randint(0, len(smooth_vals) - 1)]
                new_val2 = smooth_vals[random.randint(0, len(smooth_vals) - 1)]
                
                if new_val1 == new_val2:
                    continue
                if new_val1 in current or new_val2 in current:
                    continue
                if new_val1 < 1 or new_val1 > max_val or new_val2 < 1 or new_val2 > max_val:
                    continue
                
                temp_set = current[:]
                temp_set[idx1] = new_val1
                temp_set[idx2] = new_val2
            
            temp_set.sort()
            
            if len(set(temp_set)) != n:
                continue
            
            new_score = evaluate(temp_set)
            
            if new_score < current_score:
                current = temp_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            elif random.random() < 0.05 and new_score < current_score * 1.15:
                current = temp_set
                current_score = new_score

    # Fallback
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set
