import time
import random
from math import gcd

# Precompute prime factors for 1..511
def get_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Precompute for all numbers
PRIME_FACTORS = [set() for _ in range(512)]
for i in range(1, 512):
    PRIME_FACTORS[i] = get_prime_factors(i)

# Precompute values
VALUES = [0.0] * 512
for i in range(1, 512):
    VALUES[i] = 1.0 / (512 - i)

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s margin
    
    # 1. Greedy Initialization
    # Sort candidates by value descending
    candidates = list(range(1, 512))
    candidates.sort(key=lambda x: VALUES[x], reverse=True)
    
    chosen = []
    prime_counts = {}
    
    for a in candidates:
        # Check if a is coprime to all chosen
        valid = True
        for p in PRIME_FACTORS[a]:
            if prime_counts.get(p, 0) > 0:
                valid = False
                break
        if valid:
            chosen.append(a)
            for p in PRIME_FACTORS[a]:
                prime_counts[p] = prime_counts.get(p, 0) + 1
                
    current_score = sum(VALUES[a] for a in chosen)
    best_set = set(chosen)
    best_prime_counts = dict(prime_counts)
    best_score = current_score
    
    def try_add(a, set_to_check, counts_to_check):
        """Check if a can be added to set_to_check given counts_to_check. Returns gain if valid, else None."""
        for p in PRIME_FACTORS[a]:
            if counts_to_check.get(p, 0) > 0:
                return None
        return VALUES[a]

    def try_swap_1_1(a_out, b_in, set_to_check, counts_to_check):
        """Check if swapping a_out with b_in is valid. Returns gain if valid, else None."""
        # Temporarily remove a_out
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
        
        gain = try_add(b_in, set_to_check - {a_out}, counts_to_check)
        
        # Restore
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        if gain is not None:
            return gain - VALUES[a_out]
        return None

    def try_swap_1_2(a_out, b_in, c_in, set_to_check, counts_to_check):
        """Check if swapping a_out with b_in, c_in is valid. Returns gain if valid, else None."""
        # Temporarily remove a_out
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
        
        # Check b_in
        gain_b = try_add(b_in, set_to_check - {a_out}, counts_to_check)
        if gain_b is None:
            # Restore
            for p in PRIME_FACTORS[a_out]:
                counts_to_check[p] = counts_to_check.get(p, 0) + 1
            return None
            
        # Tentatively add b_in
        for p in PRIME_FACTORS[b_in]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        # Check c_in
        gain_c = try_add(c_in, (set_to_check - {a_out}) | {b_in}, counts_to_check)
        
        # Restore a_out removal and b_in addition
        for p in PRIME_FACTORS[b_in]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        if gain_c is not None:
            return gain_b + gain_c - VALUES[a_out]
        return None

    def try_swap_2_1(a_out1, a_out2, b_in, set_to_check, counts_to_check):
        """Check if swapping a_out1, a_out2 with b_in is valid. Returns gain if valid, else None."""
        # Temporarily remove a_out1 and a_out2
        for p in PRIME_FACTORS[a_out1]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
        for p in PRIME_FACTORS[a_out2]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
                
        gain = try_add(b_in, set_to_check - {a_out1, a_out2}, counts_to_check)
        
        # Restore
        for p in PRIME_FACTORS[a_out1]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
        for p in PRIME_FACTORS[a_out2]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        if gain is not None:
            return gain - VALUES[a_out1] - VALUES[a_out2]
        return None

    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        current_set = best_set
        current_counts = dict(best_prime_counts)
        current_score_val = best_score
        
        # 1. Try adding a single element
        best_add = None
        best_add_gain = 0
        for a in range(1, 512):
            if a not in current_set:
                gain = try_add(a, current_set, current_counts)
                if gain is not None and gain > best_add_gain:
                    best_add_gain = gain
                    best_add = a
        
        if best_add is not None:
            current_set.add(best_add)
            for p in PRIME_FACTORS[best_add]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_add_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue
            
        # 2. Try swapping 1 element with 1 element
        best_swap = None
        best_swap_gain = 0
        # To speed up, we can limit candidates or just do it all
        # 511 * 100 = 51100 checks. Fast enough.
        for a_out in list(current_set):
            for b_in in range(1, 512):
                if b_in not in current_set:
                    gain = try_swap_1_1(a_out, b_in, current_set, current_counts)
                    if gain is not None and gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (a_out, b_in)
                        
        if best_swap is not None:
            a_out, b_in = best_swap
            current_set.remove(a_out)
            for p in PRIME_FACTORS[a_out]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.add(b_in)
            for p in PRIME_FACTORS[b_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_swap_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue
            
        # 3. Try swapping 1 element with 2 elements
        # This is O(|S| * 511^2). 100 * 260k = 26M. Too slow in Python if done naively.
        # Optimization: Only try b_in, c_in from a pre-sorted list of top candidates?
        # Or limit the search.
        # Let's try a limited version: Top 100 candidates for b_in and c_in.
        candidates_sorted = sorted(range(1, 512), key=lambda x: VALUES[x], reverse=True)
        top_candidates = candidates_sorted[:100]
        
        best_1_for_2 = None
        best_1_for_2_gain = 0
        for a_out in list(current_set):
            for i, b_in in enumerate(top_candidates):
                if b_in in current_set: continue
                # Quick check: if b_in can't be added even if a_out is removed, skip?
                # try_swap_1_2 handles this.
                for c_in in top_candidates[i+1:]:
                    if c_in in current_set: continue
                    gain = try_swap_1_2(a_out, b_in, c_in, current_set, current_counts)
                    if gain is not None and gain > best_1_for_2_gain:
                        best_1_for_2_gain = gain
                        best_1_for_2 = (a_out, b_in, c_in)
                        
        if best_1_for_2 is not None:
            a_out, b_in, c_in = best_1_for_2
            current_set.remove(a_out)
            for p in PRIME_FACTORS[a_out]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.add(b_in)
            for p in PRIME_FACTORS[b_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_set.add(c_in)
            for p in PRIME_FACTORS[c_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_1_for_2_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue
            
        # 4. Try swapping 2 elements with 1 element
        # O(|S|^2 * 511). 100^2 * 511 = 5M. Might be okay.
        best_2_for_1 = None
        best_2_for_1_gain = 0
        set_list = list(current_set)
        for i in range(len(set_list)):
            for j in range(i+1, len(set_list)):
                a_out1 = set_list[i]
                a_out2 = set_list[j]
                for b_in in range(1, 512):
                    if b_in in current_set: continue
                    gain = try_swap_2_1(a_out1, a_out2, b_in, current_set, current_counts)
                    if gain is not None and gain > best_2_for_1_gain:
                        best_2_for_1_gain = gain
                        best_2_for_1 = (a_out1, a_out2, b_in)
                        
        if best_2_for_1 is not None:
            a_out1, a_out2, b_in = best_2_for_1
            current_set.remove(a_out1)
            for p in PRIME_FACTORS[a_out1]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.remove(a_out2)
            for p in PRIME_FACTORS[a_out2]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.add(b_in)
            for p in PRIME_FACTORS[b_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_2_for_1_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue

        # 5. Perturbation: Random restarts
        # If stuck, remove 5 random elements and greedily refill
        if len(current_set) > 5:
            # Remove 5 random elements
            to_remove = random.sample(list(current_set), 5)
            for a in to_remove:
                current_set.remove(a)
                for p in PRIME_FACTORS[a]:
                    current_counts[p] -= 1
                    if current_counts[p] == 0:
                        del current_counts[p]
            
            # Greedily add back best valid elements
            for a in candidates_sorted:
                if a not in current_set:
                    gain = try_add(a, current_set, current_counts)
                    if gain is not None:
                        current_set.add(a)
                        for p in PRIME_FACTORS[a]:
                            current_counts[p] = current_counts.get(p, 0) + 1
                            
            # Update score
            current_score_val = sum(VALUES[a] for a in current_set)
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True # Force another loop iteration to see if further improvements are possible
            continue

        break
        
    return list(best_set)
