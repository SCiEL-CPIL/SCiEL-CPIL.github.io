"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def _factorize(n, primes):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 1) + 1
    return factors

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Get all primes up to 256
    primes = _sieve_primes(_N)
    
    # Factorize all numbers from 2 to 256
    factorizations = {}
    for i in range(2, _N + 1):
        factorizations[i] = _factorize(i, primes)
    
    # Build conflict graph: vertices are 2..256, edge if not coprime
    # Use bitmasks for fast conflict checking
    n_vertices = _N - 1  # vertices 0..254 represent numbers 2..256
    weights = [i + 2 for i in range(n_vertices)]
    
    # Precompute conflict masks
    conflict = [0] * n_vertices
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if math.gcd(i + 2, j + 2) != 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy initial solution for lower bound
    def is_coprime_to_all(num, selected_set):
        for s in selected_set:
            if math.gcd(num, s) != 1:
                return False
        return True
    
    greedy_selected = {1}
    for num in range(_N, 1, -1):
        if is_coprime_to_all(num, greedy_selected):
            greedy_selected.add(num)
    
    best_sum = sum(greedy_selected)
    best_set = set(greedy_selected)
    
    # Branch and bound for maximum weight independent set
    # Using greedy coloring bound
    def greedy_color_bound(candidates_mask):
        """
        Compute upper bound using greedy coloring.
        candidates_mask: bitmask of candidate vertices
        Returns: upper bound on max weight independent set weight
        """
        if candidates_mask == 0:
            return 0
        
        # Greedy coloring: assign colors to vertices
        # Color classes are independent sets, so bound = sum of max weight in each color
        uncolored = candidates_mask
        color_bound = 0
        
        while uncolored:
            # Pick vertex with highest weight among uncolored
            max_w = -1
            max_v = -1
            temp = uncolored
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if weights[v] > max_w:
                    max_w = weights[v]
                    max_v = v
                temp ^= lsb
            
            if max_v == -1:
                break
            
            # Find all vertices that can share this color (no conflict with max_v)
            color_class_mask = 0
            temp = uncolored
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if (conflict[max_v] & lsb) == 0:
                    color_class_mask |= lsb
                temp ^= lsb
            
            color_bound += max_w
            uncolored &= ~color_class_mask
        
        return color_bound
    
    # Branch and bound
    def branch_and_bound(candidates_mask, current_weight, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_weight > best_sum:
                best_sum = current_weight
                best_set = set(current_set)
            return
        
        # Upper bound: current_weight + greedy_color_bound(candidates_mask)
        ub = current_weight + greedy_color_bound(candidates_mask)
        if ub <= best_sum:
            return
        
        # Pick vertex with highest weight
        max_w = -1
        max_v = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if weights[v] > max_w:
                max_w = weights[v]
                max_v = v
            temp ^= lsb
        
        if max_v == -1:
            return
        
        # Branch 1: include max_v
        v = max_v
        # New candidates: those not conflicting with v
        new_candidates = candidates_mask & ~(conflict[v] | (1 << v))
        current_set.add(v + 2)
        branch_and_bound(new_candidates, current_weight + weights[v], current_set)
        current_set.discard(v + 2)
        
        # Branch 2: exclude max_v
        new_candidates = candidates_mask & ~(1 << v)
        branch_and_bound(new_candidates, current_weight, current_set)
    
    # Start B&B
    initial_mask = (1 << n_vertices) - 1
    branch_and_bound(initial_mask, 0, set())
    
    # Add 1 to best set (always optimal to include 1)
    best_set.add(1)
    
    # Final local search polish
    selected_list = list(best_set)
    selected_set = set(selected_list)
    current_sum = sum(selected_list)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try single additions
        for num in range(_N, 1, -1):
            if num not in selected_set and is_coprime_to_all(num, selected_set):
                selected_list.append(num)
                selected_set.add(num)
                current_sum += num
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps
        for remove_idx in range(len(selected_list)):
            if time.time() > deadline:
                break
            if selected_list[remove_idx] == 1:
                continue
            
            removed = selected_list[remove_idx]
            selected_set.remove(removed)
            
            best_addition = None
            best_gain = 0
            
            for num in range(_N, 1, -1):
                if num in selected_set or num == removed:
                    continue
                if is_coprime_to_all(num, selected_set):
                    gain = num - removed
                    if gain > best_gain:
                        best_gain = gain
                        best_addition = num
            
            if best_addition is not None:
                selected_list[remove_idx] = best_addition
                selected_set.add(best_addition)
                current_sum += best_gain
                improved = True
            else:
                selected_set.add(removed)
        
        if improved:
            continue
        
        # Try 2-for-2 swaps: remove two, add two different ones
        found_2swap = False
        n_sel = len(selected_list)
        for i in range(n_sel):
            if time.time() > deadline:
                break
            if selected_list[i] == 1:
                continue
            for j in range(i + 1, n_sel):
                if time.time() > deadline:
                    break
                if selected_list[j] == 1:
                    continue
                
                rem1 = selected_list[i]
                rem2 = selected_list[j]
                
                # Temporarily remove both
                selected_set.discard(rem1)
                selected_set.discard(rem2)
                
                # Find best pair to add
                best_pair = None
                best_pair_gain = 0
                
                # Try all pairs of candidates
                for a in range(_N, 1, -1):
                    if a in selected_set or a == rem1 or a == rem2:
                        continue
                    if not is_coprime_to_all(a, selected_set):
                        continue
                    for b in range(a - 1, 1, -1):
                        if b in selected_set or b == rem1 or b == rem2 or b == a:
                            continue
                        if not is_coprime_to_all(b, selected_set):
                            continue
                        if math.gcd(a, b) != 1:
                            continue
                        gain = a + b - rem1 - rem2
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_pair = (a, b)
                
                # Restore removed elements
                selected_set.add(rem1)
                selected_set.add(rem2)
                
                if best_pair is not None:
                    a, b = best_pair
                    selected_set.discard(rem1)
                    selected_set.discard(rem2)
                    selected_list[i] = a
                    selected_list[j] = b
                    selected_set.add(a)
                    selected_set.add(b)
                    current_sum += best_pair_gain
                    improved = True
                    found_2swap = True
                    break
            
            if found_2swap:
                break
        
        if not improved:
            break
    
    # Ensure result is a list of Python ints
    result = [int(x) for x in selected_list]
    return result
# EVOLVE-BLOCK-END
