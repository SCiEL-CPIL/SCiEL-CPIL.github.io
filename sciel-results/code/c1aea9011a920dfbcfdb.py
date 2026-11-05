"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if S is sum-free."""
        S_set = set(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                if a + b in S_set:
                    return False
        return True

    def find_violations(S):
        """Return list of (a, b, c) such that a+b=c, all in S."""
        S_set = set(S)
        viol = []
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                c = a + b
                if c in S_set:
                    viol.append((a, b, c))
        return viol

    def can_add(S_set, S_list, x):
        """Check if x can be added to sum-free set S without violation."""
        for a in S_list:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def repair_set(S, V):
        """Remove elements to make S sum-free, then greedily expand."""
        S_list = list(S)
        S_set = set(S_list)
        V_set = set(V)
        
        # Phase 1: Remove elements to fix violations
        while True:
            viol = find_violations(S_list)
            if not viol:
                break
            
            counts = Counter()
            for a, b, c in viol:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
            
            if counts:
                worst = max(counts, key=counts.get)
                if worst in S_set:
                    S_list.remove(worst)
                    S_set.discard(worst)
                else:
                    break
            else:
                break

        # Phase 2: Greedily add safe elements
        improved = True
        while improved:
            improved = False
            candidates = [x for x in V if x not in S_set]
            for x in candidates:
                if can_add(S_set, S_list, x):
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                    break
        return S_list

    def local_search_1opt(S, V, max_iter=200):
        """Try to improve S by swapping elements or adding/removing."""
        S_list = list(S)
        S_set = set(S_list)
        V_set = set(V)
        
        for _ in range(max_iter):
            # Try adding
            added = False
            candidates = [x for x in V if x not in S_set]
            for x in candidates:
                if can_add(S_set, S_list, x):
                    S_list.append(x)
                    S_set.add(x)
                    added = True
                    break
            
            if added:
                continue
            
            # Try 1-opt swap: remove one, try to add two
            # For simplicity, try removing each element and see if we can add back more
            best_swap = None
            best_gain = 0
            
            # Try removing each element
            for i, elem in enumerate(S_list):
                temp_set = set(S_list)
                temp_set.discard(elem)
                temp_list = [x for x in S_list if x != elem]
                
                # Try adding candidates
                added_count = 0
                new_elements = []
                for x in V:
                    if x not in temp_set:
                        if can_add(temp_set, temp_list, x):
                            temp_set.add(x)
                            temp_list.append(x)
                            added_count += 1
                
                gain = added_count - 1
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (i, new_elements)
            
            if best_swap and best_gain > 0:
                i, new_elements = best_swap
                S_list.pop(i)
                S_set.discard(S_list[i] if i < len(S_list) else S_list[-1])
                # Rebuild properly
                S_list = [x for x in S_list if x != S_list[i]] if i < len(S_list) else S_list
                S_set = set(S_list)
                for x in new_elements:
                    if x not in S_set:
                        S_list.append(x)
                        S_set.add(x)
            else:
                break
        
        return S_list

    def search_instance(elements, seed_offset=0):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        
        # Generate multiple initial candidates
        candidates = []
        
        # 1. Odds
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)
        
        # 2. Largest values
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6]:
            k = int(len(elements) * frac)
            candidates.append(sorted_desc[:k])
        
        # 3. Smallest absolute values
        sorted_abs = sorted(elements, key=lambda x: abs(x))
        for frac in [0.3, 0.4, 0.5]:
            k = int(len(elements) * frac)
            candidates.append(sorted_abs[:k])
        
        # 4. Random samples
        for _ in range(8):
            k = int(len(elements) * rng.uniform(0.3, 0.8))
            candidates.append(rng.sample(elements, k))
            
        # 5. Positive numbers
        pos = [x for x in elements if x > 0]
        candidates.append(pos)
        
        # 6. Negative numbers
        neg = [x for x in elements if x < 0]
        candidates.append(neg)
        
        # 7. High values (top 1/3)
        high = sorted(elements, reverse=True)[:len(elements)//3]
        candidates.append(high)
        
        # 8. Low values (bottom 1/3)
        low = sorted(elements)[:len(elements)//3]
        candidates.append(low)
        
        # Process each candidate
        for cand in candidates:
            S = repair_set(cand, elements)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        # Additional random restarts if time permits
        start_restarts = time.time()
        for _ in range(50):
            if time.time() - start_restarts > 3:
                break
            k = int(len(elements) * rng.uniform(0.3, 0.8))
            cand = rng.sample(elements, k)
            S = repair_set(cand, elements)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        # Local search on best
        if time.time() - start_time < DEADLINE - 10:
            improved = local_search_1opt(best, elements, max_iter=100)
            if len(improved) > best_size:
                best = improved
        
        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        result = search_instance(elems, seed_offset=801 + i)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
