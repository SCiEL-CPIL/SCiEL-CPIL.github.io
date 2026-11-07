import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin before 60s hard limit
    sys.setrecursionlimit(10000)
    
    def check_addition(current_sums, candidate):
        """
        Try to add candidate to current set.
        Returns new_sums if valid, else None.
        """
        if candidate in current_sums:
            return None
        
        new_sums = set()
        for s in current_sums:
            ns = s + candidate
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        
        if candidate in new_sums:
            return None
        new_sums.add(candidate)
        
        all_sums = current_sums.union(new_sums)
        
        # Check divisibility: for each new sum, check against all other sums
        for s in new_sums:
            for t in all_sums:
                if s == t:
                    continue
                if t % s == 0 or s % t == 0:
                    return None
        
        return new_sums

    def greedy_construct(candidates, deadline):
        """Greedily construct a valid set."""
        current_elements = set()
        current_sums = set()
        
        for c in candidates:
            if time.time() > deadline:
                break
            new_sums = check_addition(current_sums, c)
            if new_sums is not None:
                current_elements.add(c)
                current_sums = current_sums.union(new_sums)
        
        return current_elements, current_sums

    def local_search(best_set, candidates, deadline):
        """Try to improve by swapping elements."""
        best_sums = set()
        elements = sorted(best_set)
        for mask in range(1, 1 << len(elements)):
            s = 0
            for i in range(len(elements)):
                if mask & (1 << i):
                    s += elements[i]
            best_sums.add(s)
        
        best_size = len(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            pool = [c for c in candidates if c not in best_set]
            random.shuffle(pool)
            
            # Try adding one element
            for c in pool:
                if time.time() > deadline:
                    break
                new_sums = check_addition(best_sums, c)
                if new_sums is not None:
                    best_set = best_set.union({c})
                    best_sums = best_sums.union(new_sums)
                    best_size = len(best_set)
                    improved = True
            
            # Try removing one and adding another
            for remove_c in list(best_set):
                if time.time() > deadline:
                    break
                reduced_set = best_set - {remove_c}
                reduced_sums = set()
                elems = sorted(reduced_set)
                for mask in range(1, 1 << len(elems)):
                    s = 0
                    for i in range(len(elems)):
                        if mask & (1 << i):
                            s += elems[i]
                    reduced_sums.add(s)
                
                pool2 = [c for c in candidates if c not in reduced_set]
                random.shuffle(pool2)
                
                for add_c in pool2:
                    if time.time() > deadline:
                        break
                    new_sums = check_addition(reduced_sums, add_c)
                    if new_sums is not None:
                        new_set = reduced_set.union({add_c})
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_sums = reduced_sums.union(new_sums)
                            best_size = len(best_set)
                            improved = True
                            break
        
        return best_set

    # Generate candidates with priority: primes, prime powers, then others
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def is_prime_power(n):
        if n < 4:
            return False
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
        return False
    
    primes = []
    prime_powers = []
    others = []
    
    for n in range(3, 257):
        if is_prime(n):
            primes.append(n)
        elif is_prime_power(n):
            prime_powers.append(n)
        else:
            others.append(n)
    
    # Priority order: primes first, then prime powers, then others
    candidates = primes + prime_powers + others
    
    best_set = set()
    best_sums = set()
    best_size = 0
    
    # Phase 1: Multiple greedy passes
    # Sorted ascending
    result = greedy_construct(sorted(candidates), deadline)
    if len(result[0]) > best_size:
        best_set, best_sums = result
        best_size = len(best_set)
    
    # Sorted descending
    result = greedy_construct(sorted(candidates, reverse=True), deadline)
    if len(result[0]) > best_size:
        best_set, best_sums = result
        best_size = len(best_set)
    
    # Random shuffles
    for i in range(30):
        if time.time() > deadline - 10.0:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_construct(shuffled, deadline)
        if len(result[0]) > best_size:
            best_set, best_sums = result
            best_size = len(best_set)
    
    # Phase 2: Local search
    if time.time() < deadline - 5.0:
        improved = local_search(best_set, candidates, deadline - 5.0)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
            # Recompute sums
            elements = sorted(best_set)
            best_sums = set()
            for mask in range(1, 1 << len(elements)):
                s = 0
                for i in range(len(elements)):
                    if mask & (1 << i):
                        s += elements[i]
                best_sums.add(s)
    
    # Final verification
    if best_set:
        elements = sorted(best_set)
        all_sums = set()
        valid = True
        for mask in range(1, 1 << len(elements)):
            s = 0
            for i in range(len(elements)):
                if mask & (1 << i):
                    s += elements[i]
            if s in all_sums:
                valid = False
                break
            all_sums.add(s)
        
        if valid:
            sums_list = sorted(all_sums)
            for i in range(len(sums_list)):
                for j in range(i + 1, len(sums_list)):
                    if sums_list[j] % sums_list[i] == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return [int(x) for x in sorted(best_set)]
    
    # Fallback
    return [3]
