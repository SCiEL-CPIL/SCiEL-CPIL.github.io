import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    def sf_kernel(n):
        """Return squarefree kernel of n."""
        if n <= 1:
            return 1
        result = 1
        f = n
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    result *= d
            d += 1 if d == 2 else 2
        if f > 1:
            result *= f
        return result
    
    # Precompute squarefree kernels up to 10210
    max_val = 10210
    sf = [0] * (max_val + 1)
    sf[1] = 1
    for i in range(2, max_val + 1):
        sf[i] = sf_kernel(i)
    
    # Target: squarefree kernel of 210 = 2*3*5*7 = 210
    target = 210
    
    # Check size 1: find n in [211, 10210] with sf[n] == 210
    best_max = max_val  # worst case
    best_list = None
    
    # Size 1
    for n in range(211, max_val + 1):
        if sf[n] == target:
            if n < best_max:
                best_max = n
                best_list = [n]
            break  # first one is smallest
    
    # Build a map from sf value to list of numbers (sorted) with that sf value
    sf_to_nums = defaultdict(list)
    for n in range(211, max_val + 1):
        s = sf[n]
        sf_to_nums[s].append(n)
    for s in sf_to_nums:
        sf_to_nums[s].sort()
    
    # Also keep sf_to_min for quick lookup
    sf_to_min = {}
    for n in range(211, max_val + 1):
        s = sf[n]
        if s not in sf_to_min or n < sf_to_min[s]:
            sf_to_min[s] = n
    
    # For size 2: need sf[a] * sf[b] to have squarefree part 210
    # The squarefree part of product of two squarefree numbers x*y is x*y/gcd(x,y)^2
    # So we need: sf[a] * sf[b] / gcd(sf[a], sf[b])^2 = 210
    # Let x = sf[a], y = sf[b], g = gcd(x,y)
    # x*y/g^2 = 210 => (x/g)*(y/g) = 210
    # Let x' = x/g, y' = y/g, then x'*y' = 210 and gcd(x',y')=1
    
    def coprime_factor_pairs(n):
        pairs = []
        for d in range(1, int(math.isqrt(n)) + 1):
            if n % d == 0:
                e = n // d
                if math.gcd(d, e) == 1:
                    pairs.append((d, e))
                    if d != e:
                        pairs.append((e, d))
        return pairs
    
    pairs_210 = coprime_factor_pairs(210)
    
    # For size 2, try all coprime factor pairs and g values
    for x_prime, y_prime in pairs_210:
        if time.time() > deadline:
            break
        max_g = max_val // max(x_prime, y_prime)
        # g must be coprime to 210
        g = 1
        while g <= max_g:
            if math.gcd(g, 210) == 1:
                sa = x_prime * g
                sb = y_prime * g
                if sa in sf_to_min and sb in sf_to_min:
                    a = sf_to_min[sa]
                    b = sf_to_min[sb]
                    if a != b:
                        m = max(a, b)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b]
            g += 1
    
    # Greedy set-cover approach for size 3+
    # The product of selected integers multiplied by 210 must be a perfect square.
    # Equivalently, the XOR (mod 2) of the prime exponent vectors of all selected integers
    # plus the prime exponent vector of 210 must be all zeros.
    # 
    # Represent each number by its squarefree kernel. The condition is:
    # sf[210] * sf[n1] * sf[n2] * ... * sf[nk] must have all prime exponents even.
    # This is equivalent to: the product of all sf values (including 210) is a perfect square.
    #
    # Greedy: start with "need" = 210 (the sf of 210). At each step, pick the smallest
    # number n > 210 such that sf[n] "reduces" the remaining need most effectively.
    # After picking n, the new need is sf_part(need * sf[n]) where sf_part gives squarefree part.
    
    def sf_part_of_product(*sf_vals):
        """Compute squarefree part of product of squarefree numbers."""
        # For squarefree numbers, sf_part(x*y) = x*y/gcd(x,y)^2
        # For multiple: iteratively combine
        result = 1
        for v in sf_vals:
            g = math.gcd(result, v)
            result = result * v // (g * g)
        return result
    
    # Greedy search for size 3, 4, 5...
    # We want to minimize max(selected), so we should prefer smaller numbers.
    # Strategy: for each candidate max M, try to find a set with all elements <= M.
    # But that's expensive. Instead, use greedy with a priority on smaller numbers.
    
    # Alternative greedy: repeatedly pick the smallest n that makes progress toward
    # making the product a perfect square, with a bias toward smaller values.
    
    # Let's try a different greedy: 
    # 1. Start with need = 210
    # 2. At each step, among all n > 210 not yet selected, find the one that minimizes
    #    max(n, current_max) subject to making progress (reducing the "distance" to 1)
    # 3. A good heuristic: pick the smallest n such that sf_part(need * sf[n]) < need
    #    (i.e., the squarefree part gets smaller)
    
    # Actually, let's think about this differently. The "distance" from need to 1 can be
    # measured by the number of prime factors of need (with multiplicity in the squarefree sense).
    # When we multiply need by sf[n], the new need is sf_part(need * sf[n]).
    # This reduces the need if sf[n] shares prime factors with need.
    
    # Let's define: progress = number of prime factors of sf_part(need * sf[n]) 
    #              - number of prime factors of need
    # We want to minimize progress (most negative is best), and among ties, minimize n.
    
    def num_prime_factors(n):
        """Count total prime factors with multiplicity for squarefree n."""
        if n <= 1:
            return 0
        count = 0
        f = n
        d = 2
        while d * d <= f:
            if f % d == 0:
                count += 1
                f //= d
            d += 1 if d == 2 else 2
        if f > 1:
            count += 1
        return count
    
    # Precompute num_prime_factors for all sf values
    sf_np = {}
    for s in sf_to_nums:
        sf_np[s] = num_prime_factors(s)
    
    # Greedy search
    # To keep it efficient, limit the number of steps and the range of candidates considered.
    
    def greedy_search(max_steps, candidate_limit=None):
        """
        Greedy search for a set of integers whose product with 210 is a perfect square.
        Returns (list_of_ints, max_value) or (None, None) if no solution found.
        """
        need = 210
        selected = []
        used = set()
        
        for step in range(max_steps):
            if need == 1:
                break
            
            best_n = None
            best_new_need = None
            best_score = None
            
            # Consider candidates: all n > 210 not in used
            # To limit search, consider n up to some limit
            upper = candidate_limit if candidate_limit else best_max - 1
            upper = min(upper, max_val)
            
            for n in range(211, upper + 1):
                if n in used:
                    continue
                s = sf[n]
                new_need = sf_part_of_product(need, s)
                
                # Score: prefer smaller new_need, then smaller n
                # Use (num_prime_factors(new_need), n) as score
                np_new = sf_np.get(new_need, num_prime_factors(new_need))
                score = (np_new, n)
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_n = n
                    best_new_need = new_need
            
            if best_n is None:
                break
            
            if best_new_need >= need and best_new_need != 1:
                # No progress, break
                break
            
            selected.append(best_n)
            used.add(best_n)
            need = best_new_need
        
        if need == 1 and selected:
            m = max(selected)
            return (selected, m)
        return (None, None)
    
    # Try greedy with various step limits
    for steps in range(3, 11):
        if time.time() > deadline:
            break
        result, m = greedy_search(steps)
        if result is not None and m < best_max:
            best_max = m
            best_list = result
    
    # Also try a more targeted approach: for size 3, use the pair-based method
    # For each pair (a,b), compute what c needs to be
    if best_list is None or best_max > 850:
        # For size 3: iterate over pairs (a,b) with a<b, both in [211, best_max-1]
        # Compute what sf[c] needs to be, then look up c
        # sf[210*a*b*c] = 1 => sf[c] = sf_part(210 * sf[a] * sf[b])
        
        # Precompute sf_part for combinations of sf values
        # To limit work, only consider pairs where max(a,b) < best_max
        
        # Build a set of sf values for quick lookup
        sf_set = set(sf_to_nums.keys())
        
        # For efficiency, precompute sf_part(210 * sf[a] * sf[b]) for common sf values
        # But this could be large. Let's just compute on the fly for limited pairs.
        
        limit = min(600, best_max)
        for a in range(211, limit):
            if time.time() > deadline:
                break
            sa = sf[a]
            for b in range(a + 1, limit):
                if time.time() > deadline:
                    break
                sb = sf[b]
                # Compute sf_part(210 * sa * sb)
                temp = 210 * sa
                g = math.gcd(temp, sb)
                needed_sf = temp * sb // (g * g)
                
                if needed_sf in sf_to_min:
                    c = sf_to_min[needed_sf]
                    if c > 210 and c != a and c != b:
                        m = max(a, b, c)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b, c]
    
    # If we found a good result, return it. Otherwise, fall back to size 1.
    if best_list is None:
        best_list = [840]
        best_max = 840
    
    # Ensure all values are Python ints
    result = [int(x) for x in best_list]
    return result
