import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin
    
    def score(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    def score_components(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return len(sums), len(prods)

    # Generate pool of smooth numbers (2,3,5,7-smooth) <= 10000
    primes = [2, 3, 5, 7]
    pool = []
    for i in range(1, 10001):
        x = i
        for p in primes:
            while x % p == 0:
                x //= p
        if x == 1:
            pool.append(i)
            
    # Ensure pool has enough elements
    if len(pool) < 100:
        pool = list(range(1, 10001))

    best_set = None
    best_score = float('inf')
    
    def sa_search(initial_set):
        nonlocal best_set, best_score
        current = initial_set[:]
        current_score = score(current)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
            
        T = 5.0
        cooling = 0.995
        
        while time.time() - start_time < timeout:
            idx = random.randint(0, 19)
            old_val = current[idx]
            
            new_val = random.choice(pool)
            
            if new_val == old_val:
                continue
            if new_val in current:
                continue
                
            new_set = current[:]
            new_set[idx] = new_val
            new_score = score(new_set)
            
            delta = new_score - current_score
            
            if delta < 0:
                current = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                if random.random() < prob:
                    current = new_set
                    current_score = new_score
            
            T *= cooling
            if T < 0.5:
                T = 0.5

    def polynomial_search():
        """
        Model A as p(0), p(1), ..., p(19) where p(t) = a0 + a1*t + a2*t^2 + ...
        For low-degree polynomials, |A+A| is bounded by O(deg^2 * n) but can be
        much smaller than n*(n+1)/2 if there are additive collisions.
        
        We'll use degree 1-3 polynomials with integer coefficients, then optimize
        the coefficients to minimize max(|A+A|, |A*A|).
        """
        nonlocal best_set, best_score
        
        def eval_poly(coeffs, t):
            """Evaluate polynomial at t"""
            result = 0
            for i, c in enumerate(coeffs):
                result += c * (t ** i)
            return result
        
        def coeffs_to_set(coeffs, max_val=10000):
            """Convert polynomial coefficients to a set of 20 distinct positive integers <= max_val"""
            A = []
            seen = set()
            for t in range(20):
                val = eval_poly(coeffs, t)
                if val < 1 or val > max_val or val in seen:
                    return None
                A.append(val)
                seen.add(val)
            return A
        
        def perturb_coeffs(coeffs, degree, max_coeff=50):
            """Create a perturbed version of coefficients"""
            new_coeffs = coeffs[:]
            for i in range(len(new_coeffs)):
                if random.random() < 0.5:
                    new_coeffs[i] += random.randint(-max_coeff, max_coeff)
            return new_coeffs
        
        # Try different degrees
        for degree in [1, 2, 3]:
            # Initialize with small coefficients
            base_coeffs = [random.randint(1, 100) for _ in range(degree + 1)]
            
            current_coeffs = base_coeffs[:]
            current_set = coeffs_to_set(current_coeffs)
            
            if current_set is None:
                # Try to find a valid starting point
                for _ in range(100):
                    current_coeffs = [random.randint(1, 200) for _ in range(degree + 1)]
                    current_set = coeffs_to_set(current_coeffs)
                    if current_set is not None:
                        break
            
            if current_set is None:
                continue
            
            current_score = score(current_set)
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
            
            T = 10.0
            cooling = 0.998
            stuck_count = 0
            max_stuck = 50
            
            while time.time() - start_time < timeout:
                # Perturb coefficients
                new_coeffs = perturb_coeffs(current_coeffs, degree, max_coeff=20)
                new_set = coeffs_to_set(new_coeffs)
                
                if new_set is None:
                    stuck_count += 1
                    if stuck_count > max_stuck:
                        # Try a larger perturbation
                        new_coeffs = [random.randint(1, 500) for _ in range(degree + 1)]
                        new_set = coeffs_to_set(new_coeffs)
                        if new_set is None:
                            stuck_count += 1
                            if stuck_count > max_stuck * 2:
                                break
                    continue
                
                new_score = score(new_set)
                delta = new_score - current_score
                
                if delta < 0:
                    current_coeffs = new_coeffs
                    current_set = new_set
                    current_score = new_score
                    stuck_count = 0
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_set[:]
                else:
                    if T > 0:
                        prob = math.exp(-delta / T)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        current_coeffs = new_coeffs
                        current_set = new_set
                        current_score = new_score
                        stuck_count = 0
                    else:
                        stuck_count += 1
                
                T *= cooling
                if T < 1.0:
                    T = 1.0
                
                if stuck_count > max_stuck * 3:
                    break
        
        return best_set

    # Initial Candidates
    candidates = []
    
    # 1. Arithmetic Progression
    candidates.append(list(range(1, 21)))
    
    # 2. Odd Arithmetic Progression
    candidates.append(list(range(1, 41, 2)))
    
    # 3. First 20 Smooth Numbers
    candidates.append(pool[:20])
    
    # 4. Random Smooth Numbers
    for _ in range(3):
        random_set = random.sample(pool, 20)
        candidates.append(random_set)
        
    # 5. Random Uniform
    for _ in range(3):
        random_set = random.sample(range(1, 10001), 20)
        candidates.append(random_set)

    # Run SA on each candidate, but stop if time is up
    for cand in candidates:
        if time.time() - start_time > timeout:
            break
        sa_search(cand)
    
    # Run polynomial search if time remains
    if time.time() - start_time < timeout - 5:
        polynomial_search()
    
    return best_set
