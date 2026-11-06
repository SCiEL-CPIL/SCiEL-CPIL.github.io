import math
import time

def construct():
    T12 = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12! that are > 12
    all_divisors = []
    
    def generate_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                all_divisors.append(current_val)
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            generate_divisors(idx + 1, current_val * val)
            val *= p

    generate_divisors(0, 1)
    all_divisors.sort()
    
    # Map each divisor to its prime exponent vector
    divisor_to_exp = {}
    for d in all_divisors:
        temp = d
        exp_vec = []
        for p in primes:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            exp_vec.append(count)
        divisor_to_exp[d] = tuple(exp_vec)
        
    # Precompute list of (divisor, exp_vec) for faster access
    div_exp_list = [(d, divisor_to_exp[d]) for d in all_divisors]
    n_divs = len(div_exp_list)
    
    # Global best solution
    best_max = float('inf')
    best_solution = []
    
    # Greedy initialization to get a good upper bound
    # Try to pick smallest possible divisors
    rem_exp_greedy = list(exponents)
    used_greedy = set()
    current_max_greedy = 0
    sol_greedy = []
    
    # Simple greedy: always pick the smallest divisor that fits
    # This might fail if it gets stuck, but we can just try it
    # Actually, a better greedy is to pick divisors that reduce the "largeness"
    # But let's just run a quick DFS with a tight time limit or just the greedy
    # If greedy fails, best_max remains inf, and B&B will work without the d >= best_max prune initially
    # But we can try a few random or heuristic starts? No, let's just run B&B.
    # The first solution found will set best_max.
    
    # To ensure we find a solution quickly, we can try to construct one greedily
    # If we pick divisors in increasing order, we might end up with many small factors,
    # which minimizes the max.
    
    def greedy_solve():
        rem_exp = list(exponents)
        used = set()
        current_max = 0
        sol = []
        
        # We need to pick divisors such that their product is T12.
        # Greedy: pick smallest d > 12, not used, d divides remaining product.
        # "d divides remaining product" means exp_vec(d) <= rem_exp.
        
        while any(e > 0 for e in rem_exp):
            found = False
            for d, ev in div_exp_list:
                if d in used:
                    continue
                # Check if d fits in rem_exp
                fits = True
                for j in range(5):
                    if rem_exp[j] < ev[j]:
                        fits = False
                        break
                if fits:
                    # Use d
                    used.add(d)
                    for j in range(5):
                        rem_exp[j] -= ev[j]
                    sol.append(d)
                    current_max = max(current_max, d)
                    found = True
                    break
            if not found:
                return None, 0 # Greedy failed
        
        return sol, current_max

    sol_g, max_g = greedy_solve()
    if sol_g is not None:
        best_max = max_g
        best_solution = sol_g[:]

    def dfs(rem_exp, used, current_max, depth):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
            
        if all(e == 0 for e in rem_exp):
            if current_max < best_max:
                best_max = current_max
                best_solution = list(used)
            return
            
        # If we have used 12 factors, we can't add more
        if depth >= 12:
            return
            
        # Iterate through divisors
        for d, ev in div_exp_list:
            # Pruning 1: Distinctness
            if d in used:
                continue
                
            # Pruning 2: Minimize Max
            # If d >= best_max, any solution extending this will have max >= d >= best_max
            # So it cannot be strictly better.
            if d >= best_max:
                break # Since div_exp_list is sorted by d
                
            # Pruning 3: Exponent feasibility
            fits = True
            new_rem_exp = [0]*5
            for j in range(5):
                if rem_exp[j] < ev[j]:
                    fits = False
                    break
                new_rem_exp[j] = rem_exp[j] - ev[j]
                
            if not fits:
                continue
                
            # Additional Pruning: 
            # The remaining product must be 1.
            # The product of remaining factors must be exactly the remaining number.
            # We don't have an easy check for "can remaining number be factored into distinct divisors > 12 not in used".
            # But we can check if the remaining number is 1.
            
            # Recurse
            dfs(tuple(new_rem_exp), used | {d}, max(current_max, d), depth + 1)
            
            # If time is up, return
            if time.time() > deadline:
                return

    dfs(tuple(exponents), set(), 0, 0)
    
    if best_solution:
        return best_solution
    else:
        # Fallback: return a valid solution if found by greedy, or just T12?
        # T12 is not > 12? 12! is huge, so it is > 12.
        # But it must be distinct integers. A single integer is distinct from others (vacuously).
        # "1..12 DISTINCT integers" usually implies a set of size 1 to 12.
        # So [T12] is a valid solution of size 1.
        # But we want to minimize the largest. T12 is the largest possible.
        # If DFS failed, return greedy result if it exists, else [T12].
        if sol_g:
            return sol_g
        return [T12]
