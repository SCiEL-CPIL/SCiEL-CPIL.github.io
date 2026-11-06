import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Primes 1 mod 4 for building M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421]
    
    def generate_reps(primes):
        """
        Generates all representations of M = product(primes) as sum of two squares.
        primes: list of distinct primes p = 1 mod 4.
        Returns list of (x, y) with x>0, y>0, x<=y, x^2+y^2=M.
        """
        if not primes:
            return []
        
        base_reps = []
        for p in primes:
            found = False
            for u in range(1, int(math.isqrt(p)) + 1):
                v2 = p - u*u
                v = math.isqrt(v2)
                if v*v == v2 and v > 0:
                    base_reps.append((u, v))
                    found = True
                    break
            if not found:
                # Fallback, should not happen for p=1 mod 4
                base_reps.append((1, 1))
        
        reps = [(1, 0)]
        for (u, v) in base_reps:
            new_reps = []
            for (x, y) in reps:
                # Multiply (x+iy) by (u+iv)
                x1 = x*u - y*v
                y1 = x*v + y*u
                new_reps.append((x1, y1))
                
                # Multiply (x+iy) by (u-iv)
                x2 = x*u + y*v
                y2 = -x*v + y*u
                new_reps.append((x2, y2))
            reps = new_reps
        
        pairs = []
        seen = set()
        for (x, y) in reps:
            if x < 0: x = -x
            if y < 0: y = -y
            if x == 0 or y == 0: continue
            if x > y:
                x, y = y, x
            if (x, y) not in seen:
                seen.add((x, y))
                pairs.append((x, y))
        
        return pairs

    best_N = 0
    best_pairs = []
    best_count = 0

    # Strategy 1: Q=1, maximize number of distinct prime factors in M
    # This yields 2^(k-1) pairs.
    # We find the max k such that product of first k primes 1 mod 4 <= 10^18.
    M1 = 1
    used_primes = []
    for p in p_1mod4:
        if M1 * p > 10**18:
            break
        M1 *= p
        used_primes.append(p)
    
    if M1 > 1:
        pairs_xy = generate_reps(used_primes)
        final_pairs = []
        for (x, y) in pairs_xy:
            a = x * x
            b = y * y
            if a <= b:
                final_pairs.append([a, b])
            else:
                final_pairs.append([b, a])
        
        unique_pairs = []
        seen_pairs = set()
        for p in final_pairs:
            if tuple(p) not in seen_pairs:
                seen_pairs.add(tuple(p))
                unique_pairs.append(p)
                
        if len(unique_pairs) > best_count:
            best_count = len(unique_pairs)
            best_N = M1
            best_pairs = unique_pairs

    # Strategy 2: Try Q > 1 to see if we can get more pairs via exponents?
    # Actually, Q=1 is generally optimal for distinct primes.
    # But let's double check if using a small Q allows a different structure?
    # No, Q=1 gives max M.
    
    # However, what if we can include a prime with exponent > 1 in M?
    # If M = p^2 * q * r ...
    # Number of reps is proportional to (e_p+1) * (e_q+1) ...
    # For distinct primes, it's 2^k.
    # If we use p^2, we replace a prime q with p^2?
    # Size: p^2 vs q.
    # If p^2 < q, we can fit it.
    # But does it increase the count?
    # Replacing q (factor 2) with p^2 (factor 3) increases count by 1.5x.
    # But we lose the prime q.
    # Let's see if we can swap a large prime for a smaller prime squared.
    
    # Current M1 uses primes: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89
    # Product ~ 10^17.
    # Can we replace 89 with 5^2=25?
    # New M = M1 / 89 * 25.
    # 25 < 89, so M decreases. We can add more primes!
    # Let's try to optimize M to maximize the product of (e_i+1).
    
    # We'll do a greedy or local search on the exponents.
    # Start with the set of primes from M1.
    # Try replacing the largest prime with the square of the smallest prime not already maxed?
    # Or just try all combinations of exponents for the first few primes?
    
    # Let's try a limited DFS on exponents for the first 10-12 primes.
    # But Q=1.
    
    # The current best is 256 pairs (2^8).
    # If we can get 3 factors instead of 2 for one prime, we get 3 * 2^7 = 384.
    # Can we fit p^2 * q1 * ... * q8?
    # Smallest p=5. 5^2=25.
    # Next smallest primes: 13, 17, 29, 37, 41, 53, 61, 73.
    # Product: 25 * 13 * 17 * 29 * 37 * 41 * 53 * 61 * 73
    # 25 * 13 = 325
    # 325 * 17 = 5525
    # 5525 * 29 = 160225
    # 160225 * 37 = 5928325
    # 5928325 * 41 = 243061325
    # 243061325 * 53 = 12882250225
    # 12882250225 * 61 = 785817263725
    # 785817263725 * 73 = 57364660251925
    # This is ~5.7 * 10^13.
    # We have room for more primes!
    # Next: 89.
    # 57364660251925 * 89 = 5105454762421325
    # Next: 97.
    # 5105454762421325 * 97 = 495229111954868525
    # Next: 101.
    # 495229111954868525 * 101 = 50018140307441721025
    # Next: 109.
    # 50018140307441721025 * 109 = 5451977293511147591725 > 10^18.
    # So we can fit 5^2 * 13 * 17 * 29 * 37 * 41 * 53 * 61 * 73 * 89 * 97 * 101.
    # That's 12 prime factors (one squared).
    # Number of reps proportional to (2+1) * 2^11 = 3 * 2048 = 6144.
    # Pairs = 6144 / 4 = 1536?
    # Let's check the formula.
    # r_2(M) = 4 * prod(e_i+1) if M has no 3 mod 4 factors.
    # Pairs (x,y) with x>0, y>0, x<=y is r_2(M)/4 if M is not a square and not 2*square?
    # Wait, for M=5, r_2=8, pairs=1. 8/4=2? No.
    # My previous analysis said pairs = 2^(k-1).
    # For k=1, 2^0=1. Correct.
    # For k=2, 2^1=2. Correct.
    # For k=12 with one exponent 2:
    # Prod(e_i+1) = 3 * 2^11.
    # r_2 = 4 * 3 * 2^11.
    # Pairs = r_2 / 8?
    # Let's re-verify.
    # Each pair {x,y} with x!=y, x!=0, y!=0 corresponds to 8 representations:
    # (x,y), (y,x), (-x,y), (x,-y), (-x,-y), (-y,x), (y,-x), (-y,-x).
    # So pairs = r_2 / 8.
    # For M=5, r_2=8. Pairs=1. 8/8=1. Correct.
    # For M=65, r_2=16. Pairs=2. 16/8=2. Correct.
    # So pairs = prod(e_i+1) / 2.
    # For distinct primes, prod=2^k. Pairs=2^(k-1).
    # For 5^2 * 11 distinct primes:
    # prod = 3 * 2^11 = 6144.
    # Pairs = 6144 / 2 = 3072.
    
    # This is much better than 256!
    # Let's try to maximize this.
    
    # We want to maximize prod(e_i+1) subject to prod(p_i^e_i) <= 10^18.
    # This is a knapsack-like problem.
    # Since we want to maximize the product, we should use small primes.
    # And we should increase exponents of the smallest primes first?
    # Let's try a greedy approach:
    # Start with all exponents 0.
    # At each step, choose the prime p and exponent increment e->e+1 that gives the best "gain per log size".
    # Gain: (e+2)/(e+1).
    # Cost: log(p).
    # Ratio: (e+2)/(e+1) / log(p).
    
    # Let's implement a simple greedy.
    
    max_N = 10**18
    exponents = {p: 0 for p in p_1mod4[:20]}
    current_M = 1
    
    # We need to be careful with the greedy.
    # Let's just try to fit as many factors as possible.
    # Start with 5^2.
    # Then add 13, 17, ...
    
    # Let's try to construct a candidate M2.
    # Start with 5^2.
    M2 = 25
    exps2 = {5: 2}
    for p in p_1mod4[1:]:
        if M2 * p <= max_N:
            M2 *= p
            exps2[p] = 1
        else:
            # Try increasing exponent of 5?
            # If M2 * 5 <= max_N, then M2 *= 5, exps2[5]+=1
            # But we already passed 5 in the loop.
            # Let's just break and try to optimize later.
            break
    
    # Now try to optimize exponents of small primes.
    # We have some room: max_N / M2.
    # Try to increase exponents of 5, 13, 17...
    
    # Actually, a better greedy:
    # While there is room, pick the best next step.
    
    def optimize_M(max_val):
        # Initialize with smallest primes
        # We'll use a list of (prime, exponent)
        # Start with 5^1, 13^1, etc.
        
        # Let's start with the set of primes from the first strategy, 
        # but allow exponents > 1.
        
        # Greedy:
        # State: current_M, exponents dict.
        # Goal: maximize prod(e+1).
        
        # Start with empty
        M = 1
        exps = {}
        
        # Precompute candidates: (prime, current_exp)
        # We will try to add to the smallest primes first.
        
        # List of primes to consider
        primes_to_consider = p_1mod4[:15]
        
        # We will do a best-first search or just a simple greedy.
        # Greedy:
        # At each step, for each prime p in primes_to_consider,
        # if M * p <= max_val, calculate new_score = prod(e_i+1) * (e_p+2)/(e_p+1)
        # Actually, we want to maximize the final product.
        # Greedy choice: maximize (e_p+2)/(e_p+1) / log(p).
        
        # Let's just iterate.
        
        # Start with 5^2?
        # Let's try to build M from scratch using greedy.
        
        # Initial M=1.
        # Best move: 5^1? Ratio 2/log(5)=0.57. 5^2? No, we add one at a time.
        # So we always add p^1 to existing M.
        
        # Let's use a priority queue or just re-evaluate.
        
        # Since the number of primes is small, we can just loop.
        
        # But note: increasing exponent of 5 is cheaper than adding 13.
        # Cost of 5: log(5)=1.6. Gain: 2/1=2 (first time), 3/2=1.5 (second), 4/3=1.33.
        # Cost of 13: log(13)=2.56. Gain: 2/1=2.
        # So 5 is better.
        
        # Let's run the greedy.
        
        # We will maintain a list of (prime, exp).
        # And a running product.
        
        # To avoid floating point issues, we can compare fractions.
        # But floats are fine for this scale.
        
        # Let's just do it.
        
        # Start with M=1.
        # We will try to add factors.
        
        # We will use a list of primes.
        # For each prime, we can increase its exponent.
        
        # Let's just try to maximize the count by trying all combinations of exponents
        # for the first 5-6 primes?
        # 5, 13, 17, 29, 37, 41.
        # Max exponent for 5: 5^12 ~ 10^8.
        # This space is too large for brute force.
        
        # Greedy is safer.
        
        # Greedy Algorithm:
        # 1. M = 1
        # 2. While True:
        #    best_gain = 0
        #    best_p = None
        #    for p in primes_to_consider:
        #       if M * p <= max_val:
        #          gain = (exps.get(p, 0) + 2) / (exps.get(p, 0) + 1) / math.log(p)
        #          if gain > best_gain:
        #             best_gain = gain
        #             best_p = p
        #    if best_p is None: break
        #    M *= best_p
        #    exps[best_p] = exps.get(best_p, 0) + 1
        
        # This greedy might get stuck in local optima?
        # But for this specific structure, it's likely optimal.
        
        M_greedy = 1
        exps_greedy = {}
        
        # To be safe, we'll run this for a limited number of steps or until no improvement.
        # Since we are limited by size, it will stop.
        
        # Let's use a slightly smarter greedy:
        # Consider the "marginal utility" of adding one more factor of p.
        
        while True:
            best_gain = 0
            best_p = None
            for p in primes_to_consider:
                if M_greedy * p > max_val:
                    continue
                current_exp = exps_greedy.get(p, 0)
                # The factor (e+1) becomes (e+2).
                # Multiplicative gain in score: (e+2)/(e+1).
                # Log cost: log(p).
                gain = (current_exp + 2) / (current_exp + 1) / math.log(p)
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            
            if best_p is None:
                break
            
            M_greedy *= best_p
            exps_greedy[best_p] = exps_greedy.get(best_p, 0) + 1
        
        return M_greedy, exps_greedy

    M_opt, exps_opt = optimize_M(10**18)
    
    if M_opt > 1:
        # Generate reps for M_opt
        # We need the list of primes with exponents.
        # My generate_reps assumes distinct primes.
        # I need to modify it to handle exponents.
        
        # If p^e, we have e+1 choices for the factor in Gaussian integers?
        # No, p = (u+iv)(u-iv).
        # p^e = (u+iv)^e (u-iv)^e.
        # We can choose k factors of (u+iv) and e-k factors of (u-iv).
        # There are e+1 choices.
        # So the total number of Gaussian integer factors is prod(e_i+1).
        # Each choice gives a representation.
        
        # Let's write a new generator for exponents.
        
        def generate_reps_with_exps(primes_exps):
            # primes_exps: list of (p, e)
            base_factors = []
            for p, e in primes_exps:
                # Find u, v for p
                found = False
                for u in range(1, int(math.isqrt(p)) + 1):
                    v2 = p - u*u
                    v = math.isqrt(v2)
                    if v*v == v2 and v > 0:
                        # We need to generate all (u+iv)^k (u-iv)^(e-k) for k=0..e
                        # This is equivalent to choosing k from 0 to e.
                        # The factor is (u+iv)^k (u-iv)^(e-k).
                        # Let's just store (u, v) and e.
                        base_factors.append((u, v, e))
                        found = True
                        break
                if not found:
                    base_factors.append((1, 1, e))
            
            reps = [(1, 0)]
            
            for (u, v, e) in base_factors:
                new_reps = []
                for (x, y) in reps:
                    # We need to multiply by (u+iv)^k (u-iv)^(e-k) for k=0..e
                    # Let z = u+iv.
                    # We want z^k * conj(z)^(e-k).
                    # Note z * conj(z) = p.
                    # So z^k * conj(z)^(e-k) = z^(2k-e) * p^((e-|2k-e|)/2)?
                    # No, simpler:
                    # Let A = u+iv, B = u-iv.
                    # We want A^k B^(e-k).
                    # We can compute this iteratively.
                    
                    # Start with 1.
                    # Multiply by A or B, e times.
                    # But we have to choose the sequence.
                    # Actually, since A and B commute, the result is independent of order.
                    # So we just need to compute A^k B^(e-k).
                    
                    # Let's compute A^j for j=0..e and B^j for j=0..e.
                    # Then for each k, result = A^k * B^(e-k).
                    
                    # Precompute powers of A and B for this prime?
                    # No, we can just do it in the loop.
                    
                    # To be efficient, we can precompute the list of factors for this prime.
                    # factors_for_p = [A^k * B^(e-k) for k in 0..e]
                    
                    # But we are inside the loop over primes.
                    # Let's compute the factors for this prime once.
                    
                    # Actually, the current loop structure multiplies existing reps by the new factors.
                    # So we should precompute the list of e+1 factors for this prime.
                    
                    pass # This logic is getting complex. Let's simplify.
                    
                # Let's restart the generation logic.
                pass
            
            return [] # Placeholder

        # Let's implement a simpler version.
        # Since we only have a few primes with e>1, we can expand them.
        # If p^e, treat it as e distinct primes p, p, ..., p?
        # No, that's not correct for Gaussian integers.
        
        # Let's stick to the distinct prime case if possible, or implement the general case carefully.
        
        # Given time, let's just use the distinct prime case with Q=1.
        # The greedy M_opt likely has exponents > 1.
        # If we can't generate reps easily, we fall back to M1.
        
        # Let's try to generate reps for M_opt.
        
        # Modified generate_reps:
        
        def generate_reps_general(primes_exps):
            # primes_exps: list of (p, e)
            base_factors = []
            for p, e in primes_exps:
                found = False
                for u in range(1, int(math.isqrt(p)) + 1):
                    v2 = p - u*u
                    v = math.isqrt(v2)
                    if v*v == v2 and v > 0:
                        base_factors.append((u, v, e))
                        found = True
                        break
                if not found:
                    base_factors.append((1, 1, e))
            
            reps = [(1, 0)]
            
            for (u, v, e) in base_factors:
                # Precompute the e+1 factors: F_k = (u+iv)^k (u-iv)^(e-k)
                factors = []
                # We can compute A^k and B^(e-k)
                A = (u, v)
                B = (u, -v)
                
                # Compute powers of A
                powA = [(1, 0)]
                for i in range(e):
                    x, y = powA[-1]
                    nx = x*u - y*v
                    ny = x*v + y*u
                    powA.append((nx, ny))
                
                # Compute powers of B
                powB = [(1, 0)]
                for i in range(e):
                    x, y = powB[-1]
                    nx = x*u - y*(-v) # x*u + y*v
                    ny = x*(-v) + y*u # -x*v + y*u
                    powB.append((nx, ny))
                
                for k in range(e + 1):
                    ax, ay = powA[k]
                    bx, by = powB[e-k]
                    # Multiply A^k * B^(e-k)
                    fx = ax*bx - ay*by
                    fy = ax*by + ay*bx
                    factors.append((fx, fy))
                
                new_reps = []
                for (x, y) in reps:
                    for (fx, fy) in factors:
                        nx = x*fx - y*fy
                        ny = x*fy + y*fx
                        new_reps.append((nx, ny))
                reps = new_reps
            
            pairs = []
            seen = set()
            for (x, y) in reps:
                if x < 0: x = -x
                if y < 0: y = -y
                if x == 0 or y == 0: continue
                if x > y:
                    x, y = y, x
                if (x, y) not in seen:
                    seen.add((x, y))
                    pairs.append((x, y))
            
            return pairs

        pairs_xy_opt = generate_reps_general(list(exps_opt.items()))
        
        final_pairs_opt = []
        for (x, y) in pairs_xy_opt:
            a = x * x
            b = y * y
            if a <= b:
                final_pairs_opt.append([a, b])
            else:
                final_pairs_opt.append([b, a])
        
        unique_pairs_opt = []
        seen_pairs_opt = set()
        for p in final_pairs_opt:
            if tuple(p) not in seen_pairs_opt:
                seen_pairs_opt.add(tuple(p))
                unique_pairs_opt.append(p)
                
        if len(unique_pairs_opt) > best_count:
            best_count = len(unique_pairs_opt)
            best_N = M_opt
            best_pairs = unique_pairs_opt

    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
