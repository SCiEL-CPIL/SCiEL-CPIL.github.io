<<<<<<< SEARCH
def construct():
    best_count = 23
    best_coeffs = [1]*12
    
    def count_nonzero(coeffs):
        n = len(coeffs)
        # Use numpy for faster convolution
        square = np.convolve(coeffs, coeffs)
        return np.count_nonzero(square)
        
    def count_nonzero_list(coeffs):
        # Fallback or for compatibility if numpy not preferred, but numpy is allowed
        n = len(coeffs)
        square = [0]*(2*n-1)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0: continue
            for j in range(n):
                cj = coeffs[j]
                if cj == 0: continue
                square[i+j] += ci*cj
        return sum(1 for x in square if x != 0)

    # 1. Simulated Annealing on general 12 coefficients
    # Start with random initialization
    random.seed(42)
    
    def simulated_annealing(initial_coeffs, steps=200000, start_temp=10.0, end_temp=0.01):
        current_coeffs = initial_coeffs[:]
        current_score = count_nonzero_list(current_coeffs)
        best_local_coeffs = current_coeffs[:]
        best_local_score = current_score
        
        for i in range(steps):
            # Cooling schedule
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            # Pick a random index and change value
            idx = random.randint(0, 11)
            old_val = current_coeffs[idx]
            # Propose new value: random in [-10, 10] or +/- 1 step
            # Let's try random in small range
            new_val = random.randint(-10, 10)
            if new_val == old_val:
                continue
            if new_val == 0:
                # We must maintain exactly 12 non-zero terms?
                # The task says "exactly 12 nonzero terms" in the INPUT polynomial.
                # So we cannot set any coefficient to 0.
                continue
            
            current_coeffs[idx] = new_val
            new_score = count_nonzero_list(current_coeffs)
            
            # Accept if better, or with probability exp(-(delta)/temp)
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_coeffs = current_coeffs[:]
            else:
                if random.random() < np.exp(-delta / temp):
                    current_score = new_score
                else:
                    current_coeffs[idx] = old_val
        
        return best_local_coeffs, best_local_score

    # Run SA a few times with different seeds/starts
    for run in range(5):
        random.seed(42 + run)
        init = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        # Ensure non-zero
        for i in range(12):
            if init[i] == 0: init[i] = 1
        b_c, b_s = simulated_annealing(init, steps=100000, start_temp=5.0, end_temp=0.01)
        if b_s < best_count:
            best_count = b_s
            best_coeffs = b_c[:]

    # 2. Search Anti-palindromic polynomials: a_i = -a_{11-i}
    # Variables: a0..a5. a6=-a5, a7=-a4, a8=-a3, a9=-a2, a10=-a1, a11=-a0
    # This ensures P(1) = 0, so (1-x) is a factor.
    def get_anti_palindromic(vars6):
        a = vars6[:]
        return a + [-a[5], -a[4], -a[3], -a[2], -a[1], -a[0]]

    def sa_anti_palindromic():
        best_local_score = 999
        best_local_vars = None
        current_vars = [random.randint(-5, 5) for _ in range(6)]
        # Ensure non-zero for the first 6? The mapped ones will be non-zero if vars are non-zero.
        for i in range(6):
            if current_vars[i] == 0: current_vars[i] = 1
        current_coeffs = get_anti_palindromic(current_vars)
        current_score = count_nonzero_list(current_coeffs)
        
        for i in range(50000):
            temp = 5.0 * (0.01 / 5.0) ** (i / 50000)
            idx = random.randint(0, 5)
            old_val = current_vars[idx]
            new_val = random.randint(-5, 5)
            if new_val == 0: continue
            if new_val == old_val: continue
            
            current_vars[idx] = new_val
            new_coeffs = get_anti_palindromic(current_vars)
            new_score = count_nonzero_list(new_coeffs)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_vars = current_vars[:]
            else:
                current_vars[idx] = old_val
        
        return get_anti_palindromic(best_local_vars), best_local_score

    for run in range(5):
        random.seed(100 + run)
        b_c, b_s = sa_anti_palindromic()
        if b_s < best_count:
            best_count = b_s
            best_coeffs = b_c[:]

    # 3. Search Palindromic polynomials: a_i = a_{11-i}
    def get_palindromic(vars6):
        a = vars6[:]
        return a + a[5::-1]

    def sa_palindromic():
        best_local_score = 999
        best_local_vars = None
        current_vars = [random.randint(-5, 5) for _ in range(6)]
        for i in range(6):
            if current_vars[i] == 0: current_vars[i] = 1
        current_coeffs = get_palindromic(current_vars)
        current_score = count_nonzero_list(current_coeffs)
        
        for i in range(50000):
            temp = 5.0 * (0.01 / 5.0) ** (i / 50000)
            idx = random.randint(0, 5)
            old_val = current_vars[idx]
            new_val = random.randint(-5, 5)
            if new_val == 0: continue
            if new_val == old_val: continue
            
            current_vars[idx] = new_val
            new_coeffs = get_palindromic(current_vars)
            new_score = count_nonzero_list(new_coeffs)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_vars = current_vars[:]
            else:
                current_vars[idx] = old_val
        
        return get_palindromic(best_local_vars), best_local_score

    for run in range(5):
        random.seed(200 + run)
        b_c, b_s = sa_palindromic()
        if b_s < best_count:
            best_count = b_s
            best_coeffs = b_c[:]

    # Final local search on the best found so far
    current = count_nonzero_list(best_coeffs)
    improved = True
    while improved:
        improved = False
        for i in range(12):
            for new_c in range(-10, 11):
                if new_c == 0 or new_c == best_coeffs[i]:
                    continue
                old = best_coeffs[i]
                best_coeffs[i] = new_c
                new_count = count_nonzero_list(best_coeffs)
                if new_count < current:
                    current = new_count
                    improved = True
                else:
                    best_coeffs[i] = old

    return best_coeffs
=======
def construct():
    best_count = 23
    best_coeffs = [1]*12
    
    def count_nonzero_square(A, B):
        max_deg = 0
        for d, c in A:
            if d > max_deg: max_deg = d
        for d, c in B:
            if d > max_deg: max_deg = d
            
        if max_deg > 40:
            return 999

        p_vec = [0] * (max_deg + 1)
        for d, c in A:
            p_vec[d] = c
        for d, c in B:
            p_vec[d] = c
            
        sq = np.convolve(p_vec, p_vec)
        return np.count_nonzero(sq)

    def is_valid(A, B):
        degs_A = [d for d, c in A]
        degs_B = [d for d, c in B]
        if len(set(degs_A + degs_B)) != 12:
            return False
        for d, c in A:
            if c == 0 or d > 40 or c < -64 or c > 64: return False
        for d, c in B:
            if c == 0 or d > 40 or c < -64 or c > 64: return False
        return True

    def get_coeffs(A, B):
        max_deg = 0
        for d, c in A:
            if d > max_deg: max_deg = d
        for d, c in B:
            if d > max_deg: max_deg = d
            
        p_vec = [0] * (max_deg + 1)
        for d, c in A:
            p_vec[d] = c
        for d, c in B:
            p_vec[d] = c
        return p_vec

    def simulated_annealing_AB_v2():
        all_degs = random.sample(range(0, 41), 12)
        degs_A = sorted(all_degs[:6])
        degs_B = sorted(all_degs[6:])
        
        coeffs_A = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        coeffs_B = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        
        A = list(zip(degs_A, coeffs_A))
        B = list(zip(degs_B, coeffs_B))
        
        current_score = count_nonzero_square(A, B)
        best_score = current_score
        best_A = A[:]
        best_B = B[:]
        
        steps = 30000
        start_temp = 10.0
        end_temp = 0.01
        
        for i in range(steps):
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            A_old = A[:]
            B_old = B[:]
            
            if random.random() < 0.5:
                idx = random.randint(0, 5)
                old_d, old_c = A[idx]
                if random.random() < 0.5:
                    new_d = old_d + random.randint(-3, 3)
                    new_c = old_c
                else:
                    new_d = old_d
                    new_c = random.choice([-2, -1, 1, 2])
                    
                A[idx] = (new_d, new_c)
            else:
                idx = random.randint(0, 5)
                old_d, old_c = B[idx]
                if random.random() < 0.5:
                    new_d = old_d + random.randint(-3, 3)
                    new_c = old_c
                else:
                    new_d = old_d
                    new_c = random.choice([-2, -1, 1, 2])
                    
                B[idx] = (new_d, new_c)
            
            if not is_valid(A, B):
                A = A_old
                B = B_old
                continue
            
            new_score = count_nonzero_square(A, B)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A[:]
                    best_B = B[:]
            else:
                A = A_old
                B = B_old

        return get_coeffs(best_A, best_B), best_score

    for run in range(10):
        random.seed(1000 + run)
        coeffs, score = simulated_annealing_AB_v2()
        if coeffs and score < best_count:
            best_count = score
            best_coeffs = coeffs

    current = 0
    sq = np.convolve(np.array(best_coeffs), np.array(best_coeffs))
    current = np.count_nonzero(sq)
    
    improved = True
    while improved:
        improved = False
        for i in range(len(best_coeffs)):
            for new_c in range(-10, 11):
                if new_c == 0 or new_c == best_coeffs[i]:
                    continue
                old = best_coeffs[i]
                best_coeffs[i] = new_c
                sq = np.convolve(np.array(best_coeffs), np.array(best_coeffs))
                new_count = np.count_nonzero(sq)
                if new_count < current:
                    current = new_count
                    improved = True
                else:
                    best_coeffs[i] = old

    return best_coeffs
>>>>>>> REPLACE
