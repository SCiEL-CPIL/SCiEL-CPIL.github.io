def construct():
    import random
    import time

    def compute_sizes(A):
        sums = set()
        prods = set()
        for i, a in enumerate(A):
            for j, b in enumerate(A):
                sums.add(a + b)
                prods.add(a * b)
        return len(sums), len(prods)
    
    def score(A):
        s, p = compute_sizes(A)
        return max(s, p)

    def generate_lattice_seed(primes, count, max_val):
        all_nums = set()
        def gen(idx, current):
            if current > max_val:
                return
            if idx == len(primes):
                all_nums.add(current)
                return
            p = primes[idx]
            val = current
            while val <= max_val:
                gen(idx + 1, val)
                val *= p
        gen(0, 1)
        sorted_nums = sorted(all_nums)
        return sorted_nums[:count]

    def local_search(A, max_time=7.0):
        best_A = A[:]
        best_score = score(best_A)
        start_time = time.time()
        rng = random.Random(123)
        
        while time.time() - start_time < max_time:
            improved = False
            # Try single swaps
            for i in range(20):
                old_val = best_A[i]
                for _ in range(100):
                    new_val = rng.randint(1, 10000)
                    if new_val == old_val or new_val in best_A:
                        continue
                    best_A[i] = new_val
                    s_val = score(best_A)
                    if s_val < best_score:
                        best_score = s_val
                        improved = True
                    else:
                        best_A[i] = old_val
                if improved:
                    break
            if not improved:
                # Perturbation
                for _ in range(5):
                    idxs = rng.sample(range(20), 3)
                    old_vals = [best_A[i] for i in idxs]
                    for i in idxs:
                        best_A[i] = rng.randint(1, 10000)
                    if len(set(best_A)) < 20:
                        for k, i in enumerate(idxs):
                            best_A[i] = old_vals[k]
                        continue
                    s_val = score(best_A)
                    if s_val < best_score:
                        best_score = s_val
                    else:
                        for k, i in enumerate(idxs):
                            best_A[i] = old_vals[k]
        return best_A, best_score

    seeds = []
    seeds.append(generate_lattice_seed([2, 3], 20, 10000))
    seeds.append(generate_lattice_seed([2, 5], 20, 10000))
    seeds.append(generate_lattice_seed([3, 5], 20, 10000))
    seeds.append(generate_lattice_seed([2, 3, 5], 20, 10000))
    
    # Geometric Progression
    gp = [1]
    while len(gp) < 20 and gp[-1] * 1.1 <= 10000:
        gp.append(int(gp[-1] * 1.1) + 1)
    gp = list(set(gp))
    if len(gp) < 20:
        for i in range(1, 10001):
            if i not in gp:
                gp.append(i)
            if len(gp) == 20:
                break
    seeds.append(sorted(gp[:20]))
    
    # Small integers
    seeds.append(list(range(1, 21)))
    
    # Random
    random.seed(42)
    seeds.append(sorted(random.sample(range(1, 10001), 20)))

    best_overall_A = None
    best_overall_score = float('inf')
    
    for seed in seeds:
        if len(seed) != 20 or len(set(seed)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in seed):
            continue
        A, sc = local_search(seed)
        if sc < best_overall_score:
            best_overall_score = sc
            best_overall_A = A[:]
            
    if best_overall_A is None:
        return list(range(1, 21))
        
    return sorted(best_overall_A)
