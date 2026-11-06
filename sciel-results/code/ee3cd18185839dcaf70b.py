import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 140
    max_val = 10**18

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    nprimes = len(primes)

    def feasible(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
                if prod > max_val:
                    return False
        return True

    def score_from_exps(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
                if prod > max_val:
                    return -1
        
        # Generate divisors with bitmask of which primes divide them
        divs = [(1, 0)]
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            mask_bit = 1 << i
            cur_len = len(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for j in range(cur_len):
                    divs.append((divs[j][0] * factor, divs[j][1] | mask_bit))
        
        divs.sort(key=lambda x: x[0])
        m = len(divs)
        if m <= 1:
            return 0
        
        # Use numpy for fast coprime check
        masks = np.array([d[1] for d in divs], dtype=np.uint32)
        # Adjacent pairs are coprime if their masks have no common bits
        adj_and = masks[:-1] & masks[1:]
        score = int(np.sum(adj_and == 0))
        return score

    def get_n(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
        return prod

    def local_search(exps, iters=5000, use_annealing=True):
        cur_exps = list(exps)
        cur_score = score_from_exps(cur_exps)
        if cur_score < 0:
            return -1, 2, cur_exps
        
        best_exps = cur_exps[:]
        best_score = cur_score
        
        # Simulated annealing parameters
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        rng_local = random.Random(12345)
        
        for it in range(iters):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            improved = False
            
            # Try increasing/decreasing each exponent
            for i in range(nprimes):
                # Increase
                if feasible(cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]):
                    new = cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]
                    s = score_from_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score = new, s
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_exps = cur_exps[:]
                # Decrease
                if cur_exps[i] > 0:
                    new = cur_exps[:i] + [cur_exps[i] - 1] + cur_exps[i + 1:]
                    s = score_from_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score = new, s
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_exps = cur_exps[:]
            
            # Try swap moves: replace prime i with prime j (j > i) where exps[j] == 0
            for i in range(nprimes):
                if cur_exps[i] == 0:
                    continue
                for j in range(i + 1, nprimes):
                    if cur_exps[j] > 0:
                        continue
                    new = cur_exps[:]
                    new[i] = 0
                    new[j] = cur_exps[i]
                    if feasible(new):
                        s = score_from_exps(new)
                        if s > cur_score:
                            cur_exps, cur_score = new, s
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_exps = cur_exps[:]
            
            # 2-step moves: change two exponents at once
            if use_annealing and it % 3 == 0:
                for i in range(nprimes):
                    for j in range(i + 1, nprimes):
                        # Try (i+1, j-1)
                        if cur_exps[i] + 1 >= 0 and cur_exps[j] - 1 >= 0:
                            new = cur_exps[:]
                            new[i] += 1
                            new[j] -= 1
                            if feasible(new):
                                s = score_from_exps(new)
                                if s > cur_score:
                                    cur_exps, cur_score = new, s
                                    improved = True
                                    if cur_score > best_score:
                                        best_score = cur_score
                                        best_exps = cur_exps[:]
                        # Try (i-1, j+1)
                        if cur_exps[i] - 1 >= 0 and cur_exps[j] + 1 >= 0:
                            new = cur_exps[:]
                            new[i] -= 1
                            new[j] += 1
                            if feasible(new):
                                s = score_from_exps(new)
                                if s > cur_score:
                                    cur_exps, cur_score = new, s
                                    improved = True
                                    if cur_score > best_score:
                                        best_score = cur_score
                                        best_exps = cur_exps[:]
            
            # Simulated annealing: accept worse moves with probability
            if use_annealing and T > T_min:
                # Random perturbation
                i = rng_local.randint(0, nprimes - 1)
                delta = rng_local.choice([-1, 1, -2, 2])
                new = cur_exps[:]
                new[i] = max(0, new[i] + delta)
                if new != cur_exps and feasible(new):
                    s = score_from_exps(new)
                    if s >= cur_score or rng_local.random() < math.exp((s - cur_score) / T):
                        cur_exps = new
                        cur_score = s
                        if cur_score > best_score:
                            best_score = cur_score
                            best_exps = cur_exps[:]
                T *= cooling
            
            if not improved and not use_annealing:
                break
        
        return best_score, get_n(best_exps), best_exps

    best_score = -1
    best_n = 2
    best_exps = [0] * nprimes

    # Phase 1: Generate strong seeds
    seeds = []
    
    # 1. Product of first k primes
    for k in range(1, 16):
        exps = [0] * nprimes
        for i in range(k):
            exps[i] = 1
        if feasible(exps):
            seeds.append(exps)
            
    # 2. Variations of N_15 with extra powers of 2
    base_15 = [1]*15 + [0]
    for extra_2 in range(1, 5):
        exps = base_15[:]
        exps[0] += extra_2
        if feasible(exps):
            seeds.append(exps)
            
    # 3. N_14 with powers of 2 and 3
    base_14 = [1]*14 + [0, 0]
    for e2 in range(1, 10):
        for e3 in range(0, 10):
            exps = base_14[:]
            exps[0] += e2
            exps[1] += e3
            if feasible(exps):
                seeds.append(exps)
                
    # 4. Randomized square-free subsets of size 13-15
    rng = random.Random(42)
    for _ in range(80):
        exps = [0] * nprimes
        indices = list(range(16))
        rng.shuffle(indices)
        size = rng.randint(13, 15)
        for i in range(size):
            exps[indices[i]] = 1
        if feasible(exps):
            seeds.append(exps)
    
    # 5. Seeds with mixed exponents (some primes have exponent 2)
    for base_size in [13, 14, 15]:
        for _ in range(30):
            exps = [0] * nprimes
            indices = list(range(16))
            rng.shuffle(indices)
            for i in range(base_size):
                exps[indices[i]] = 1
            # Add extra power to a random prime
            if base_size < 16:
                extra_idx = rng.choice([i for i in range(16) if exps[i] == 0])
                exps[extra_idx] = rng.randint(1, 3)
            if feasible(exps):
                seeds.append(exps)

    # 6. More diverse seeds: non-consecutive prime selections
    for _ in range(50):
        exps = [0] * nprimes
        indices = list(range(16))
        rng.shuffle(indices)
        size = rng.randint(12, 15)
        for i in range(size):
            exps[indices[i]] = 1
        # Randomly add some extra powers
        for _ in range(rng.randint(0, 3)):
            idx = rng.randint(0, 15)
            if exps[idx] == 0:
                exps[idx] = rng.randint(1, 2)
            else:
                exps[idx] += rng.randint(1, 2)
        if feasible(exps):
            seeds.append(exps)

    # Evaluate seeds and run local search on the best ones
    seed_scores = []
    for s in seeds:
        sc = score_from_exps(s)
        if sc > 0:
            seed_scores.append((sc, s))
            
    # Sort by score descending
    seed_scores.sort(reverse=True)
    
    # Run local search on top 40 seeds with more iterations
    for sc, exps in seed_scores[:40]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        final_score, final_n, final_exps = local_search(exps, iters=4000, use_annealing=True)
        if final_score > best_score:
            best_score = final_score
            best_n = final_n
            best_exps = final_exps

    # Final polish: run local search on the best found with even more iterations
    if time.time() - start_time < TIME_LIMIT - 20:
        final_score, final_n, final_exps = local_search(best_exps, iters=6000, use_annealing=True)
        if final_score > best_score:
            best_score = final_score
            best_n = final_n
            best_exps = final_exps

    # Final check on the best found
    if best_n > 10**18 or best_n < 2:
        return 2
        
    return best_n
