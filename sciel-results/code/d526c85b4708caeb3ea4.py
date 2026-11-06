"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def is_prime_power_or_one(n):
        if n == 1:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                return m == 1
        return True
    
    # Precompute atomic sums (1 and prime powers) in [2, 256]
    atomic_sums = set()
    for n in range(2, 257):
        if is_prime_power_or_one(n):
            atomic_sums.add(n)
    
    # Precompute gcd compatibility for all possible sums
    # compat[s1][s2] = 1 if gcd(s1,s2)==1 else 0
    max_sum = 256
    compat = {}
    for s1 in range(2, max_sum + 1):
        compat[s1] = {}
        for s2 in range(2, max_sum + 1):
            if s1 != s2:
                compat[s1][s2] = 1 if math.gcd(s1, s2) == 1 else 0
            else:
                compat[s1][s2] = 1
    
    def evaluate_fast(A, B):
        """Fast evaluation using precomputed compatibility"""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = list(sums)
        for i in range(len(slist)):
            for j in range(i + 1, len(slist)):
                if compat[slist[i]][slist[j]] != 1:
                    return -1
        return len(slist)
    
    # Score each value in [1,128] by how many prime-power sums it can form
    all_vals = list(range(1, 129))
    val_score = {}
    for v in all_vals:
        count = 0
        for w in all_vals:
            if (v + w) in atomic_sums:
                count += 1
        val_score[v] = count
    
    sorted_vals = sorted(all_vals, key=lambda x: val_score[x], reverse=True)
    
    best_score = 0
    best_A = None
    best_B = None
    
    odd_vals = [v for v in range(1, 129, 2)]  # 1,3,5,...,127
    even_vals = [v for v in range(2, 129, 2)]  # 2,4,6,...,128
    
    # Score odd values by how many prime-power sums they form with even values
    odd_score = {}
    for v in odd_vals:
        count = 0
        for w in even_vals:
            if (v + w) in atomic_sums:
                count += 1
        odd_score[v] = count
    
    even_score = {}
    for v in even_vals:
        count = 0
        for w in odd_vals:
            if (v + w) in atomic_sums:
                count += 1
        even_score[v] = count
    
    sorted_odds = sorted(odd_vals, key=lambda x: odd_score[x], reverse=True)
    sorted_evens = sorted(even_vals, key=lambda x: even_score[x], reverse=True)
    
    random.seed(42)
    
    # Phase 1: Exhaustive search over top candidates
    # Try all combinations of top 15 odds and top 15 evens
    top_odds = sorted_odds[:15]
    top_evens = sorted_evens[:15]
    
    for A in combinations(top_odds, 3):
        if time.time() > deadline - 15:
            break
        A = list(A)
        for B in combinations(top_evens, 3):
            B = list(B)
            score = evaluate_fast(A, B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
                if best_score >= 9:
                    return [best_A, best_B]
    
    # Phase 2: Targeted local search from best found so far
    if best_A is not None:
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            A = best_A[:]
            B = best_B[:]
            score = best_score
            
            # Try multiple random restarts of local search
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                # Randomly perturb one element
                if random.random() < 0.5:
                    idx = random.randint(0, 2)
                    candidates = [v for v in sorted_odds[:40] if v not in A]
                    if not candidates:
                        continue
                    old_val = A[idx]
                    A[idx] = random.choice(candidates)
                    new_score = evaluate_fast(A, B)
                    if new_score > score:
                        score = new_score
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return [best_A, best_B]
                    else:
                        A[idx] = old_val
                else:
                    idx = random.randint(0, 2)
                    candidates = [v for v in sorted_evens[:40] if v not in B]
                    if not candidates:
                        continue
                    old_val = B[idx]
                    B[idx] = random.choice(candidates)
                    new_score = evaluate_fast(A, B)
                    if new_score > score:
                        score = new_score
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return [best_A, best_B]
                    else:
                        B[idx] = old_val
    
    # Phase 3: Broader random search with local improvement
    for _ in range(3000):
        if time.time() > deadline - 5:
            break
        
        # Random A from odds, B from evens
        A = random.sample(odd_vals, 3)
        B = random.sample(even_vals, 3)
        score = evaluate_fast(A, B)
        
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return [best_A, best_B]
        
        # Local search
        if score > 0:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try modifying A
                for idx in range(3):
                    for new_val in sorted_odds[:30]:
                        if new_val in A:
                            continue
                        old_val = A[idx]
                        A[idx] = new_val
                        new_score = evaluate_fast(A, B)
                        if new_score > score:
                            score = new_score
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_A = A[:]
                                best_B = B[:]
                                if best_score >= 9:
                                    return [best_A, best_B]
                            break
                        else:
                            A[idx] = old_val
                # Try modifying B
                for idx in range(3):
                    for new_val in sorted_evens[:30]:
                        if new_val in B:
                            continue
                        old_val = B[idx]
                        B[idx] = new_val
                        new_score = evaluate_fast(A, B)
                        if new_score > score:
                            score = new_score
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_A = A[:]
                                best_B = B[:]
                                if best_score >= 9:
                                    return [best_A, best_B]
                            break
                        else:
                            B[idx] = old_val
    
    # Phase 4: Try mixed parity (not all same parity) in case we can do better
    for _ in range(1500):
        if time.time() > deadline - 3:
            break
        A = random.sample(all_vals, 3)
        B = random.sample(all_vals, 3)
        score = evaluate_fast(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return [best_A, best_B]
        
        if score > 0:
            for _ in range(30):
                if time.time() > deadline - 3:
                    break
                idx = random.randint(0, 2)
                which = random.choice([0, 1])
                if which == 0:
                    old_val = A[idx]
                    new_val = random.choice(sorted_vals[:60])
                    if new_val in A:
                        continue
                    A[idx] = new_val
                    new_score = evaluate_fast(A, B)
                    if new_score > score:
                        score = new_score
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return [best_A, best_B]
                    else:
                        A[idx] = old_val
                else:
                    old_val = B[idx]
                    new_val = random.choice(sorted_vals[:60])
                    if new_val in B:
                        continue
                    B[idx] = new_val
                    new_score = evaluate_fast(A, B)
                    if new_score > score:
                        score = new_score
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return [best_A, best_B]
                    else:
                        B[idx] = old_val
    
    if best_A is not None and best_B is not None:
        return [best_A, best_B]
    else:
        # Fallback: known good configuration with 7 distinct coprime sums
        return [[1, 7, 13], [4, 6, 10]]
# EVOLVE-BLOCK-END
