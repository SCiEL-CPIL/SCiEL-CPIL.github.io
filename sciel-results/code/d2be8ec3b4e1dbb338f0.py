import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    all_primes = [p for p in range(2, 1001) if is_prime(p)]
    initial_primes = [p for p in all_primes if p >= 13]
    MAX_CHECK = 100000
    
    def compute_L(residues, primes):
        covered = bytearray(MAX_CHECK + 1)
        for i in range(len(primes)):
            p = primes[i]
            r = residues[i]
            start = p if r == 0 else r
            k = (MAX_CHECK - start) // p + 1
            if k > 0:
                covered[start::p] = b'\x01' * k
        L = 0
        for n in range(1, MAX_CHECK + 1):
            if covered[n]:
                L = n
            else:
                break
        return L
    
    def optimize_residues(primes, max_iters, time_limit):
        n = len(primes)
        residues = [(i + 1) % primes[i] for i in range(n)]
        best_L = compute_L(residues, primes)
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                p = primes[i]
                old_r = residues[i]
                best_r = old_r
                best_local = best_L
                cands = [old_r, 0]
                for t in range(1, min(20, p + 1)):
                    cands.append(t % p)
                for _ in range(3):
                    cands.append(random.randint(0, p - 1))
                for r in cands:
                    if r == old_r:
                        continue
                    residues[i] = r
                    L = compute_L(residues, primes)
                    if L > best_local:
                        best_local = L
                        best_r = r
                    residues[i] = old_r
                if best_r != old_r:
                    residues[i] = best_r
                    if best_local > best_L:
                        best_L = best_local
                        improved = True
            if not improved:
                break
        return residues, best_L
    
    current_primes = initial_primes[:]
    current_residues, current_L = optimize_residues(current_primes, 10, deadline - 30)
    
    used = set(current_primes)
    unused = sorted([p for p in all_primes if p not in used])
    budget = sum(Fraction(1, p) for p in current_primes)
    
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        best_improve = 0
        best_idx = -1
        best_p = -1
        order = sorted(range(len(current_primes)), key=lambda i: current_primes[i], reverse=True)
        for idx in order[:2]:
            p_old = current_primes[idx]
            for p_new in unused[:3]:
                if p_new >= p_old:
                    break
                nb = budget - Fraction(1, p_old) + Fraction(1, p_new)
                if nb > Fraction(1, 1):
                    continue
                tp = current_primes[:]
                tp[idx] = p_new
                _, tL = optimize_residues(tp, 3, deadline - 15)
                imp = tL - current_L
                if imp > best_improve:
                    best_improve = imp
                    best_idx = idx
                    best_p = p_new
        if best_idx >= 0:
            p_old = current_primes[best_idx]
            budget = budget - Fraction(1, p_old) + Fraction(1, best_p)
            current_primes[best_idx] = best_p
            used.remove(p_old)
            used.add(best_p)
            if best_p in unused:
                unused.remove(best_p)
            current_residues, current_L = optimize_residues(current_primes, 5, deadline - 5)
        else:
            break
    
    current_residues, current_L = optimize_residues(current_primes, 5, deadline - 2)
    return [[current_primes[i], current_residues[i]] for i in range(len(current_primes))]
