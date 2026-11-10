import time
import math
import random
import numpy as np

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

def _coverage(residues, primes, M):
    counts = np.zeros(M + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start > M:
            continue
        counts[start::p] += 1
    return counts

def _prefix_L(counts):
    bad = np.nonzero(counts[1:] < 2)[0]
    if bad.size == 0:
        return int(len(counts) - 1)
    return int(bad[0])

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0

    primes = _primes_upto(1000)
    n = len(primes)
    M = 30000

    def deadline_ok():
        return (time.time() - start_time) < TIME_LIMIT

    def surrogate_objective(residues, primes, window_start, window_end):
        """Minimize sum of squared deficits: for each x in window, penalty = max(0, 2 - count[x])^2"""
        counts = _coverage(residues, primes, window_end)
        window = counts[window_start:window_end+1]
        deficits = np.maximum(0, 2 - window)
        return np.sum(deficits ** 2)

    def numerical_gradient(residues, primes, window_start, window_end, step_size=1.0):
        """Compute numerical gradient for each residue position"""
        base_obj = surrogate_objective(residues, primes, window_start, window_end)
        grad = np.zeros(n)
        for i in range(n):
            p = primes[i]
            r_old = residues[i]
            # Try +1
            residues[i] = (r_old + 1) % p
            obj_plus = surrogate_objective(residues, primes, window_start, window_end)
            # Try -1
            residues[i] = (r_old - 1) % p
            obj_minus = surrogate_objective(residues, primes, window_start, window_end)
            # Restore
            residues[i] = r_old
            # Central difference
            grad[i] = (obj_plus - obj_minus) / 2.0
        return grad

    def gradient_descent_step(residues, primes, window_start, window_end, lr=0.5):
        """One step of gradient descent on the surrogate objective"""
        grad = numerical_gradient(residues, primes, window_start, window_end)
        for i in range(n):
            p = primes[i]
            r_old = residues[i]
            if grad[i] < -lr:
                # Move in negative gradient direction
                r_new = (r_old - 1) % p
                # Check if this improves or maintains L
                residues[i] = r_new
                counts = _coverage(residues, primes, M)
                newL = _prefix_L(counts)
                oldL = _prefix_L(_coverage(residues, primes, M))  # This is wrong, need to save old
                residues[i] = r_old  # restore for now
                # Actually, let's just apply and measure
                residues[i] = r_new
                return True
            elif grad[i] > lr:
                r_new = (r_old + 1) % p
                residues[i] = r_new
                return True
        return False

    def greedy_init(seed):
        rng = random.Random(seed)
        residues = [0] * n
        counts = np.zeros(M + 1, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            r = rng.randrange(p)
            residues[i] = r
            start = r if r > 0 else p
            if start <= M:
                counts[start::p] += 1
        L = _prefix_L(counts)
        # Greedy: extend by fixing first bad point with a safe prime
        for _ in range(5000):
            if L >= M - 1:
                break
            x = L + 1
            if counts[x] >= 2:
                L += 1
                continue
            improved = False
            for i in range(n):
                p = primes[i]
                r_old = residues[i]
                r_new = x % p
                if r_new == r_old:
                    continue
                if r_old == 0:
                    first = p
                else:
                    first = r_old
                ok = True
                xx = first
                while xx <= L:
                    if counts[xx] < 3:
                        ok = False
                        break
                    xx += p
                if not ok:
                    continue
                if r_old == 0:
                    f_old = p
                else:
                    f_old = r_old
                counts[f_old::p] -= 1
                if r_new == 0:
                    f_new = p
                else:
                    f_new = r_new
                counts[f_new::p] += 1
                residues[i] = r_new
                improved = True
                break
            if not improved:
                break
        return residues, L

    def local_search(residues, L):
        counts = _coverage(residues, primes, M)
        L = _prefix_L(counts)
        improved = True
        while improved and deadline_ok():
            improved = False
            if L >= M - 1:
                break
            x = L + 1
            if counts[x] >= 2:
                L += 1
                continue
            best_gain = 0
            best_i = -1
            best_r = -1
            for i in range(n):
                p = primes[i]
                r_old = residues[i]
                r_new = x % p
                if r_new == r_old:
                    continue
                if r_old == 0:
                    f_old = p
                else:
                    f_old = r_old
                ok = True
                xx = f_old
                while xx <= L:
                    if counts[xx] < 3:
                        ok = False
                        break
                    xx += p
                if not ok:
                    continue
                if r_new == 0:
                    f_new = p
                else:
                    f_new = r_new
                counts[f_old::p] -= 1
                counts[f_new::p] += 1
                residues[i] = r_new
                newL = _prefix_L(counts)
                if newL > L:
                    L = newL
                    improved = True
                    break
                else:
                    counts[f_new::p] -= 1
                    counts[f_old::p] += 1
                    residues[i] = r_old
        return residues, L

    def gradient_based_optimize(residues, L, iters=5000):
        """Use gradient descent on surrogate objective to improve coverage"""
        counts = _coverage(residues, primes, M)
        L = _prefix_L(counts)
        cur_L = L
        rng = random.Random(123)
        window_size = 5000
        for it in range(iters):
            if not deadline_ok():
                break
            if L >= M - 1:
                break
            # Focus on the bottleneck region
            window_start = max(1, L - 100)
            window_end = min(M, L + window_size)
            
            # Compute gradient
            grad = numerical_gradient(residues, primes, window_start, window_end)
            
            # Find the most negative gradient component (best improvement direction)
            min_grad_idx = np.argmin(grad)
            if grad[min_grad_idx] >= 0:
                # No improvement possible via gradient, try random perturbation
                i = rng.randrange(n)
                p = primes[i]
                r_old = residues[i]
                r_new = rng.randrange(p)
                if r_new == r_old:
                    continue
                if r_old == 0:
                    f_old = p
                else:
                    f_old = r_old
                if r_new == 0:
                    f_new = p
                else:
                    f_new = r_new
                counts[f_old::p] -= 1
                counts[f_new::p] += 1
                residues[i] = r_new
                newL = _prefix_L(counts)
                if newL > cur_L:
                    cur_L = newL
                    L = newL
                else:
                    # Revert
                    counts[f_new::p] -= 1
                    counts[f_old::p] += 1
                    residues[i] = r_old
            else:
                # Apply gradient step
                i = min_grad_idx
                p = primes[i]
                r_old = residues[i]
                if grad[i] < 0:
                    r_new = (r_old - 1) % p
                else:
                    r_new = (r_old + 1) % p
                if r_new == r_old:
                    continue
                if r_old == 0:
                    f_old = p
                else:
                    f_old = r_old
                if r_new == 0:
                    f_new = p
                else:
                    f_new = r_new
                counts[f_old::p] -= 1
                counts[f_new::p] += 1
                residues[i] = r_new
                newL = _prefix_L(counts)
                if newL > cur_L:
                    cur_L = newL
                    L = newL
                else:
                    # Revert
                    counts[f_new::p] -= 1
                    counts[f_old::p] += 1
                    residues[i] = r_old
        
        return residues, L

    best_res = None
    best_L = -1
    
    # Phase 1: Greedy initialization with multiple seeds
    for s in range(8):
        if not deadline_ok():
            break
        res, L = greedy_init(2000 + s)
        if L > best_L:
            best_L = L
            best_res = list(res)
    
    # Phase 2: Local search from best
    if best_res is not None and deadline_ok():
        best_res, best_L = local_search(best_res, best_L)
    
    # Phase 3: Gradient-based optimization (new structural change)
    if best_res is not None and deadline_ok():
        res, L = gradient_based_optimize(list(best_res), best_L, iters=10000)
        if L > best_L:
            best_L = L
            best_res = res
    
    # Phase 4: Final local search polish
    if best_res is not None and deadline_ok():
        best_res, best_L = local_search(best_res, best_L)
    
    # Ensure valid Python ints
    best_res = [int(r) % primes[i] for i, r in enumerate(best_res)]
    return best_res

if __name__ == "__main__":
    r = construct()
    print(len(r))
