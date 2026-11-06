import time
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill

    N = 511
    MAX_A = 512

    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(MAX_A)]
    for p in range(2, MAX_A):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(math.isqrt(p)) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            for m in range(p, MAX_A, p):
                prime_factors[m].add(p)

    # Weight for each a
    weight = [0.0] * MAX_A
    for a in range(1, MAX_A):
        weight[a] = 1.0 / (MAX_A - a)

    # Greedy: pick highest weight first (i.e., largest a) that is coprime to all chosen
    def greedy_select():
        chosen = []
        used_primes = set()
        for a in range(MAX_A - 1, 0, -1):
            if a == 1:
                # 1 is coprime to everything, weight = 1/511
                chosen.append(1)
                continue
            # Check if a's prime factors conflict with used_primes
            if prime_factors[a] & used_primes:
                continue
            chosen.append(a)
            used_primes |= prime_factors[a]
        return chosen

    best = greedy_select()
    best_score = sum(weight[a] for a in best)

    def score_of(arr):
        return sum(weight[a] for a in arr)

    def is_valid(arr):
        used = set()
        for a in arr:
            if a == 1:
                continue
            if prime_factors[a] & used:
                return False
            used |= prime_factors[a]
        return True

    # Local search: try swaps
    def local_search(current):
        current = list(current)
        cur_score = score_of(current)
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            n = len(current)
            # Try swapping each selected with each unselected
            selected_set = set(current)
            unselected = [a for a in range(1, MAX_A) if a not in selected_set]
            for i in range(n):
                for a in unselected:
                    # Try swap: remove current[i], add a
                    trial = current[:i] + current[i+1:] + [a]
                    if not is_valid(trial):
                        continue
                    new_score = cur_score - weight[current[i]] + weight[a]
                    if new_score > cur_score + 1e-15:
                        current = trial
                        cur_score = new_score
                        improved = True
                        selected_set = set(current)
                        unselected = [a for a in range(1, MAX_A) if a not in selected_set]
                        break
                if improved:
                    break
            # Also try adding an unselected element if it doesn't conflict
            if not improved and (time.time() - start_time) < deadline:
                selected_set = set(current)
                used = set()
                for a in current:
                    if a != 1:
                        used |= prime_factors[a]
                for a in range(MAX_A - 1, 0, -1):
                    if a in selected_set:
                        continue
                    if a == 1:
                        new_score = cur_score + weight[1]
                        if new_score > cur_score + 1e-15:
                            current.append(1)
                            cur_score = new_score
                            improved = True
                        break
                    if not (prime_factors[a] & used):
                        new_score = cur_score + weight[a]
                        if new_score > cur_score + 1e-15:
                            current.append(a)
                            cur_score = new_score
                            improved = True
                            break
            # Try removing an element if it allows adding a better one
            if not improved and (time.time() - start_time) < deadline:
                for i in range(len(current)):
                    removed = current[i]
                    trial = current[:i] + current[i+1:]
                    trial_score = score_of(trial)
                    # Now try to add the best available
                    used = set()
                    for a in trial:
                        if a != 1:
                            used |= prime_factors[a]
                    best_add = None
                    best_add_gain = 0
                    for a in range(MAX_A - 1, 0, -1):
                        if a in trial:
                            continue
                        if a == 1:
                            gain = weight[1]
                            if gain > best_add_gain:
                                best_add_gain = gain
                                best_add = 1
                            break
                        if not (prime_factors[a] & used):
                            gain = weight[a]
                            if gain > best_add_gain:
                                best_add_gain = gain
                                best_add = a
                    if best_add is not None and trial_score + best_add_gain > cur_score + 1e-15:
                        current = trial + [best_add]
                        cur_score = trial_score + best_add_gain
                        improved = True
                        break
        return current, cur_score

    best, best_score = local_search(best)

    # Check time
    if (time.time() - start_time) < deadline:
        # Try a few more restarts with perturbation
        import random
        random.seed(42)
        for restart in range(20):
            if (time.time() - start_time) > deadline - 5:
                break
            # Perturb: remove a random element from best, then re-greedy
            trial = list(best)
            if len(trial) > 1:
                idx = random.randrange(len(trial))
                trial.pop(idx)
            # Re-greedy from trial
            used = set()
            for a in trial:
                if a != 1:
                    used |= prime_factors[a]
            # Add best available
            for a in range(MAX_A - 1, 0, -1):
                if a in trial:
                    continue
                if a == 1:
                    trial.append(1)
                    break
                if not (prime_factors[a] & used):
                    trial.append(a)
                    used |= prime_factors[a]
                    break
            trial, trial_score = local_search(trial)
            if trial_score > best_score + 1e-15:
                best = trial
                best_score = trial_score

    return best
