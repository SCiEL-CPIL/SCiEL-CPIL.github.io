"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
    deadline = time.time() + 115.0

    # For the specific case q=3, n=6, radius=1, we aim for the known optimum of 73 (or better).
    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(deadline)

    # Generic fallback: direct sum construction or greedy
    if n <= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        else:
            pass

    return _greedy_set_cover(q, n, radius, deadline)

def _solve_ternary_6_1(deadline):
    q, n, radius = 3, 6, 1
    total = q ** n  # 729
    
    # Precompute balls as lists of indices
    # ball[cw_idx] = list of word indices covered by cw_idx
    balls = []
    for i in range(total):
        digits = [(i // (3 ** (n - 1 - pos))) % 3 for pos in range(n)]
        ball_indices = [i]
        for pos in range(n):
            for val in range(3):
                if val != digits[pos]:
                    new_digits = digits[:]
                    new_digits[pos] = val
                    new_idx = sum(d * (3 ** (n - 1 - p)) for p, d in enumerate(new_digits))
                    ball_indices.append(new_idx)
        balls.append(ball_indices)
        
    # Precompute ball membership matrix M (word, codeword)
    # M[w, c] = 1 if word w is in ball(c)
    M = np.zeros((total, total), dtype=np.int8)
    for c in range(total):
        for w in balls[c]:
            M[w, c] = 1

    # 1. Fast Greedy Initialization
    def greedy_init():
        covered = np.zeros(total, dtype=bool)
        cw_set = set()
        uncovered_mask = np.ones(total, dtype=np.int8)
        
        while uncovered_mask.sum() > 0:
            # gains[c] = number of currently uncovered words covered by c
            gains = M.T @ uncovered_mask
            # Pick the codeword with max gain
            best_c = np.argmax(gains)
            if gains[best_c] == 0:
                break # Should not happen if we start empty and balls cover everything
            cw_set.add(int(best_c))
            # Update covered and uncovered_mask
            covered[balls[best_c]] = True
            uncovered_mask[balls[best_c]] = 0
            
        return cw_set

    best_set = greedy_init()
    best_size = len(best_set)
    
    # 2. Local Search: Remove redundant codewords
    def local_reduce(cw_set, time_limit):
        cw_set = set(cw_set)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in cw_set:
            coverage_count[balls[c]] += 1
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for c in list(cw_set):
                # Check if any word in ball(c) has count 1
                # If min count > 1, we can remove
                if np.min(coverage_count[balls[c]]) > 1:
                    cw_set.remove(c)
                    coverage_count[balls[c]] -= 1
                    improved = True
        return cw_set

    best_set = local_reduce(best_set, deadline - 60.0)
    best_size = len(best_set)

    # 3. Simulated Annealing
    def simulated_annealing(initial_set, time_limit):
        cw_set = set(initial_set)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in cw_set:
            coverage_count[balls[c]] += 1
        
        best_set = set(cw_set)
        best_size = len(cw_set)
        best_cost = len(cw_set) + 1000 * np.sum(coverage_count == 0)
        
        temp = 10.0
        cooling = 0.99
        min_temp = 0.01
        
        # Precompute random choices for speed? No, just use random.choice
        while time.time() < time_limit and temp > min_temp:
            # Choose a move
            move_type = random.random()
            if len(cw_set) == 0:
                break
                
            if move_type < 0.4:
                # Remove a random codeword
                c = random.choice(list(cw_set))
                cw_set.remove(c)
                coverage_count[balls[c]] -= 1
                new_size = len(cw_set)
                new_uncovered = np.sum(coverage_count == 0)
                new_cost = new_size + 1000 * new_uncovered
                
                delta_cost = new_cost - best_cost
                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / temp):
                    # Accept
                    pass
                else:
                    # Reject
                    cw_set.add(c)
                    coverage_count[balls[c]] += 1
                    
            elif move_type < 0.8:
                # Add a random codeword
                c = random.randint(0, total - 1)
                if c not in cw_set:
                    cw_set.add(c)
                    coverage_count[balls[c]] += 1
                    new_size = len(cw_set)
                    new_uncovered = np.sum(coverage_count == 0)
                    new_cost = new_size + 1000 * new_uncovered
                    
                    delta_cost = new_cost - best_cost
                    if delta_cost <= 0 or random.random() < np.exp(-delta_cost / temp):
                        pass
                    else:
                        cw_set.remove(c)
                        coverage_count[balls[c]] -= 1
            else:
                # Swap: remove one, add another
                if len(cw_set) > 0:
                    c_out = random.choice(list(cw_set))
                    c_in = random.randint(0, total - 1)
                    if c_in != c_out and c_in not in cw_set:
                        cw_set.remove(c_out)
                        coverage_count[balls[c_out]] -= 1
                        cw_set.add(c_in)
                        coverage_count[balls[c_in]] += 1
                        
                        new_size = len(cw_set)
                        new_uncovered = np.sum(coverage_count == 0)
                        new_cost = new_size + 1000 * new_uncovered
                        
                        delta_cost = new_cost - best_cost
                        if delta_cost <= 0 or random.random() < np.exp(-delta_cost / temp):
                            pass
                        else:
                            # Reject
                            cw_set.remove(c_in)
                            coverage_count[balls[c_in]] -= 1
                            cw_set.add(c_out)
                            coverage_count[balls[c_out]] += 1
            
            # Update best
            current_size = len(cw_set)
            current_uncovered = np.sum(coverage_count == 0)
            current_cost = current_size + 1000 * current_uncovered
            if current_cost < best_cost:
                best_cost = current_cost
                best_size = current_size
                best_set = set(cw_set)
                
            temp *= cooling
            
        # Final cleanup
        best_set = local_reduce(best_set, time_limit)
        return best_set

    if time.time() < deadline - 10.0:
        # Run SA for a while
        sa_set = simulated_annealing(best_set, deadline - 10.0)
        if len(sa_set) < best_size:
            best_set = sa_set
            best_size = len(best_set)

    # Convert to list of lists
    result = []
    for idx in best_set:
        digits = [(idx // (3 ** (n - 1 - pos))) % 3 for pos in range(n)]
        result.append(digits)
    return result

def _greedy_set_cover(q, n, radius, deadline):
    total = q ** n

    def tuple_to_idx(tup):
        idx = 0
        for v in tup:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        return tuple(digits)

    # Precompute balls
    balls = []
    for i in range(total):
        w = idx_to_tuple(i)
        ball = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(tuple(new_w))
        idxs = [tuple_to_idx(b) for b in ball]
        balls.append(idxs)

    covered = np.zeros(total, dtype=bool)
    codeword_set = set()

    while not covered.all() and time.time() < deadline - 5.0:
        best_idx = -1
        best_gain = -1
        for idx in range(total):
            if idx in codeword_set:
                continue
            gain = 0
            for b_idx in balls[idx]:
                if not covered[b_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain <= 0:
            break
        codeword_set.add(best_idx)
        for b_idx in balls[best_idx]:
            covered[b_idx] = True

    result = []
    for idx in codeword_set:
        result.append(list(idx_to_tuple(idx)))
    return result
# EVOLVE-BLOCK-END
