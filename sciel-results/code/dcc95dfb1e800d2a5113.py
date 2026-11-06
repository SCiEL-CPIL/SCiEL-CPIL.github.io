import random
import math
import time

def construct():
    # Target differences
    targets = set(range(1, 31))
    
    def calculate_energy(A):
        if not A or A[0] != 0:
            return float('inf')
        
        diffs = []
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                d = A[j] - A[i]
                diffs.append(d)
        
        diff_counts = {}
        for d in diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collisions = 0
        for count in diff_counts.values():
            if count > 1:
                collisions += (count - 1)
        
        missing = 0
        for t in targets:
            if t not in diff_counts:
                missing += 1
        
        diameter = A[-1]
        
        size = len(A)
        size_penalty = 0.0
        if size < 9:
            size_penalty = (9 - size) * 1000.0
        elif size > 10:
            size_penalty = (size - 10) * 10.0

        W_COLLISION = 1000.0
        W_COVERAGE = 100.0
        W_DIAMETER = 1.0
        
        energy = (W_COLLISION * collisions) + (W_COVERAGE * missing) + (W_DIAMETER * diameter) + size_penalty
        return energy

    def is_valid(A):
        if not A or A[0] != 0:
            return False
        if len(A) < 9:
            return False
        
        diffs = []
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(A[j] - A[i])
        
        if len(diffs) != len(set(diffs)):
            return False
            
        diff_set = set(diffs)
        for t in targets:
            if t not in diff_set:
                return False
        return True

    def greedy_init():
        A = [0]
        # Add random small distinct numbers
        candidates = list(range(1, 60))
        random.shuffle(candidates)
        for c in candidates:
            if len(A) >= 9:
                break
            if c not in A:
                A.append(c)
        return sorted(A)

    best_solution = None
    best_energy = float('inf')
    
    A = greedy_init()
    current_energy = calculate_energy(A)
    
    T_start = 1000.0
    T_end = 1.0
    cooling_rate = 0.9999
    max_iters = 500000
    start_time = time.time()
    timeout = 110.0
    
    T = T_start
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        new_A = A[:]
        move_type = random.random()
        
        if move_type < 0.5:
            # Jitter
            if len(new_A) > 1:
                idx = random.randint(1, len(new_A) - 1)
                step = random.randint(-5, 5)
                new_val = new_A[idx] + step
                if new_val > 0 and new_val not in new_A:
                    new_A[idx] = new_val
                    new_A.sort()
                else:
                    continue
        elif move_type < 0.8:
            # Jump
            if len(new_A) > 1:
                idx = random.randint(1, len(new_A) - 1)
                max_val = max(new_A)
                new_val = random.randint(1, max(10, 2 * max_val))
                if new_val not in new_A:
                    new_A[idx] = new_val
                    new_A.sort()
                else:
                    continue
        elif move_type < 0.9:
            # Add
            if len(new_A) < 12:
                max_val = max(new_A)
                new_val = random.randint(1, max(10, 2 * max_val))
                if new_val not in new_A:
                    new_A.append(new_val)
                    new_A.sort()
                else:
                    continue
        else:
            # Remove
            if len(new_A) > 9:
                idx = random.randint(1, len(new_A) - 1)
                new_A.pop(idx)
        
        new_energy = calculate_energy(new_A)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            A = new_A
            current_energy = new_energy
            
            if is_valid(A):
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = A[:]
                    
        T *= cooling_rate
        if T < T_end:
            T = T_end

    if best_solution:
        return best_solution
    else:
        # Fallback: Try to repair the current A
        A_sorted = sorted(A)
        for _ in range(10000):
            if is_valid(A_sorted):
                return A_sorted
            
            idx = random.randint(1, len(A_sorted) - 1)
            A_sorted[idx] = random.randint(1, 2 * max(A_sorted))
            A_sorted.sort()
            
        return A

# Run construct
if __name__ == "__main__":
    res = construct()
    print(res)
