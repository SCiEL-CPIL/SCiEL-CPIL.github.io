import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        # Returns list of counts for sums N_LOW..N_HIGH
        reps = [0] * RANGE_SIZE
        A_list = list(A)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                aj = A_list[j]
                s = ai + aj
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        return reps

    def is_covered(reps):
        return all(r > 0 for r in reps)

    def get_peak(reps):
        return max(reps)

    # Phase 1: Initialize a covering set
    # Start with 0 and 128 to anchor the range
    A = set([0, 128])
    
    # Greedy add to ensure coverage
    while True:
        if time.time() > deadline:
            break
        reps = calc_reps(A)
        if is_covered(reps):
            break
        
        # Find a missing sum
        missing_indices = [i for i, r in enumerate(reps) if r == 0]
        if not missing_indices:
            break
            
        # Pick a random missing sum
        idx = random.choice(missing_indices)
        S = N_LOW + idx
        
        # Find candidates y
        candidates = []
        for z in A:
            y = S - z
            if 0 <= y <= MAX_VAL and y not in A:
                candidates.append(y)
        
        if candidates:
            y = random.choice(candidates)
            A.add(y)
        else:
            non_A = [v for v in range(MAX_VAL + 1) if v not in A]
            if non_A:
                A.add(random.choice(non_A))
            else:
                break # A is full
        
    if not is_covered(calc_reps(A)):
        # Fallback
        return list(range(64, 129))

    # Phase 2: Optimization with Simulated Annealing
    best_A = set(A)
    best_reps = calc_reps(best_A)
    best_peak = get_peak(best_reps)
    
    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Pick a random swap
        if not current_A:
            break
            
        a = random.choice(list(current_A))
        
        # Find b not in A
        while True:
            b = random.randint(0, MAX_VAL)
            if b not in current_A:
                break
        
        # Perform swap
        new_A = set(current_A)
        new_A.remove(a)
        new_A.add(b)
        
        new_reps = calc_reps(new_A)
        
        # Check validity
        if not is_covered(new_reps):
            T *= cooling
            continue
            
        new_peak = get_peak(new_reps)
        
        # Calculate energy difference
        delta = new_peak - current_peak
        
        # Accept if better, or with probability exp(-delta/T) if worse
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            # Accept half the time to explore
            accept = random.random() < 0.5
        else:
            # delta > 0
            if T > 0:
                accept = random.random() < math.exp(-delta / T)
        
        if accept:
            current_A = new_A
            current_reps = new_reps
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
                best_reps = current_reps
        
        T *= cooling
        
    # Final Polish: Greedy Removals
    # Try to remove elements that don't increase peak or break coverage
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in list(best_A):
            if time.time() > deadline:
                break
            new_A = best_A - {a}
            if not new_A:
                continue
            new_reps = calc_reps(new_A)
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                if new_peak <= best_peak:
                    best_A = new_A
                    best_reps = new_reps
                    best_peak = new_peak
                    improved = True

    return list(best_A)

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
