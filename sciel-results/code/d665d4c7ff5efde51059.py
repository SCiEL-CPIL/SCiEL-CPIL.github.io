import time
import random
import numpy as np

N = 51975

def get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

DIVS = [d for d in get_divisors(N) if d > 1]

def score_solution(classes):
    covered = np.zeros(N, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(N - covered.sum())

def greedy_construct(divs, N, rng=None, mode='random'):
    if rng is None:
        rng = random
    covered = np.zeros(N, dtype=np.uint8)
    used = set()
    classes = []
    
    if mode == 'random':
        order = divs[:]
        rng.shuffle(order)
    elif mode == 'small_first':
        order = sorted(divs)
    elif mode == 'large_first':
        order = sorted(divs, reverse=True)
    elif mode == 'density_first':
        order = sorted(divs, key=lambda m: N//m, reverse=True)
    else:
        order = divs[:]
        rng.shuffle(order)
    
    for m in order:
        if m in used:
            continue
        # Find best residue using numpy
        # For each r in [0, m), count how many positions r, r+m, r+2m, ... are uncovered
        # covered[r::m] gives us the coverage pattern
        gains = np.array([covered[r::m].sum() for r in range(m)], dtype=np.int32)
        # We want to MINIMIZE covered, so maximize uncovered = total - covered
        total_in_class = len(range(0, N, m))  # N//m or (N-1)//m + 1
        best_r = int(np.argmin(gains))
        best_gain = total_in_class - gains[best_r]
        
        if best_gain > 0:
            classes.append([m, best_r])
            used.add(m)
            # Update covered using numpy
            covered[best_r::m] = 1
    return classes

def local_search_numpy(classes, N, max_passes=20, time_limit=None):
    if not classes:
        return classes, score_solution(classes)
    
    if time_limit is None:
        time_limit = 10
    
    start = time.time()
    current_score = score_solution(classes)
    
    for _ in range(max_passes):
        if time.time() - start > time_limit:
            break
        improved = False
        # Try optimizing each class
        for i in range(len(classes)):
            m, r_old = classes[i]
            
            # Calculate coverage of all OTHER classes using numpy
            cov_rest = np.zeros(N, dtype=np.uint8)
            for j in range(len(classes)):
                if j == i:
                    continue
                mj, rj = classes[j]
                cov_rest[rj::mj] = 1
            
            # Find best r for m given cov_rest
            # We want to minimize covered positions in cov_rest
            # For each r, count how many positions in r, r+m, r+2m, ... are already covered
            gains = np.array([cov_rest[r::m].sum() for r in range(m)], dtype=np.int32)
            best_r = int(np.argmin(gains))
            
            if gains[best_r] < gains[r_old]:
                classes[i][1] = best_r
                improved = True
        
        if not improved:
            break
            
    final_score = score_solution(classes)
    return classes, final_score

def simulated_annealing(classes, N, start_temp=100, end_temp=1, iterations=500, time_limit=None):
    if time_limit is None:
        time_limit = 30
    
    start = time.time()
    current_score = score_solution(classes)
    best_classes = [c[:] for c in classes]
    best_score = current_score
    
    for it in range(iterations):
        if time.time() - start > time_limit:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / iterations)
        
        # Pick a random class and change its residue
        idx = random.randint(0, len(classes) - 1)
        m = classes[idx][0]
        old_r = classes[idx][1]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
        
        classes[idx][1] = new_r
        new_score = score_solution(classes)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in classes]
        else:
            classes[idx][1] = old_r
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_classes = []
    best_score = N  # Max possible score
    
    rng = random.Random(42)
    
    # Try different greedy strategies
    for mode in ['small_first', 'large_first', 'density_first', 'random', 'random', 'random']:
        if time.time() > deadline - 30:
            break
        classes = greedy_construct(DIVS, N, rng, mode=mode)
        classes, score = local_search_numpy(classes, N, max_passes=15, time_limit=5)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
            if best_score == 0:
                return best_classes
    
    # Simulated annealing polish on best solution
    if best_classes and time.time() < deadline - 10:
        polished, score = simulated_annealing([c[:] for c in best_classes], N, 
                                               start_temp=50, end_temp=0.1, 
                                               iterations=200, time_limit=20)
        if score < best_score:
            best_score = score
            best_classes = polished
    
    # More randomized restarts with time remaining
    while time.time() < deadline - 5:
        classes = greedy_construct(DIVS, N, rng, mode='random')
        classes, score = local_search_numpy(classes, N, max_passes=10, time_limit=3)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
            if best_score == 0:
                break
            # Polish the new best
            if time.time() < deadline - 10:
                polished, score2 = simulated_annealing([c[:] for c in best_classes], N,
                                                       start_temp=30, end_temp=0.1,
                                                       iterations=100, time_limit=8)
                if score2 < best_score:
                    best_score = score2
                    best_classes = polished
    
    return best_classes
