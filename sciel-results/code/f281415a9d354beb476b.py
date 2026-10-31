import math
import time
import random
import numpy as np
from scipy import signal

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_small(U):
    """Exact O(n^2) score for small sets."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _fft_score(U):
    """
    Fast score calculation using FFT/SciPy.
    U: list of non-negative integers.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    L = 1
    while L < 2 * M + 1:
        L *= 2
        
    # Create indicator vector
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # Sumset: A * A
    F_A = np.fft.fft(A)
    F_S = F_A * F_A
    S = np.fft.ifft(F_S).real
    
    sum_count = np.sum(S[:2*M+1] > 0.1)
    
    # Diffset: Use SciPy correlate for robustness
    autocorr = signal.correlate(A, A, mode='full')
    diff_count = np.sum(autocorr[L-1-M : L-1+M+1] > 0.1)
    
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count, 2) / denom

def _get_score(U):
    """Hybrid scorer: exact for small, FFT for large."""
    if max(U) < 2000:
        return _score_small(U)
    return _fft_score(U)

def _greedy_construct(max_size=50, max_val=10000, time_budget=10.0):
    """
    Construct a set by greedily adding elements that maximize the score.
    This explores a different design space than local search on fixed-size sets.
    """
    U = [0]
    current_score = 0.0
    rng = random.Random(123)
    start_time = time.time()
    
    # We will try to grow the set up to max_size or until time runs out.
    # We also need to choose where to place the new element.
    # Instead of trying all positions in [1, max_val], we can use a heuristic.
    # Or we can do a coarse search for the best position.
    
    for _ in range(max_size - 1):
        if time.time() - start_time > time_budget or _remaining() < 5.0:
            break
            
        best_new_val = -1
        best_score = current_score
        
        # Candidate positions: 
        # 1. Just beyond the current max
        # 2. Random positions in a range
        # 3. Positions that might fill gaps in the sum/diff sets?
        
        # To keep it fast, we sample candidate positions.
        current_max = max(U)
        # Search range: [1, max_val] but focus on areas likely to be useful.
        # Let's sample 100 random positions and the max+1 position.
        candidates = set()
        candidates.add(current_max + 1)
        candidates.add(current_max + rng.randint(1, 10))
        
        for _ in range(50):
            candidates.add(rng.randint(1, max_val))
            
        # Also try to "spread out"
        for _ in range(20):
            candidates.add(int(rng.uniform(1, max_val)))
            
        for c in candidates:
            if c in U:
                continue
            new_U = U + [c]
            s = _get_score(new_U)
            if s > best_score:
                best_score = s
                best_new_val = c
                
        if best_new_val != -1:
            U.append(best_new_val)
            U.sort()
            current_score = best_score
        else:
            break
            
    return U, current_score

def _simulated_annealing(U, time_budget=30.0):
    """
    Optimize the set using simulated annealing.
    Moves: 
    1. Move an element to a new position.
    2. Add a new element (if size < max_allowed).
    3. Remove an element (if size > min_allowed).
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    
    current_U = list(U)
    current_score = _get_score(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    rng = random.Random(456)
    start_time = time.time()
    
    T = 0.1  # Initial temperature
    cooling_rate = 0.99
    
    min_size = 3
    max_size = 100
    max_val = 20000
    
    while time.time() - start_time < time_budget and _remaining() > 5.0:
        T *= cooling_rate
        
        # Generate a move
        move_type = rng.random()
        new_U = list(current_U)
        
        if move_type < 0.7:
            # Move an element
            if len(new_U) > 1:
                idx = rng.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                # New position: random in range or near current
                if rng.random() < 0.5:
                    new_val = rng.randint(1, max_val)
                else:
                    new_val = old_val + rng.randint(-100, 100)
                
                if new_val < 0:
                    new_val = 0
                if new_val in new_U:
                    continue
                
                new_U[idx] = new_val
                new_U.sort()
                
        elif move_type < 0.85:
            # Add an element
            if len(new_U) < max_size:
                new_val = rng.randint(1, max_val)
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U.sort()
                else:
                    continue
        else:
            # Remove an element
            if len(new_U) > min_size:
                idx = rng.randint(1, len(new_U) - 1)
                new_U.pop(idx)
        
        new_score = _get_score(new_U)
        
        # Accept or reject
        if new_score > current_score or rng.random() < math.exp((new_score - current_score) / T):
            current_U = new_U
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
                
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    best_U = [0, 1]
    best_score = 0.0
    
    # Phase 1: Greedy Construction
    # Try a few different greedy runs with different parameters
    for i in range(3):
        if _remaining() < 10.0:
            break
        U, s = _greedy_construct(max_size=50 + i*10, max_val=5000 + i*5000, time_budget=5.0)
        if s > best_score:
            best_score = s
            best_U = U
            
    # Phase 2: Simulated Annealing on the best greedy result
    if _remaining() > 20.0:
        U, s = _simulated_annealing(best_U, time_budget=min(30.0, _remaining() - 10.0))
        if s > best_score:
            best_score = s
            best_U = U
            
    # Phase 3: Local Search Polish
    if _remaining() > 10.0:
        # Simple local search on the best SA result
        U = sorted(list(best_U))
        current_score = best_score
        rng = random.Random(789)
        start = time.time()
        
        while time.time() - start < min(10.0, _remaining() - 5.0):
            improved = False
            indices = list(range(1, len(U)))
            rng.shuffle(indices)
            
            for i in indices:
                orig = U[i]
                # Try small shifts
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = orig + delta
                    if new_val < 0 or new_val in U:
                        continue
                    new_U = list(U)
                    new_U[i] = new_val
                    new_U.sort()
                    s = _get_score(new_U)
                    if s > current_score:
                        current_score = s
                        U = new_U
                        improved = True
                        break
                if improved:
                    break
                    
            if not improved:
                # Random perturbation
                if rng.random() < 0.1:
                    idx = rng.randint(1, len(U)-1)
                    new_val = U[idx] + rng.randint(-20, 20)
                    if new_val > 0 and new_val not in U:
                        U[idx] = new_val
                        U.sort()
                        s = _get_score(U)
                        if s > current_score:
                            current_score = s
                            
        if current_score > best_score:
            best_score = current_score
            best_U = U
            
    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
