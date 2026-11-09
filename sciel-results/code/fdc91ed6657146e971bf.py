import random
import time
import numpy as np

N = 256

def _build_sums(S):
    """Build the set of all sums a+b with a<=b from S."""
    sums = set()
    for i in range(len(S)):
        sums.add(2 * S[i])
        for j in range(i):
            sums.add(S[i] + S[j])
    return sums

def _is_sidon(S):
    """Check if S is a Sidon set (all unordered pair sums including diagonal are distinct)."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S, N):
    """Check if S is inclusion-maximal: no element from [1,N]\S can be added."""
    sums = _build_sums(S)
    S_set = set(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        compatible = True
        for a in S:
            if x + a in sums:
                compatible = False
                break
        if compatible:
            return False
    return True

def _greedy_blocking(seed, deadline):
    """Greedy construction with blocking heuristic."""
    random.seed(seed)
    S = []
    S_set = set()
    sums = set()
    
    while True:
        best_x = -1
        best_score = -1
        tied_candidates = []
        
        for x in range(1, N + 1):
            if x in S_set:
                continue
            
            # Check compatibility
            compatible = True
            if 2 * x in sums:
                compatible = False
            else:
                for a in S:
                    if x + a in sums:
                        compatible = False
                        break
            
            if not compatible:
                continue
            
            # Score: how many other candidates does x block?
            score = 0
            for y in range(1, N + 1):
                if y == x or y in S_set:
                    continue
                if x + y in sums:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
                tied_candidates = [x]
            elif score == best_score:
                tied_candidates.append(x)
        
        if not tied_candidates:
            break
        
        if time.time() > deadline:
            break
        
        x_star = random.choice(tied_candidates)
        S.append(x_star)
        S_set.add(x_star)
        sums.add(2 * x_star)
        for a in S[:-1]:
            sums.add(x_star + a)
    
    return S

def _local_search_reduce(S, deadline):
    """Try to remove elements while maintaining maximality."""
    S = list(S)
    sums = _build_sums(S)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(len(S)):
            if time.time() > deadline:
                break
            # Try removing S[i]
            S_test = S[:i] + S[i+1:]
            if len(S_test) == 0:
                continue
            
            # Check if still Sidon (it will be, since it's a subset)
            # Check if still maximal
            if _is_maximal(S_test, N):
                S = S_test
                sums = _build_sums(S)
                improved = True
                break
    
    return S

def _swap_search(S, deadline):
    """Try swapping elements to reduce size while maintaining maximality."""
    S = list(S)
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            for x in range(1, N + 1):
                if x in S:
                    continue
                
                # Try swapping S[i] with x
                S_test = S[:]
                S_test[i] = x
                S_test.sort()
                
                # Check if still Sidon
                if not _is_sidon(S_test):
                    continue
                
                # Check if still maximal
                if _is_maximal(S_test, N):
                    S = S_test
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return S

def _structured_construction(deadline):
    """Try a structured approach: pick elements to maximize spread."""
    # Start with a few well-spread elements
    base_candidates = [1, 64, 128, 192, 255]
    
    best_S = None
    best_size = float('inf')
    
    # Try different subsets of base candidates
    from itertools import combinations
    
    for r in range(1, len(base_candidates) + 1):
        if time.time() > deadline:
            break
        
        for combo in combinations(base_candidates, r):
            S = list(combo)
            S.sort()
            
            if not _is_sidon(S):
                continue
            
            # Now greedily add more elements
            S_set = set(S)
            sums = _build_sums(S)
            
            while True:
                best_x = -1
                best_score = -1
                
                for x in range(1, N + 1):
                    if x in S_set:
                        continue
                    
                    compatible = True
                    if 2 * x in sums:
                        compatible = False
                    else:
                        for a in S:
                            if x + a in sums:
                                compatible = False
                                break
                    
                    if not compatible:
                        continue
                    
                    score = 0
                    for y in range(1, N + 1):
                        if y == x or y in S_set:
                            continue
                        if x + y in sums:
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_x = x
                
                if best_x == -1:
                    break
                
                if time.time() > deadline:
                    break
                
                S.append(best_x)
                S.sort()
                S_set.add(best_x)
                sums.add(2 * best_x)
                for a in S[:-1]:
                    sums.add(best_x + a)
            
            if len(S) < best_size:
                best_size = len(S)
                best_S = S[:]
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Greedy blocking with multiple seeds
    for i in range(100):
        if time.time() > deadline:
            break
        
        S = _greedy_blocking(i, deadline)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 2: Local search reduction on best found
    if best_set is not None and time.time() < deadline:
        S = _local_search_reduce(best_set, deadline)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 3: Swap search
    if best_set is not None and time.time() < deadline:
        S = _swap_search(best_set, deadline)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 4: Structured construction
    if time.time() < deadline:
        S = _structured_construction(deadline)
        if S is not None and len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 5: More diverse restarts if time remains
    if best_set is not None and time.time() < deadline - 5.0:
        for i in range(100, 200):
            if time.time() > deadline:
                break
            
            S = _greedy_blocking(i, deadline)
            S = _local_search_reduce(S, deadline)
            if len(S) < best_size:
                best_size = len(S)
                best_set = S
    
    # Fallback: simple ascending greedy
    if best_set is None:
        S = []
        S_set = set()
        sums = set()
        for x in range(1, N + 1):
            if 2 * x in sums:
                continue
            compatible = True
            for a in S:
                if x + a in sums:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_set.add(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        best_set = S
    
    return best_set
