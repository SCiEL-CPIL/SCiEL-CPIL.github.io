import time
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(sorted(set(U)), dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Efficient counting via FFT
    N = 2 * mx + 1
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _golomb_ruler(n, max_val=5000):
    """Construct a Golomb ruler with n marks using greedy approach."""
    if n <= 1:
        return [0]
    marks = [0]
    for i in range(1, n):
        best_pos = -1
        best_min_diff = -1
        for pos in range(marks[-1] + 1, max_val + 1):
            # Check all pairwise differences
            valid = True
            min_diff = float('inf')
            for j in range(len(marks)):
                for k in range(j + 1, len(marks)):
                    d = abs(marks[j] - marks[k])
                    if d == abs(pos - marks[j]) or d == abs(pos - marks[k]):
                        valid = False
                        break
                    min_diff = min(min_diff, d)
                if not valid:
                    break
            if not valid:
                continue
            # Also check diff with new pos
            for j in range(len(marks)):
                d = abs(pos - marks[j])
                for k in range(len(marks)):
                    if k != j:
                        d2 = abs(pos - marks[k])
                        if d == d2:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue
            if min_diff > best_min_diff:
                best_min_diff = min_diff
                best_pos = pos
                # Early exit if we find a good one
                if best_min_diff >= 5:
                    break
        if best_pos == -1:
            # Fallback: just use next integer
            best_pos = marks[-1] + 1
        marks.append(best_pos)
    return marks

def _greedy_build(deadline, max_elements=50, max_val=3000):
    """
    Greedy element-addition: start with {0}, then at each step try adding
    candidate elements and keep the one that maximizes score.
    """
    U = [0]
    current_score = 0.0
    
    # Start with a Golomb ruler as seed
    golomb = _golomb_ruler(min(max_elements, 15), max_val=max_val)
    U = golomb
    current_score = _score_set(U)
    
    # Greedy addition
    while time.time() < deadline - 10 and len(U) < max_elements:
        best_candidate = None
        best_score = current_score
        
        # Try candidates: multiples of existing gaps, primes, etc.
        candidates = set()
        # Add some structured candidates
        for x in U:
            for gap in [1, 2, 3, 5, 7, 11, 13]:
                candidates.add(x + gap)
                candidates.add(x + gap * 2)
                candidates.add(x + gap * 3)
        # Add some random candidates
        for _ in range(20):
            candidates.add(np.random.randint(1, max_val))
        # Add primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            candidates.add(p)
            candidates.add(p * 2)
            candidates.add(p * 3)
        
        # Filter: must be nonnegative, not already in U, not too large
        U_set = set(U)
        candidates = [c for c in candidates if c > 0 and c not in U_set and c <= max_val]
        
        if not candidates:
            break
        
        # Try each candidate
        for cand in candidates:
            if time.time() > deadline - 10:
                break
            new_U = U + [cand]
            try:
                s = _score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_candidate = cand
            except Exception:
                pass
        
        if best_candidate is None:
            # Try a broader random search
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                cand = np.random.randint(1, max_val)
                if cand in U_set:
                    continue
                new_U = U + [cand]
                try:
                    s = _score_set(new_U)
                    if s > best_score:
                        best_score = s
                        best_candidate = cand
                except Exception:
                    pass
        
        if best_candidate is None:
            break
        
        U.append(best_candidate)
        U.sort()
        current_score = best_score
    
    return U, current_score

def _local_search(U, deadline):
    """Local search: try swapping elements, adding/removing elements."""
    U = list(U)
    current_score = _score_set(U)
    max_val = max(U) * 2 + 100 if max(U) > 0 else 100
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try adding a new element
        candidates = set()
        for x in U:
            for gap in range(1, 20):
                candidates.add(x + gap)
        for _ in range(30):
            candidates.add(np.random.randint(1, max_val))
        
        U_set = set(U)
        candidates = [c for c in candidates if c not in U_set and c > 0 and c <= max_val]
        
        best_add = None
        best_add_score = current_score
        for cand in candidates[:100]:  # Limit to avoid too many FFT calls
            if time.time() > deadline - 5:
                break
            new_U = U + [cand]
            try:
                s = _score_set(new_U)
                if s > best_add_score:
                    best_add_score = s
                    best_add = cand
            except Exception:
                pass
        
        if best_add is not None:
            U.append(best_add)
            U.sort()
            current_score = best_add_score
            improved = True
            max_val = max(max_val, best_add + 100)
        
        # Try removing an element (if set is large)
        if len(U) > 5:
            for i in range(len(U)):
                if time.time() > deadline - 5:
                    break
                if U[i] == 0:
                    continue  # Keep 0
                new_U = U[:i] + U[i+1:]
                try:
                    s = _score_set(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        improved = True
                        break
                except Exception:
                    pass
        
        if not improved:
            # Try replacing an element
            for i in range(len(U)):
                if time.time() > deadline - 5:
                    break
                if U[i] == 0:
                    continue
                # Try replacing with nearby values
                for delta in range(-10, 11):
                    new_val = U[i] + delta
                    if new_val <= 0 or new_val in U_set:
                        continue
                    new_U = U[:i] + [new_val] + U[i+1:]
                    new_U.sort()
                    try:
                        s = _score_set(new_U)
                        if s > current_score:
                            U = new_U
                            U_set = set(U)
                            current_score = s
                            improved = True
                            break
                    except Exception:
                        pass
                if improved:
                    break
        
        if not improved:
            # Random restart in neighborhood
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                idx = np.random.randint(0, len(U))
                if U[idx] == 0:
                    continue
                new_val = U[idx] + np.random.randint(-50, 51)
                if new_val <= 0 or new_val in U_set:
                    continue
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_U.sort()
                try:
                    s = _score_set(new_U)
                    if s > current_score:
                        U = new_U
                        U_set = set(U)
                        current_score = s
                        improved = True
                        break
                except Exception:
                    pass
            if not improved:
                break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Phase 1: Greedy build
    U, score = _greedy_build(deadline, max_elements=30, max_val=2000)
    
    # Phase 2: Local search refinement
    U, score = _local_search(U, deadline)
    
    # Ensure 0 is in the set
    if 0 not in U:
        U.insert(0, 0)
    
    U.sort()
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
