import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def compute_scores(A):
        A = sorted(A)
        n = len(A)
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    # Strategy: use multiplicatively structured sets
    # Try various candidate families
    
    best = None
    best_score = float('inf')
    
    # Family 1: Pure geometric sequences {b, b^2, ..., b^20}
    for b in range(2, 12):
        A = [b**i for i in range(1, 21)]
        if max(A) <= 10000:
            score = compute_scores(A)
            if score < best_score:
                best_score = score
                best = A[:]
    
    # Family 2: Powers of 2: {1, 2, 4, ..., 2^19}
    A = [1 << i for i in range(20)]
    if max(A) <= 10000:
        score = compute_scores(A)
        if score < best_score:
            best_score = score
            best = A[:]
    
    # Family 3: 2,3-smooth numbers - triangular structure
    # Generate all 2,3-smooth numbers <= 10000
    smooth = set()
    for i in range(0, 14):  # 2^i <= 10000
        for j in range(0, 10):  # 3^j <= 10000
            val = (1 << i) * (3 ** j)
            if val <= 10000:
                smooth.add(val)
    smooth = sorted(smooth)
    
    # Try picking 20 from smooth with good structure
    # Use a greedy approach: start with a geometric core, add nearby smooth numbers
    for base in [2, 3, 4, 5, 6]:
        # Core: powers of base
        core = [base**i for i in range(1, 21)]
        if max(core) <= 10000:
            core_set = set(core)
            # Try replacing some elements with nearby smooth numbers to reduce sumset
            candidates = [s for s in smooth if s not in core_set and s <= 10000]
            # Greedy: try swapping each core element with a smooth number
            current = core[:]
            current_score = compute_scores(current)
            improved = True
            while improved and (time.time() < deadline):
                improved = False
                for idx in range(20):
                    for cand in candidates[:50]:  # limit candidates for speed
                        if cand in current:
                            continue
                        new_A = current[:]
                        new_A[idx] = cand
                        if len(set(new_A)) != 20:
                            continue
                        if max(new_A) > 10000:
                            continue
                        s = compute_scores(new_A)
                        if s < current_score:
                            current = new_A
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            if current_score < best_score:
                best_score = current_score
                best = current[:]
    
    # Family 4: Arithmetic-geometric hybrid
    # {a * b^i : i=0..19} for various a, b
    for a in range(1, 10):
        for b in range(2, 8):
            A = [a * (b**i) for i in range(20)]
            if max(A) <= 10000 and len(set(A)) == 20:
                score = compute_scores(A)
                if score < best_score:
                    best_score = score
                    best = A[:]
    
    # Family 5: Small consecutive integers with a twist
    # Try {1,2,3,...,10, 11,13,17,19,23,29,31,37,41,43} (primes after 10)
    # Actually, try various small sets
    small_sets = [
        list(range(1, 21)),
        list(range(2, 22)),
        list(range(1, 11)) + [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    ]
    for A in small_sets:
        if len(set(A)) == 20 and max(A) <= 10000:
            score = compute_scores(A)
            if score < best_score:
                best_score = score
                best = A[:]
    
    # Family 6: Powers of 2 and 3 combined: {2^i * 3^j} with i+j <= k
    for k in range(0, 10):
        candidates = []
        for i in range(0, k+1):
            for j in range(0, k+1-i):
                val = (1 << i) * (3 ** j)
                if val <= 10000:
                    candidates.append(val)
        candidates = sorted(set(candidates))
        if len(candidates) >= 20:
            # Take first 20 (smallest)
            A = candidates[:20]
            if len(set(A)) == 20:
                score = compute_scores(A)
                if score < best_score:
                    best_score = score
                    best = A[:]
    
    # Local search on the best found
    if best is not None and time.time() < deadline:
        current = best[:]
        current_score = compute_scores(current)
        improved = True
        while improved and (time.time() < deadline - 5):
            improved = False
            for idx in range(20):
                # Try replacing current[idx] with a nearby number
                old_val = current[idx]
                for delta in range(-10, 11):
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in current:
                        continue
                    new_A = current[:]
                    new_A[idx] = new_val
                    s = compute_scores(new_A)
                    if s < current_score:
                        current = new_A
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
        if current_score < best_score:
            best_score = current_score
            best = current[:]
    
    return best
