"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80
    random.seed(99)
    
    N = 20
    MAX_VAL = 10000
    
    def compute_scores(A):
        sset = set()
        pset = set()
        for i in range(N):
            ai = A[i]
            for j in range(i, N):
                sset.add(ai + A[j])
                pset.add(ai * A[j])
        return len(sset), len(pset)
    
    def objective(A):
        s, p = compute_scores(A)
        return max(s, p)
    
    def fast_objective(A):
        """Compute max(|A+A|, |A*A|) with early termination if either exceeds current best."""
        sset = set()
        pset = set()
        for i in range(N):
            ai = A[i]
            for j in range(i, N):
                sset.add(ai + A[j])
                pset.add(ai * A[j])
        return max(len(sset), len(pset))
    
    def local_improve(A, score, time_limit=30):
        """Local search by single-element replacement with fast objective."""
        t_end = time.time() + time_limit
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for idx in range(N):
                orig = A[idx]
                best_v = None
                best_sc = score
                # Generate candidates
                cands = set()
                for d in range(-50, 51):
                    v = orig + d
                    if 1 <= v <= MAX_VAL:
                        cands.add(v)
                for p in [2, 3, 5, 7, 11, 13]:
                    for e in range(0, 15):
                        v = p ** e
                        if 1 <= v <= MAX_VAL:
                            cands.add(v)
                for other in A:
                    for mult in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                        v = other * mult
                        if 1 <= v <= MAX_VAL:
                            cands.add(v)
                        if mult > 1 and other % mult == 0:
                            v = other // mult
                            if 1 <= v <= MAX_VAL:
                                cands.add(v)
                    # Add neighbors of other
                    for d in [-3, -2, -1, 1, 2, 3]:
                        v = other + d
                        if 1 <= v <= MAX_VAL:
                            cands.add(v)
                
                for v in cands:
                    if v in A:
                        continue
                    test_A = A[:]
                    test_A[idx] = v
                    sc = fast_objective(test_A)
                    if sc < best_sc:
                        best_sc = sc
                        best_v = v
                if best_v is not None:
                    A[idx] = best_v
                    score = best_sc
                    improved = True
        return A, score
    
    def local_improve_pairwise(A, score, time_limit=20):
        """Local search by pairwise element replacement."""
        t_end = time.time() + time_limit
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for i in range(N):
                if time.time() > t_end:
                    break
                for j in range(i+1, N):
                    if time.time() > t_end:
                        break
                    orig_i, orig_j = A[i], A[j]
                    best_pair = None
                    best_sc = score
                    
                    # Try swapping with nearby values
                    for di in range(-10, 11):
                        for dj in range(-10, 11):
                            vi = orig_i + di
                            vj = orig_j + dj
                            if not (1 <= vi <= MAX_VAL and 1 <= vj <= MAX_VAL):
                                continue
                            if vi == vj:
                                continue
                            if vi in A or vj in A:
                                continue
                            test_A = A[:]
                            test_A[i] = vi
                            test_A[j] = vj
                            sc = fast_objective(test_A)
                            if sc < best_sc:
                                best_sc = sc
                                best_pair = (vi, vj)
                    
                    if best_pair is not None:
                        A[i] = best_pair[0]
                        A[j] = best_pair[1]
                        score = best_sc
                        improved = True
        return A, score
    
    # Generate many structured candidates
    candidates = []
    
    # 2D lattices {p^i * q^j}
    for p in [2, 3, 5]:
        for q in [2, 3, 5, 7, 11]:
            if p == q:
                continue
            for i_max in range(0, 6):
                for j_max in range(0, 14):
                    pts = [(i, j) for i in range(i_max+1) for j in range(j_max+1)]
                    if len(pts) < N:
                        continue
                    vals = [p**i * q**j for i, j in pts]
                    if len(set(vals)) == len(vals) and max(vals) <= MAX_VAL:
                        candidates.append(sorted(vals)[:N])
    
    # {2^j, k*2^j} for various k
    for k in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        cand = sorted(set([2**j for j in range(12)] + [k*(2**j) for j in range(12)]))
        if len(cand) >= N and max(cand) <= MAX_VAL:
            candidates.append(cand[:N])
    
    # {2^i * 3^j} with rectangular grids
    for im in range(1, 5):
        for jm in range(1, 12):
            if (im+1)*(jm+1) < N:
                continue
            vals = [2**i * 3**j for i in range(im+1) for j in range(jm+1)]
            if len(set(vals)) == len(vals) and max(vals) <= MAX_VAL:
                candidates.append(sorted(vals)[:N])
    
    # 7-smooth numbers (numbers of form 2^a * 3^b * 5^c)
    smooth7 = []
    for a in range(15):
        for b in range(10):
            for c in range(8):
                v = (2**a) * (3**b) * (5**c)
                if v <= MAX_VAL:
                    smooth7.append(v)
    smooth7 = sorted(set(smooth7))
    if len(smooth7) >= N:
        candidates.append(smooth7[:N])
        # Also try subsets with different density
        for start in range(0, min(50, len(smooth7)-N)):
            candidates.append(smooth7[start:start+N])
    
    # Consecutive and arithmetic progressions
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(1, 41, 2)))  # odd numbers 1..39
    candidates.append(list(range(2, 41, 2)))  # even numbers 2..40
    
    # Geometric progressions
    for ratio in [1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5]:
        vals = [int(round(ratio**i)) for i in range(N)]
        if len(set(vals)) == N and max(vals) <= MAX_VAL and min(vals) >= 1:
            candidates.append(sorted(vals))
    
    # Mixed: small consecutive + powers of 2
    for base in range(1, 10):
        for count in range(5, 16):
            small = list(range(base, base+count))
            powers = [2**k for k in range(count, 20) if 2**k <= MAX_VAL]
            cand = sorted(set(small + powers))
            if len(cand) >= N and max(cand) <= MAX_VAL:
                candidates.append(cand[:N])
    
    # Random structured candidates
    for _ in range(20):
        cand = set()
        # Mix of smooth numbers and random
        for v in smooth7[:30]:
            cand.add(v)
        while len(cand) < N:
            v = random.randint(1, MAX_VAL)
            cand.add(v)
        cand = sorted(cand)[:N]
        if len(set(cand)) == N and max(cand) <= MAX_VAL:
            candidates.append(cand)
    
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != N or max(cand) > MAX_VAL or min(cand) < 1 or len(set(cand)) != N:
            continue
        sc = objective(cand)
        if sc < best_score:
            best_score = sc
            best_A = cand[:]
    
    if best_A is None:
        best_A = list(range(1, N+1))
        best_score = objective(best_A)
    
    # Local improvement on best candidate
    if time.time() < deadline - 15:
        best_A, best_score = local_improve(best_A, best_score, time_limit=min(25, deadline - time.time() - 15))
    
    # Try local improvement on top-5 candidates
    if time.time() < deadline - 10:
        scored = []
        for cand in candidates:
            if len(cand) != N or max(cand) > MAX_VAL or min(cand) < 1 or len(set(cand)) != N:
                continue
            sc = objective(cand)
            scored.append((sc, cand))
        scored.sort()
        for sc, cand in scored[:5]:
            if time.time() > deadline - 10:
                break
            A2, sc2 = local_improve(cand[:], sc, time_limit=min(10, deadline - time.time() - 10))
            if sc2 < best_score:
                best_score = sc2
                best_A = A2
    
    # Pairwise local improvement on best
    if time.time() < deadline - 5:
        best_A, best_score = local_improve_pairwise(best_A, best_score, time_limit=min(10, deadline - time.time() - 5))
    
    return best_A
# EVOLVE-BLOCK-END
