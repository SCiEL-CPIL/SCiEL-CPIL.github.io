"""Direct constructive approach for minimum maximal no-isosceles set."""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and per-pair blocked sets."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # Precompute for each pair (i,j) the set of points k that form an isosceles triangle with i,j
    # pair_blocked[i][j] = set of k such that {i,j,k} is isosceles
    # We'll store as a dict mapping (i,j) to a frozenset for fast lookup
    pair_blocked = {}
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i, j]
            # k where d(i,k) == dij (but k != j)
            mask1 = (dist[i] == dij)
            mask1[j] = False
            # k where d(j,k) == dij (but k != i)
            mask2 = (dist[j] == dij)
            mask2[i] = False
            # k where d(i,k) == d(j,k) (perpendicular bisector)
            mask3 = (dist[i] == dist[j])
            mask3[i] = False
            mask3[j] = False
            
            combined = mask1 | mask2 | mask3
            ks = np.where(combined)[0]
            blocked_set = frozenset(ks.tolist())
            pair_blocked[(i, j)] = blocked_set
            pair_blocked[(j, i)] = blocked_set
    
    return points, dist, pair_blocked

def _is_valid_set(S, pair_blocked):
    """Check if S contains no isosceles triangle."""
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            key = (S[i], S[j])
            if key in pair_blocked:
                if pair_blocked[key] & S_set:
                    return False
    return True

def _get_blocked(S, pair_blocked):
    """Get points that would create an isosceles triangle with any pair in S."""
    blocked = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            key = (S[i], S[j])
            if key in pair_blocked:
                blocked |= pair_blocked[key]
    return blocked

def _get_pair_blocked(a, b, pair_blocked):
    """Get points c such that {a,b,c} is isosceles."""
    key = (a, b)
    if key in pair_blocked:
        return set(pair_blocked[key])
    return set()

def _is_maximal(S, pair_blocked):
    """Check if S is maximal: no point outside S can be added."""
    if not S:
        return False
    S_set = set(S)
    # For each point p not in S, check if it's blocked by some pair in S
    # A point p is blocked by pair (a,b) if p is in pair_blocked[(a,b)]
    # So we need: for every p not in S, there exists a pair (a,b) in S such that p in pair_blocked[(a,b)]
    
    # Collect all blocked points from pairs in S
    all_blocked = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            key = (S[i], S[j])
            if key in pair_blocked:
                all_blocked |= pair_blocked[key]
    
    # Check if every point not in S is in all_blocked
    for p in range(V):
        if p in S_set:
            continue
        if p not in all_blocked:
            return False
    return True

def _construct_direct(dist, points, pair_blocked, time_budget=280.0):
    """
    Direct construction: 
    1. Use a greedy approach with strong heuristic to minimize set size.
    2. Then local search to reduce further.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V  # worst case
    best_set = None
    
    # Strategy: Try multiple random starts and greedy builds
    rng = random.Random(42)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build: always pick the point that blocks the most new points
        S = []
        U = set(range(V))
        
        while U:
            # Find all candidates that can be added (not blocked by any pair in S)
            candidates = []
            for p in U:
                can_add = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        key = (S[i], S[j])
                        if key in pair_blocked and p in pair_blocked[key]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    candidates.append(p)
            
            if not candidates:
                break
            
            # Score each candidate by how many points it would block
            best_score = -1
            best_p = None
            best_blocked = set()
            
            # Shuffle for randomization
            cand_list = candidates[:]
            rng.shuffle(cand_list)
            
            for p in cand_list:
                # Compute how many points in U would be blocked by adding p
                new_blocked = set()
                for a in S:
                    key = (a, p)
                    if key in pair_blocked:
                        new_blocked |= pair_blocked[key]
                
                score = len(new_blocked & U)
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_blocked = new_blocked
                elif score == best_score and rng.random() < 0.3:
                    best_p = p
                    best_blocked = new_blocked
            
            if best_p is None:
                break
            
            S.append(best_p)
            U.remove(best_p)
            U -= best_blocked
        
        # Local search: try to remove points
        S = list(S)
        for it in range(50):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(len(S)):
                trial_S = S[:idx] + S[idx+1:]
                if not trial_S:
                    continue
                # Check validity
                if not _is_valid_set(trial_S, pair_blocked):
                    continue
                # Check maximality
                if _is_maximal(trial_S, pair_blocked):
                    S = trial_S
                    improved = True
                    break
            if not improved:
                break
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    if best_set is None:
        # Fallback: just return a valid set (even if not minimal)
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_blocked = _precompute()
    return _construct_direct(dist, points, pair_blocked)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
