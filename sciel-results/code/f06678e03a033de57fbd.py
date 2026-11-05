"""
Minimum maximal no-isosceles set on [0,N)^2.

Strategy: Greedy with Blocking Power Heuristic.
1. Precompute for each point p, the set of pairs (i,j) that form an isosceles triangle with p.
   We store this as a set of tuples (min(i,j), max(i,j)).
2. Maintain a set of selected points S and a set of candidate points C.
   Initially S is empty and C is all points.
3. While C is not empty:
   a. For each candidate p in C, calculate the size of the new candidate set C' that would remain
      if p were added to S.
      C' = { q in C \ {p} | q is valid with S U {p} }.
      A point q is valid with S U {p} if:
      - q is valid with S (guaranteed since q in C).
      - {q, p, u} is not isosceles for all u in S.
        This is equivalent to checking that no pair (p, u) is in BAD_PAIRS[q] for any u in S.
   b. Select the point p* in C that minimizes |C'|. This point "blocks" the most other points.
   c. Add p* to S.
   d. Update C = C' (calculated for p*).
4. The resulting set S is maximal (by construction, C becomes empty).
5. Perform multiple restarts with random tie-breaking to find the minimum size set.
"""

import random
import time
import itertools
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute bad pairs for each point
# BAD_PAIRS[p] is a set of (i, j) with i < j such that {p, i, j} is isosceles.
BAD_PAIRS = [set() for _ in range(N_PTS)]

# Also precompute a fast lookup structure:
# For a given q, and a given p, we want to know if there exists u in S such that (p, u) is in BAD_PAIRS[q].
# We can precompute for each q, a dictionary mapping p -> set of u's.
# BAD_PARTNERS[q][p] = set of u such that (min(p,u), max(p,u)) in BAD_PAIRS[q].
BAD_PARTNERS = [dict() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        for k in bad_indices:
            i = i_idx[k]
            j = j_idx[k]
            BAD_PAIRS[p].add((i, j))
    
    # Build BAD_PARTNERS
    for q in range(N_PTS):
        for (i, j) in BAD_PAIRS[q]:
            # Pair (i, j) is bad for q.
            # This means if i and j are both in S, q cannot be added.
            # In the greedy step, when we add p to S, we need to check if (p, u) is a bad pair for q for any u in S.
            # So for each bad pair (i, j) for q, we add j to BAD_PARTNERS[q][i] and i to BAD_PARTNERS[q][j].
            if i not in BAD_PARTNERS[q]:
                BAD_PARTNERS[q][i] = set()
            BAD_PARTNERS[q][i].add(j)
            
            if j not in BAD_PARTNERS[q]:
                BAD_PARTNERS[q][j] = set()
            BAD_PARTNERS[q][j].add(i)

_start_time = time.time()
_precompute_bad_pairs()

def _greedy_blocking_power(deadline):
    """
    Greedy algorithm that picks the point which reduces the candidate set the most.
    """
    S = []
    S_set = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_new_C_size = float('inf')
        best_new_C = None
        
        # Randomize order to break ties and explore different paths
        candidates = list(C)
        random.shuffle(candidates)
        
        for p in candidates:
            # Calculate new C if we add p
            new_C = set()
            for q in C:
                if q == p:
                    continue
                # Check if q is valid with S U {p}
                # q is in C, so it is valid with S.
                # We need to check if adding p creates any isosceles triangle with q and some u in S.
                # This is equivalent to checking if (p, u) is a bad pair for q for any u in S.
                # We precomputed BAD_PARTNERS[q][p] = set of u such that (p, u) is bad for q.
                # So we just check if BAD_PARTNERS[q].get(p, empty_set) intersects S_set.
                
                bad_us = BAD_PARTNERS[q].get(p, set())
                if bad_us & S_set:
                    continue # q is invalid
                else:
                    new_C.add(q)
            
            new_C_size = len(new_C)
            if new_C_size < best_new_C_size:
                best_new_C_size = new_C_size
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_set.add(best_p)
        C = best_new_C
        
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts
    # The greedy blocking power heuristic is deterministic in its choice (min size C'),
    # but we shuffle the candidates to break ties and explore different solutions.
    # We can also try to improve the solution by local search if we have time,
    # but for now, multiple random restarts of the greedy should be sufficient.
    
    # To increase diversity, we can slightly perturb the "blocking power" calculation
    # or just rely on the shuffle.
    
    while time.time() < deadline - 10.0:
        candidate = _greedy_blocking_power(deadline - 5.0)
        
        if candidate is None:
            continue
        
        # The greedy algorithm ensures validity and maximality by construction.
        # C becomes empty, so no point can be added.
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate
            
    if best_set is None:
        # Fallback: simple random greedy
        S = []
        S_set = set()
        C = set(range(N_PTS))
        while C:
            p = random.choice(list(C))
            S.append(p)
            S_set.add(p)
            # Update C
            new_C = set()
            for q in C:
                if q == p:
                    continue
                bad_us = BAD_PARTNERS[q].get(p, set())
                if bad_us & S_set:
                    continue
                else:
                    new_C.add(q)
            C = new_C
        best_set = S

    return [ALL_POINTS[i] for i in best_set]
