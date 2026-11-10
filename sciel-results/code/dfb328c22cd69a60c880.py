# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Start from Hermitian unital baseline, then use witness-count local search
    to maximize the complement S (points not in N).
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin before 600s limit

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)

    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401

    # Precompute lines for each direction
    # Direction 0: vertical v = (0,1)
    # Directions 1-49: v = (1, m) for m in F_49
    direction_lines = []

    # Vertical
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))

    # Slope directions: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))

    # For each point and direction, precompute the set of other points on that line
    # line_neighbors[pt][dir] = list of other point indices on the line through pt in direction dir
    line_neighbors = [[[] for _ in range(50)] for _ in range(n)]
    for dir_idx in range(50):
        for line in direction_lines[dir_idx]:
            for pt_idx in line:
                line_neighbors[pt_idx][dir_idx] = [y for y in line if y != pt_idx]

    # Baseline construction: Hermitian unital
    # Exclude points on the Hermitian curve where y is NOT in a special set
    # The Hermitian curve: Tr(x*^q) + Tr(y*^q) = 0 or similar
    # Using the formula from inspiration: 2*xa + ya^2 - k*yb^2 = 0 mod 7
    # with y in a special set (quadratic residues)
    
    # Quadratic residues in F_7
    qr = set((u*u) % p for u in range(p))
    
    in_N = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        # Hermitian unital condition
        if (2*xa + ya*ya - k*yb*yb) % p == 0:
            # Only keep points where ya is in QR (or some criterion)
            if ya not in qr:
                in_N[i] = False

    # in_S[i] = True if i is in S (complement, not in N)
    in_S = [not in_N[i] for i in range(n)]
    current_size = sum(in_S)

    # Compute witness counts
    # witness_count[i] = number of directions d such that (Line(i,d) \ {i}) ∩ S = empty
    # i.e., all other points on the line are in N
    
    def compute_witness_count(in_S):
        """Compute witness counts for a given in_S array."""
        witness_count = [0] * n
        for i in range(n):
            for d in range(50):
                has_S = False
                for y in line_neighbors[i][d]:
                    if in_S[y]:
                        has_S = True
                        break
                if not has_S:
                    witness_count[i] += 1
        return witness_count

    witness_count = compute_witness_count(in_S)

    # Verify baseline is valid
    if not all(w >= 1 for w in witness_count):
        # Fallback: if baseline is invalid, use full set
        in_S = [False] * n
        witness_count = [50] * n
        current_size = 0

    best_size = current_size
    best_S = in_S[:]
    best_witness = witness_count[:]

    def try_add_to_S(in_S, witness_count, idx):
        """Try to add idx to S. Returns True if successful, updates in_S and witness_count."""
        # Check if adding idx to S would make any point have witness_count 0
        for d in range(50):
            for y in line_neighbors[idx][d]:
                if witness_count[y] == 1:
                    return False
        # Safe to add
        in_S[idx] = True
        for d in range(50):
            for y in line_neighbors[idx][d]:
                witness_count[y] -= 1
        return True

    # Phase 1: Greedy single additions
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        candidates = [i for i in range(n) if not in_S[i]]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline - 30:
                break
            if try_add_to_S(in_S, witness_count, idx):
                current_size += 1
                improved = True

    if current_size > best_size:
        best_size = current_size
        best_S = in_S[:]
        best_witness = witness_count[:]

    # Phase 2: Try pair additions to escape local minima
    in_S = best_S[:]
    witness_count = best_witness[:]
    current_size = best_size

    not_in_S = [i for i in range(n) if not in_S[i]]
    random.shuffle(not_in_S)

    max_pair_attempts = min(len(not_in_S) * 10, 3000)
    pair_attempts = 0

    while pair_attempts < max_pair_attempts and time.time() < deadline - 15:
        if len(not_in_S) < 2:
            break
        i1, i2 = random.sample(not_in_S, 2)
        
        # Try adding i1 temporarily
        can_add_i1 = True
        for d in range(50):
            for y in line_neighbors[i1][d]:
                if witness_count[y] == 1:
                    can_add_i1 = False
                    break
            if not can_add_i1:
                break
        
        if can_add_i1:
            in_S[i1] = True
            for d in range(50):
                for y in line_neighbors[i1][d]:
                    witness_count[y] -= 1
            
            # Now check if we can add i2
            can_add_i2 = True
            for d in range(50):
                for y in line_neighbors[i2][d]:
                    if witness_count[y] == 1:
                        can_add_i2 = False
                        break
                if not can_add_i2:
                    break
            
            if can_add_i2:
                in_S[i2] = True
                for d in range(50):
                    for y in line_neighbors[i2][d]:
                        witness_count[y] -= 1
                current_size += 2
                not_in_S.remove(i1)
                not_in_S.remove(i2)
            else:
                # Revert i1
                in_S[i1] = False
                for d in range(50):
                    for y in line_neighbors[i1][d]:
                        witness_count[y] += 1
        pair_attempts += 1

    if current_size > best_size:
        best_size = current_size
        best_S = in_S[:]
        best_witness = witness_count[:]

    # Phase 3: More single additions after pair moves
    in_S = best_S[:]
    witness_count = best_witness[:]
    current_size = best_size

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        candidates = [i for i in range(n) if not in_S[i]]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline - 5:
                break
            if try_add_to_S(in_S, witness_count, idx):
                current_size += 1
                improved = True

    if current_size > best_size:
        best_size = current_size
        best_S = in_S[:]
        best_witness = witness_count[:]

    # Final verification
    final_witness = compute_witness_count(best_S)
    if not all(w >= 1 for w in final_witness):
        # Invalid, fallback to full set
        pts = all_points
    else:
        pts = [all_points[i] for i in range(n) if not best_S[i]]

    if not pts:
        pts = all_points

    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
