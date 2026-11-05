import itertools
import time

def construct():
    # Target: n=25 using F5^2 construction
    # If we can find a valid partition, return it.
    # Otherwise, fallback to n=11 or try other n.
    
    # Time guard
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try n=25
    q = 5
    n = q * q
    
    # Generate all non-zero vectors in F_q^2
    vectors = []
    for x in range(q):
        for y in range(q):
            if x != 0 or y != 0:
                vectors.append((x, y))
                
    # Group into pairs {v, -v}
    # In F_q, -v is (q-x, q-y) mod q
    pairs = []
    used = set()
    for v in vectors:
        if v in used:
            continue
        neg_v = ((-v[0]) % q, (-v[1]) % q)
        if neg_v == v:
            # Only possible if 2x=0, 2y=0. In F5, only (0,0).
            pass
        pairs.append((v, neg_v))
        used.add(v)
        used.add(neg_v)
        
    # We have 12 pairs.
    # We need to assign each pair to one of 5 colors (0..4).
    # Let reps[i] be the first vector of the i-th pair.
    reps = [p[0] for p in pairs]
    
    # Precompute sums
    # For a set of indices S, check if valid.
    # Valid if for all i, j in S (i != j), 
    # reps[i] + reps[j] is not in S (as a vector or its negative)
    # and reps[i] - reps[j] is not in S
    
    def is_valid_assignment(assignment):
        # assignment is a list of 12 ints (0..4)
        # Group indices by color
        groups = {c: [] for c in range(5)}
        for i, c in enumerate(assignment):
            groups[c].append(i)
            
        for c in range(5):
            indices = groups[c]
            vecs = [reps[i] for i in indices]
            # Create a set of vectors and their negatives for fast lookup
            # Actually, we just need to check if sum is in the set of reps or their negs
            vec_set = set()
            for v in vecs:
                vec_set.add(v)
                vec_set.add((( -v[0]) % q, (-v[1]) % q))
                
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    v1 = vecs[i]
                    v2 = vecs[j]
                    # Check sum
                    s = ((v1[0] + v2[0]) % q, (v1[1] + v2[1]) % q)
                    if s in vec_set:
                        # If s is 0, it's not in vec_set (vec_set only has non-zero)
                        # But if s is in vec_set, it means s or -s is in the group.
                        # This forms a C4.
                        return False
                    # Check diff
                    d = ((v1[0] - v2[0]) % q, (v1[1] - v2[1]) % q)
                    if d in vec_set:
                        return False
        return True

    # Backtracking search
    # Heuristic: try to balance sizes? Or just DFS.
    # 12 items, 5 colors. 5^12 is too big for naive, but with pruning it's fast.
    
    solution = None
    
    def backtrack(idx, assignment):
        nonlocal solution
        if time.time() > deadline:
            return
        if idx == 12:
            if is_valid_assignment(assignment):
                solution = assignment[:]
                return
            return
        
        # Try colors 0..4
        for c in range(5):
            assignment[idx] = c
            # Pruning: check if adding idx to color c creates immediate conflict
            # with existing elements of color c
            # Get existing indices of color c
            existing = [i for i in range(idx) if assignment[i] == c]
            valid = True
            v_new = reps[idx]
            for i in existing:
                v_old = reps[i]
                s = ((v_new[0] + v_old[0]) % q, (v_new[1] + v_old[1]) % q)
                d = ((v_new[0] - v_old[0]) % q, (v_new[1] - v_old[1]) % q)
                # Check if s or d is in the existing set or its negative
                # We need to check against ALL existing vectors of color c
                # This is O(k) for each check, total O(k^2) per step. k<=12.
                # We can just use the full check on the partial assignment?
                # No, partial check is faster.
                
                # Let's just use the full is_valid on the partial assignment?
                # It's expensive.
                # Better: maintain the set of vectors for each color.
                # For now, just do the simple check:
                # If s or d is equal to any v in existing or -v in existing.
                
                # To keep it simple and robust, let's just call a lighter check
                # or rely on the final check if branching is small.
                # Branching factor 5. Depth 12.
                # With pruning, it should be very fast.
                
                # Let's check against existing vectors directly
                for j in existing:
                    v_j = reps[j]
                    neg_v_j = ((-v_j[0]) % q, (-v_j[1]) % q)
                    if s == v_j or s == neg_v_j or d == v_j or d == neg_v_j:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                backtrack(idx + 1, assignment)
                if solution:
                    return
                    
    assignment = [0] * 12
    backtrack(0, assignment)
    
    if solution:
        # Build the color list
        # Map pair index to color
        # For each edge (u, v), find diff d, find which pair it belongs to, get color.
        # Precompute map from vector to pair index
        vec_to_pair = {}
        for i, p in enumerate(pairs):
            vec_to_pair[p[0]] = i
            vec_to_pair[p[1]] = i
            
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                # u = (i//q, i%q), v = (j//q, j%q)
                u = (i // q, i % q)
                v = (j // q, j % q)
                d = ((v[0] - u[0]) % q, (v[1] - u[1]) % q)
                pair_idx = vec_to_pair[d]
                c = solution[pair_idx]
                colors.append(c)
                
        return {'n': n, 'colors': colors}
        
    # Fallback to n=11 if 25 fails (should not happen)
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
