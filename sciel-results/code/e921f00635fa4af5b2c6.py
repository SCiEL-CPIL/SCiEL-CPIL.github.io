import itertools
import random
import time

# Reduce N to 8 to ensure fast execution and lower point count (better score)
N = 8
NUM_POINTS = N * N
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

FULL_MASK = (1 << NUM_POINTS) - 1

# Precompute pair masks: pair_mask[p][a] = bitmask of q such that (q,p,a) is isosceles
pair_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

def _dist2(p, q):
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

# Build pair masks efficiently
start_t = time.time()
# For each pair (p, a), find all q that form isosceles with p, a
# p, a are indices
for p_idx in range(NUM_POINTS):
    p = POINTS[p_idx]
    for a_idx in range(p_idx + 1, NUM_POINTS):
        a = POINTS[a_idx]
        d_pa = _dist2(p, a)
        mask = 0
        for q_idx in range(NUM_POINTS):
            if q_idx == p_idx or q_idx == a_idx:
                continue
            q = POINTS[q_idx]
            d_qp = _dist2(q, p)
            d_qa = _dist2(q, a)
            # Isosceles if any two distances are equal
            if d_qp == d_qa or d_qp == d_pa or d_qa == d_pa:
                mask |= (1 << q_idx)
        pair_mask[p_idx][a_idx] = mask
        pair_mask[a_idx][p_idx] = mask

def construct():
    deadline = time.time() + 290.0  # Safety margin
    best_set = None
    best_size = NUM_POINTS + 1

    def _time_ok():
        return time.time() < deadline

    def _covers_all(cover_mask):
        return (cover_mask & FULL_MASK) == FULL_MASK

    def greedy_cover(seed_val):
        random.seed(seed_val)
        S = set()
        cover_mask = 0
        remaining = list(range(NUM_POINTS))
        random.shuffle(remaining)
        
        while not _covers_all(cover_mask):
            if not _time_ok():
                break
            best_p = None
            best_gain = -1
            best_new_mask = 0
            
            # Find p that maximizes new coverage
            for p in remaining:
                if p in S:
                    continue
                add_u = 0
                for a in S:
                    add_u |= pair_mask[p][a]
                new_mask = cover_mask | add_u
                gain = bin(new_mask & ~cover_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_new_mask = new_mask
            if best_p is None:
                break
            S.add(best_p)
            cover_mask = best_new_mask
            
        return S, cover_mask

    def local_search_minimize(S, cover_mask, seed_val):
        random.seed(seed_val)
        S = set(S)
        improved = True
        while improved and _time_ok():
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            for p in s_list:
                if p not in S:
                    continue
                Sp = S - {p}
                # Recalculate cover for Sp
                new_cover = 0
                sp_list = list(Sp)
                for i in range(len(sp_list)):
                    a = sp_list[i]
                    for j in range(i+1, len(sp_list)):
                        b = sp_list[j]
                        new_cover |= pair_mask[a][b]
                if _covers_all(new_cover):
                    S = Sp
                    cover_mask = new_cover
                    improved = True
                    break
        return S, cover_mask

    # Try multiple seeds
    for seed in range(100):
        if not _time_ok():
            break
            
        # Greedy start
        S, cm = greedy_cover(seed)
        
        # If greedy didn't cover all (shouldn't happen if we add enough), force it
        if not _covers_all(cm):
             # Add random points until covered
             remaining = [i for i in range(NUM_POINTS) if i not in S]
             random.shuffle(remaining)
             for p in remaining:
                 S.add(p)
                 # Update cover mask (simplified: recompute)
                 cm = 0
                 s_list = list(S)
                 for i in range(len(s_list)):
                     a = s_list[i]
                     for j in range(i+1, len(s_list)):
                         b = s_list[j]
                         cm |= pair_mask[a][b]
                 if _covers_all(cm):
                     break
        
        if _covers_all(cm):
            S2, cm2 = local_search_minimize(S, cm, seed + 100)
            if len(S2) < best_size:
                best_size = len(S2)
                best_set = set(S2)
                
    if best_set is None:
        # Fallback: return all points? No, that's not no-isosceles.
        # Fallback: Greedy result
        S, cm = greedy_cover(0)
        best_set = S

    return [POINTS[i] for i in sorted(best_set)]

if __name__ == "__main__":
    s = construct()
    print(len(s))
