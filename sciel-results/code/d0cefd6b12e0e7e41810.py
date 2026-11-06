import time
import random
import numpy as np
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N = len(ALL_TRIPLES)  # 560

TRIPLE_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute witnesses as NumPy arrays
# Each witness is a bitset of 560 bits.
# We represent a bitset as an array of 9 uint64s.
NUM_WORDS = (N + 63) // 64  # 9 words

def _build_witnesses():
    """Precompute witness masks for each triple as NumPy arrays."""
    # witnesses[i] will be a 2D numpy array of shape (num_witnesses_for_i, NUM_WORDS)
    witnesses_list = [[] for _ in range(N)]
    
    # Iterate over all 6-subsets of vertices
    for verts in combinations(range(V), 6):
        # 15 matchings
        matchings = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)],
        ]
        
        for m in matchings:
            p1, p2, p3 = m
            v1, v2 = verts[p1[0]], verts[p1[1]]
            v3, v4 = verts[p2[0]], verts[p2[1]]
            v5, v6 = verts[p3[0]], verts[p3[1]]
            
            eight_triples = []
            for a in (v1, v2):
                for b in (v3, v4):
                    for c in (v5, v6):
                        eight_triples.append(tuple(sorted((a, b, c))))
            
            # For each triple in this K(2,2,2), the other 7 are witnesses
            for i, t in enumerate(eight_triples):
                idx = TRIPLE_IDX[t]
                # Create bitmask of the other 7
                # We need to convert the 7 triple indices to a bitset
                mask_words = np.zeros(NUM_WORDS, dtype=np.uint64)
                for j, t2 in enumerate(eight_triples):
                    if i != j:
                        idx2 = TRIPLE_IDX[t2]
                        word_idx = idx2 // 64
                        bit_idx = idx2 % 64
                        mask_words[word_idx] |= (1 << bit_idx)
                witnesses_list[idx].append(mask_words)

    # Convert to numpy arrays for each triple
    # If a triple has no witnesses, it's an empty array
    witnesses_np = []
    for i in range(N):
        if witnesses_list[i]:
            witnesses_np.append(np.array(witnesses_list[i], dtype=np.uint64))
        else:
            witnesses_np.append(np.empty((0, NUM_WORDS), dtype=np.uint64))
            
    return witnesses_np

WITNESSES = _build_witnesses()

def _can_add_np(S_np, t_idx):
    """Check if t_idx can be added to S_np (NumPy array of uint64s)."""
    W = WITNESSES[t_idx]
    if W.size == 0:
        return True
    # (S_np & W) != W for all rows in W
    # S_np shape: (9,)
    # W shape: (K, 9)
    # S_np[:, np.newaxis] & W -> (K, 9)
    # Check if any word in any row is equal to the corresponding word in W
    # Actually, we want to check if (S & W) == W for ANY witness.
    # If (S & W) == W, then W is a subset of S.
    # We want to ensure NO witness is a subset of S.
    # So we check: if np.any(np.all((S_np[:, np.newaxis] & W) == W, axis=1)): return False
    # This is vectorized.
    
    # Optimization: 
    # diff = (S_np[:, np.newaxis] & W) ^ W
    # If any row is all zeros, then that witness is present.
    # np.any(np.all(diff == 0, axis=1))
    
    diff = (S_np[:, np.newaxis] & W) ^ W
    return not np.any(np.all(diff == 0, axis=1))

def _greedy_np(deadline):
    """Greedily add triples using NumPy validity check."""
    S_np = np.zeros(NUM_WORDS, dtype=np.uint64)
    selected = [False] * N
    result = []
    
    # Random order
    order = list(range(N))
    random.shuffle(order)
    
    for t_idx in order:
        if time.time() >= deadline:
            break
        if selected[t_idx]:
            continue
        if _can_add_np(S_np, t_idx):
            # Add
            word_idx = t_idx // 64
            bit_idx = t_idx % 64
            S_np[word_idx] |= (1 << bit_idx)
            selected[t_idx] = True
            result.append(list(ALL_TRIPLES[t_idx]))
            
    return result, S_np

def _local_search_np(S_np, selected, result, deadline, max_iters=10000):
    """
    Simulated Annealing / Local Search.
    S_np: NumPy array state
    selected: list of bools
    result: list of triples
    """
    best_S_np = S_np.copy()
    best_selected = selected[:]
    best_result = result[:]
    best_len = len(result)
    
    current_len = len(result)
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    for iteration in range(max_iters):
        if time.time() >= deadline:
            break
            
        # Choose move type
        r = random.random()
        if r < 0.4:
            # Add
            cand = random.randrange(N)
            if not selected[cand]:
                if _can_add_np(S_np, cand):
                    # Accept
                    word_idx = cand // 64
                    bit_idx = cand % 64
                    S_np[word_idx] |= (1 << bit_idx)
                    selected[cand] = True
                    result.append(list(ALL_TRIPLES[cand]))
                    current_len += 1
                    if current_len > best_len:
                        best_len = current_len
                        best_S_np = S_np.copy()
                        best_selected = selected[:]
                        best_result = result[:]
                else:
                    # Reject
                    pass
        elif r < 0.8:
            # Swap
            if current_len > 0:
                # Pick random old
                old_idx = random.randrange(N)
                while not selected[old_idx]:
                    old_idx = random.randrange(N)
                
                # Pick random new
                new_idx = random.randrange(N)
                while selected[new_idx]:
                    new_idx = random.randrange(N)
                
                # Tentatively remove old
                word_idx_old = old_idx // 64
                bit_idx_old = old_idx % 64
                S_np[word_idx_old] &= ~(1 << bit_idx_old)
                selected[old_idx] = False
                # Remove from result? We can just update best_result at the end or keep result in sync.
                # Keeping result in sync is expensive (pop). Let's just track S_np and selected.
                # We reconstruct result at the end.
                
                # Check new
                if _can_add_np(S_np, new_idx):
                    # Accept swap (size unchanged)
                    word_idx_new = new_idx // 64
                    bit_idx_new = new_idx % 64
                    S_np[word_idx_new] |= (1 << bit_idx_new)
                    selected[new_idx] = True
                    # Size is same, so best_len doesn't change
                else:
                    # Reject, restore old
                    S_np[word_idx_old] |= (1 << bit_idx_old)
                    selected[old_idx] = True
        else:
            # 2-Swap
            if current_len >= 2:
                # Pick 2 old
                old1, old2 = -1, -1
                while old1 == -1:
                    old1 = random.randrange(N)
                    if not selected[old1]: old1 = -1
                while old2 == -1:
                    old2 = random.randrange(N)
                    if not selected[old2] or old2 == old1: old2 = -1
                    
                # Pick 2 new
                new1, new2 = -1, -1
                while new1 == -1:
                    new1 = random.randrange(N)
                    if selected[new1]: new1 = -1
                while new2 == -1:
                    new2 = random.randrange(N)
                    if selected[new2] or new2 == new1: new2 = -1
                    
                # Remove old1, old2
                w1, b1 = old1 // 64, old1 % 64
                w2, b2 = old2 // 64, old2 % 64
                S_np[w1] &= ~(1 << b1)
                S_np[w2] &= ~(1 << b2)
                selected[old1] = False
                selected[old2] = False
                
                # Check new1
                if _can_add_np(S_np, new1):
                    # Add new1
                    wn1, bn1 = new1 // 64, new1 % 64
                    S_np[wn1] |= (1 << bn1)
                    selected[new1] = True
                    
                    # Check new2
                    if _can_add_np(S_np, new2):
                        # Add new2
                        wn2, bn2 = new2 // 64, new2 % 64
                        S_np[wn2] |= (1 << bn2)
                        selected[new2] = True
                        # Accept 2-swap
                    else:
                        # Reject, restore
                        S_np[wn1] &= ~(1 << bn1)
                        selected[new1] = False
                        S_np[w1] |= (1 << b1)
                        S_np[w2] |= (1 << b2)
                        selected[old1] = True
                        selected[old2] = True
                else:
                    # Reject, restore
                    S_np[w1] |= (1 << b1)
                    S_np[w2] |= (1 << b2)
                    selected[old1] = True
                    selected[old2] = True
                    
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Maybe restart temperature or just continue with low T
            # For SA, we might want to reheat occasionally?
            # Let's just keep cooling.
            
    # Reconstruct best result from best_selected
    best_result_final = [list(ALL_TRIPLES[i]) for i in range(N) if best_selected[i]]
    return best_result_final, best_S_np, best_selected, best_len

def _full_validation(triples):
    """Check that no K(2,2,2) exists in the triple set."""
    tset = set(tuple(sorted(t)) for t in triples)
    for verts in combinations(range(V), 6):
        matchings = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)],
        ]
        for m in matchings:
            p1, p2, p3 = m
            v1, v2 = verts[p1[0]], verts[p1[1]]
            v3, v4 = verts[p2[0]], verts[p2[1]]
            v5, v6 = verts[p3[0]], verts[p3[1]]
            
            eight = [
                tuple(sorted((v1, v3, v5))), tuple(sorted((v1, v3, v6))),
                tuple(sorted((v1, v4, v5))), tuple(sorted((v1, v4, v6))),
                tuple(sorted((v2, v3, v5))), tuple(sorted((v2, v3, v6))),
                tuple(sorted((v2, v4, v5))), tuple(sorted((v2, v4, v6))),
            ]
            if all(t in tset for t in eight):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_result = None
    best_len = -1
    best_S_np = None
    best_selected = None
    
    # Multiple restarts
    # Each restart: Greedy -> SA
    # We have 145s.
    # Let's do 5 restarts of 20s each? Or dynamic?
    # Greedy is fast. SA is slow.
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        restart_deadline = start_time + (r + 1) * time_per_restart
        if time.time() >= restart_deadline - 2: # Safety
            continue
            
        # Greedy
        res, S_np = _greedy_np(restart_deadline - 10)
        selected = [False] * N
        for t in res:
            idx = TRIPLE_IDX[tuple(sorted(t))]
            selected[idx] = True
            
        if len(res) > best_len:
            best_len = len(res)
            best_result = res
            best_S_np = S_np
            best_selected = selected
            
        # SA
        if time.time() < restart_deadline - 2:
            improved_res, improved_S_np, improved_sel, improved_len = _local_search_np(
                S_np, selected, res, restart_deadline - 2, max_iters=50000
            )
            if improved_len > best_len:
                best_len = improved_len
                best_result = improved_res
                best_S_np = improved_S_np
                best_selected = improved_sel

    # Final validation
    if best_result is None:
        # Fallback
        best_result = [list(t) for t in [(0, a, b) for a in range(1, V) for b in range(a + 1, V)]]
    elif not _full_validation(best_result):
        # Should not happen
        pass

    return best_result
