import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute combinatorial structures for n up to 34
    # We will try n from high to low, or low to high?
    # The task says "Maximize n". The published construction is 34.
    # We should aim for 34. If we fail, we return the best valid n found.
    
    # Since n=34 is the target, we focus our search there.
    # If we find a valid coloring for 34, we return it.
    # If not, we try 33, etc.
    
    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Incidence: which 4-sets and 5-sets contain each triple
        quad_triples = []   # list of 4 triple-indices per 4-set
        for quad in combinations(range(n), 4):
            # For a quad (a,b,c,d), the triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            idxs = [
                tri_to_idx[(quad[0], quad[1], quad[2])],
                tri_to_idx[(quad[0], quad[1], quad[3])],
                tri_to_idx[(quad[0], quad[2], quad[3])],
                tri_to_idx[(quad[1], quad[2], quad[3])]
            ]
            quad_triples.append(idxs)
        quint_triples = []  # list of 10 triple-indices per 5-set
        for quint in combinations(range(n), 5):
            idxs = []
            for t in combinations(quint, 3):
                idxs.append(tri_to_idx[t])
            quint_triples.append(idxs)

        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)

        # For each triple, the 4-sets and 5-sets it belongs to
        triple_to_4sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_4sets[t].append(qidx)
        triple_to_5sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_5sets[t].append(qidx)
        triple_to_4sets = [np.array(x, dtype=np.int32) for x in triple_to_4sets]
        triple_to_5sets = [np.array(x, dtype=np.int32) for x in triple_to_5sets]

        # Bit masks for fast violation checking
        nwords = (M + 63) // 64
        quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        rng = np.random.RandomState(seed)

        # Modular heuristic initialization
        colors = np.zeros(M, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            # This specific modular coloring is known to be a good starting point
            colors[i] = 0 if (a + b + c) % 3 == 0 else 1

        # Build red/blue bitsets
        red_set = np.zeros(nwords, dtype=np.uint64)
        blue_set = np.zeros(nwords, dtype=np.uint64)
        for t in range(M):
            if colors[t] == 0:
                red_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
            else:
                blue_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        def count_violations(red_set, blue_set):
            r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
            b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
            return int(r4) + int(b5)

        def get_violating_quads(red_set):
            return np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]

        def get_violating_quints(blue_set):
            return np.where(np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1))[0]

        def flip_score(t, red_set, blue_set):
            old = colors[t]
            if old == 0:
                # Flipping from Red to Blue
                # Breaks red K4s containing t
                qmask = quad_masks[triple_to_4sets[t]]
                broken = np.all((qmask & red_set[None, :]) == qmask, axis=1).sum()
                # Creates blue K5s containing t (where other 9 triples are already blue)
                qmask = quint_masks[triple_to_5sets[t]]
                without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                created = np.all((without_t & blue_set[None, :]) == without_t, axis=1).sum()
                return int(broken) - int(created)
            else:
                # Flipping from Blue to Red
                # Breaks blue K5s containing t
                qmask = quint_masks[triple_to_5sets[t]]
                broken = np.all((qmask & blue_set[None, :]) == qmask, axis=1).sum()
                # Creates red K4s containing t (where other 3 triples are already red)
                qmask = quad_masks[triple_to_4sets[t]]
                without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                created = np.all((without_t & red_set[None, :]) == without_t, axis=1).sum()
                return int(broken) - int(created)

        def apply_flip(t, red_set, blue_set):
            old = colors[t]
            bit = np.uint64(1) << np.uint64(t & 63)
            w = t >> 6
            if old == 0:
                red_set[w] &= ~bit
                blue_set[w] |= bit
                colors[t] = 1
            else:
                blue_set[w] &= ~bit
                red_set[w] |= bit
                colors[t] = 0
            return old

        # Local Search: Greedy + Simulated Annealing
        # Phase 1: Greedy Violation Elimination
        max_phase1_iter = 10000
        for it in range(max_phase1_iter):
            if time.time() > start_time + time_limit:
                break

            viol_quads = get_violating_quads(red_set)
            viol_quints = get_violating_quints(blue_set)

            if len(viol_quads) == 0 and len(viol_quints) == 0:
                return colors.tolist()

            # Collect candidate triples from violating cliques
            candidate_set = set()
            for q in viol_quads:
                for t in quad_triples[q]:
                    candidate_set.add(int(t))
            for q in viol_quints:
                for t in quint_triples[q]:
                    candidate_set.add(int(t))

            if not candidate_set:
                break

            # Score each candidate flip
            best_t = -1
            best_s = 0
            for t in candidate_set:
                s = flip_score(t, red_set, blue_set)
                if s > best_s:
                    best_s = s
                    best_t = t

            if best_t == -1:
                # No single flip helps; break to SA phase
                break

            apply_flip(best_t, red_set, blue_set)

        # Phase 2: Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        while time.time() < start_time + time_limit:
            t = int(rng.randint(0, M))
            s = flip_score(t, red_set, blue_set)
            apply_flip(t, red_set, blue_set)
            
            # Accept if improvement or with probability exp(-s/T)
            if s < 0 or (T > min_T and rng.random() < np.exp(-s / T)):
                pass
            else:
                # Reject: flip back
                apply_flip(t, red_set, blue_set)
            
            # Check for solution
            if count_violations(red_set, blue_set) == 0:
                return colors.tolist()
            
            T *= cooling
            if T < min_T:
                # Reheat or restart? Let's just restart SA with new random init if stuck
                # But for now, just keep going or break if time is up
                T = min_T

        return None

    # Fallback valid small coloring
    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = fallback(9)

    # Strategy: Try n=34 first with multiple restarts. If successful, return.
    # If not, try n=33, etc.
    # Given the difficulty, we should allocate most time to n=34.
    
    targets = [34, 33, 32, 31, 30]
    
    for n in targets:
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time based on importance
        if n == 34:
            time_budget = min(remaining - 5.0, 90.0)
            n_restarts = 10
        elif n == 33:
            time_budget = min(remaining - 5.0, 30.0)
            n_restarts = 5
        else:
            time_budget = min(remaining - 5.0, 10.0)
            n_restarts = 3
            
        per_restart = time_budget / n_restarts
        
        found = None
        for r in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            t_limit = min(per_restart, deadline - time.time() - 2.0)
            if t_limit <= 0.5:
                break
            res = search_for_n(n, t_limit, seed=n * 1000 + r * 13 + 7)
            if res is not None:
                found = res
                break
        
        if found is not None:
            best_n = n
            best_colors = found
            # If we found 34, we can stop as it's the max known/published
            if n == 34:
                break
        else:
            # If we failed for n, we might still have a valid smaller n from previous iterations
            # But the loop structure above updates best_n only on success.
            # We should ensure we keep the best valid one found so far.
            # The current logic keeps best_n from previous successful iterations.
            pass

    return {'n': best_n, 'colors': best_colors}
