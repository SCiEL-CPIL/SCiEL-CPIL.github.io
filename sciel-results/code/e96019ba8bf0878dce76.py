import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    D = 11

    # The theoretical maximum independent set of 4-subsets with pairwise intersection <= 2
    # is 41 (each point appears in at most C(10,3)=120 total, but with |S∩T|<=2 constraint,
    # the max packing in dimension 11 gives 41 supports * 16 signs = 656 vectors).
    # We use a smarter approach: formulate as MIS and use vectorized greedy with submodular structure.

    supports4 = list(combinations(range(D), 4))
    N4 = len(supports4)  # 330
    sup4_sets = [set(s) for s in supports4]

    # Precompute conflict matrix: conflict if |S∩T| >= 3
    # This is symmetric; use bit masks for speed
    sup4_masks = []
    for s in supports4:
        mask = 0
        for i in s:
            mask |= (1 << i)
        sup4_masks.append(mask)

    # Precompute intersection sizes
    inter_sizes = np.zeros((N4, N4), dtype=np.int8)
    for i in range(N4):
        for j in range(N4):
            # Pop count of AND of masks
            common = sup4_masks[i] & sup4_masks[j]
            cnt = bin(common).count('1')
            inter_sizes[i, j] = cnt

    # Conflict: |S∩T| >= 3
    conflict = inter_sizes >= 3

    # Vectorized Greedy Set Cover / MIS approach with submodular gains
    def greedy_mis_vectorized(conflict, N, rng, deadline, iters_per_research=1):
        """
        Greedy MIS using submodular marginal gain concept.
        At each step, select the node with maximum number of non-conflicting unselected neighbors,
        weighted by their potential. This is the classic greedy for independent set.
        """
        available = np.ones(N, dtype=bool)
        selected = []
        
        while available.sum() > 0 and time.time() < deadline:
            # For each available node, count how many available nodes it conflicts with
            # We want to pick the node with FEWEST conflicts (minimum degree heuristic)
            # among available nodes
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Compute degree within available set for each available node
            # deg[i] = number of j in available such that conflict[i,j]
            deg = np.sum(conflict[:, :].T[avail_indices] & available[:, None].T, axis=-1) if False else None
            
            # More efficient: use matrix multiplication
            # deg[i] for i in available = sum over j in available of conflict[i,j]
            avail_deg = np.sum(conflict & available[:, None], axis=1)
            
            # Among available, find minimum degree
            mask_deg = avail_deg.copy()
            mask_deg[~available] = np.inf
            min_deg = np.min(mask_deg)
            
            # All nodes with minimum degree
            candidates = np.where(available & (avail_deg == min_deg))[0]
            u = rng.choice(candidates)
            
            selected.append(u)
            # Remove u and all its neighbors from available
            available[u] = False
            neighbors = conflict[u] & available
            available[neighbors] = False
        
        return selected

    def local_search_12swap(conflict, N, iset, rng, deadline, max_rounds=10):
        """
        1-for-2 swap local search: try to remove one element and add two non-conflicting ones.
        """
        iset = set(iset)
        iset_arr = np.zeros(N, dtype=bool)
        for x in iset:
            iset_arr[x] = True
        
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline:
            improved = False
            rounds += 1
            for v in list(iset):
                if time.time() > deadline:
                    break
                iset_arr[v] = False
                # Find all nodes not in iset_arr and not conflicting with iset_arr
                potential = ~iset_arr & ~conflict[v]  # not in current, not conflicting with v
                # Actually we need nodes that don't conflict with any in iset (minus v)
                current_mask = iset_arr.copy()
                # potential nodes: not in current, not conflicting with current
                potential = ~current_mask
                for x in range(N):
                    if current_mask[x]:
                        potential[x] = False
                    elif conflict[v, x] and False:
                        pass
                
                # potential: nodes not in iset (after removing v), and not conflicting with any node in iset (after removing v)
                # A node x is compatible if for all y in iset\{v}, not conflict[x,y]
                potential = np.ones(N, dtype=bool)
                for y in iset:
                    if y != v:
                        potential &= ~conflict[y]
                potential[v] = False  # can't re-add v immediately (well, we could but skip)
                
                # Find pairs in potential that don't conflict with each other
                pot_indices = np.where(potential)[0]
                if len(pot_indices) >= 2:
                    # Try to find two non-conflicting nodes
                    for a in range(min(20, len(pot_indices))):
                        for b in range(a + 1, min(20, len(pot_indices))):
                            x1, x2 = pot_indices[a], pot_indices[b]
                            if not conflict[x1, x2]:
                                iset.remove(v)
                                iset.add(x1)
                                iset.add(x2)
                                iset_arr = np.zeros(N, dtype=bool)
                                for x in iset:
                                    iset_arr[x] = True
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        return iset

    # Run multiple restarts with different random seeds
    best_iset4 = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        iset = greedy_mis_vectorized(conflict, N4, rng, deadline - 15.0)
        iset = set(iset)
        
        # Local search
        iset = local_search_12swap(conflict, N4, iset, rng, deadline - 15.0, max_rounds=5)
        
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset4 = iset.copy()
        
        # Early stop if we reach theoretical max
        if best_size >= 41:
            break
    
    # Generate all weight-4 vectors from the selected supports
    vectors = []
    for i in best_iset4:
        sup = supports4[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    # Step 2: Try to add weight-3 vectors that are compatible with all existing vectors
    supports3 = list(combinations(range(D), 3))
    
    # Precompute for speed: for each weight-4 support, we have all 16 sign variants
    # A weight-3 vector (T, signs) is compatible with weight-4 (S, signs4) iff dot <= 1
    # Group by support for efficiency
    sup_to_indices = {}
    for idx, v in enumerate(vectors):
        nz = tuple(sorted(np.nonzero(v)[0].tolist()))
        if nz not in sup_to_indices:
            sup_to_indices[nz] = []
        sup_to_indices[nz].append(idx)

    added_w3 = 0
    for T in supports3:
        if time.time() > deadline - 8.0:
            break
        T_set = set(T)
        for signs in product([1, -1], repeat=3):
            v3 = np.zeros(D, dtype=np.int8)
            for idx, s in zip(T, signs):
                v3[idx] = s

            compatible = True
            for S_tuple in sup_to_indices:
                S_set = set(S_tuple)
                inter = S_set & T_set
                m = len(inter)
                if m == 0:
                    continue
                for vec_idx in sup_to_indices[S_tuple]:
                    v4 = vectors[vec_idx]
                    uv = sum(int(v4[i]) * int(v3[i]) for i in inter)
                    if uv > 1:
                        compatible = False
                        break
                if not compatible:
                    break
            # Also check against already-added weight-3 vectors
            if compatible:
                for v3_existing in [v for v in vectors if len(v) == 3 if False]:
                    pass
                # Check against existing weight-3 in vectors
                for v in vectors:
                    nz = np.nonzero(v)[0]
                    if len(nz) != 3:
                        continue
                    inter2 = set(nz.tolist()) & T_set
                    uv = sum(int(v[i]) * int(v3[i]) for i in inter2)
                    # weight-3 vs weight-3: dist^2 = 3+3-2uv >= 4 => uv <= 1
                    if uv > 1:
                        compatible = False
                        break
            
            if compatible:
                vectors.append(v3)
                added_w3 += 1

    # Step 3: Try to add weight-2 vectors
    supports2 = list(combinations(range(D), 2))
    
    # Rebuild support index
    sup_to_indices = {}
    for idx, v in enumerate(vectors):
        nz = tuple(sorted(np.nonzero(v)[0].tolist()))
        if nz not in sup_to_indices:
            sup_to_indices[nz] = []
        sup_to_indices[nz].append(idx)

    added_w2 = 0
    for S2 in supports2:
        if time.time() > deadline - 4.0:
            break
        S2_set = set(S2)
        for signs in product([1, -1], repeat=2):
            v2 = np.zeros(D, dtype=np.int8)
            for idx, s in zip(S2, signs):
                v2[idx] = s

            compatible = True
            for v in vectors:
                nz = np.nonzero(v)[0]
                w = len(nz)
                inter = set(nz.tolist()) & S2_set
                if len(inter) == 0:
                    continue
                uv = sum(int(v[i]) * int(v2[i]) for i in inter)
                if w == 4:
                    if uv > 1:
                        compatible = False
                        break
                elif w == 3:
                    if uv > 0:
                        compatible = False
                        break
                elif w == 2:
                    if uv > 0:
                        compatible = False
                        break
                elif w == 1:
                    if uv > -1:
                        compatible = False
                        break
            if compatible:
                vectors.append(v2)
                added_w2 += 1

    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
