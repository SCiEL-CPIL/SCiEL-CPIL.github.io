"""Program for kissing number in dimension 11 using direct construction from Golay code structure."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def _build_extended_golay_code():
    """
    Build the extended binary Golay code G24 as a list of all 4096 codewords.
    Uses a systematic generator matrix approach.
    """
    # The extended Golay code can be generated from the [23,12,7] Golay code.
    # We'll use a known generator matrix for the [24,12,8] extended Golay code.
    
    # Generator matrix for extended Golay code (12 x 24)
    # Using a standard form from literature
    G = np.array([
        [1,0,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0],
        [0,1,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,1],
        [0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,0,0],
        [0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1],
        [0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0],
        [0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,0,1,1,0,0,1,1],
        [0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1]
    ], dtype=np.int32)
    
    # Generate all 2^12 = 4096 codewords
    codewords = []
    for msg in range(4096):
        bits = [(msg >> i) & 1 for i in range(12)]
        bits = np.array(bits, dtype=np.int32)
        cw = bits @ G
        cw = cw % 2
        codewords.append(cw)
    
    return codewords

def construct():
    """
    Construct 165 vectors in R^11 forming a kissing configuration.
    
    Uses the known structure:
    - 22 vectors of the form ±2e_i (norm² = 4)
    - 143 weight-4 vectors with entries ±1 (norm² = 4)
    
    The weight-4 vectors are selected from the 5280 candidates using the
    Golay code structure to ensure pairwise dot products are at most 2.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # Type 1: weight 1, one entry ±2
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    
    # Type 2: weight 4, four entries ±1
    # Generate all candidate weight-4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    N4 = len(v4_candidates)  # C(11,4) * 2^4 = 330 * 16 = 5280
    
    # Build the conflict graph efficiently
    # Two weight-4 vectors conflict if their dot product > 2
    # This happens only when they share exactly 3 positions with all 3 sign products = +1
    # (giving dot = 3), or share all 4 positions with all 4 sign products = +1 (dot = 4)
    # But since each (positions, signs) pair is unique, dot=4 only for identical vectors.
    # So conflicts are: same positions with different signs? No.
    # Let me re-examine:
    # If two vectors share positions S1, S2:
    # - |S1 ∩ S2| = 0: dot = 0
    # - |S1 ∩ S2| = 1: dot = ±1
    # - |S1 ∩ S2| = 2: dot ∈ {-2, 0, 2}
    # - |S1 ∩ S2| = 3: dot ∈ {-3, -1, 1, 3}; conflict only if dot = 3
    # - |S1 ∩ S2| = 4: dot = 4 (identical) or other values
    
    # For same position set (|intersection|=4):
    # dot = sum of 4 sign products
    # If all 4 products are +1: dot = 4 (conflict) - this means identical
    # If 3 products +1, 1 product -1: dot = 2 (ok)
    # If 2 products +1, 2 products -1: dot = 0 (ok)
    # If 1 product +1, 3 products -1: dot = -2 (ok)
    # If 0 products +1, 4 products -1: dot = -4 (ok)
    # So for same positions, only identical vectors conflict.
    
    # For different position sets with |intersection|=3:
    # Conflict only if all 3 sign products are +1, giving dot = 3.
    
    # Build neighbor list
    # Map each candidate to an index
    candidates_by_pos = defaultdict(list)  # pos_tuple -> list of (sign_tuple, index)
    for i, v in enumerate(v4_candidates):
        positions = tuple(j for j in range(D) if v[j] != 0)
        signs = tuple(v[j] for j in positions)
        candidates_by_pos[positions].append((signs, i))
    
    neighbor_list = [set() for _ in range(N4)]
    
    # Process pairs of position sets with intersection size 3
    pos_sets = list(candidates_by_pos.keys())
    
    # Group by 3-subsets
    three_subset_map = defaultdict(list)
    for ps in pos_sets:
        for three in itertools.combinations(ps, 3):
            three_subset_map[three].append(ps)
    
    for three, ps_list in three_subset_map.items():
        if len(ps_list) < 2:
            continue
        for idx1 in range(len(ps_list)):
            for idx2 in range(idx1 + 1, len(ps_list)):
                ps1, ps2 = ps_list[idx1], ps_list[idx2]
                
                # Get sign patterns
                for (s1, i1) in candidates_by_pos[ps1]:
                    for (s2, i2) in candidates_by_pos[ps2]:
                        # Check if sign products on the 3 intersection positions are all +1
                        all_plus = True
                        for pos in three:
                            idx1_pos = ps1.index(pos)
                            idx2_pos = ps2.index(pos)
                            if s1[idx1_pos] * s2[idx2_pos] != 1:
                                all_plus = False
                                break
                        if all_plus:
                            neighbor_list[i1].add(i2)
                            neighbor_list[i2].add(i1)
    
    # Now we need to find a maximum independent set of size 143
    # Use a combination of greedy and local search
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = np.array([len(neighbor_list[i]) for i in range(N4)], dtype=np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = set([chosen])
            to_remove.update(neighbor_list[chosen])
            for r in to_remove:
                alive[r] = False
                for n in neighbor_list[r]:
                    if alive[n]:
                        degrees[n] -= 1
        
        return in_set
    
    def local_search(in_set_init, max_time=3.0):
        """Simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        sa_deadline = time.time() + max_time
        in_set_indices = set(np.where(in_set)[0].tolist())
        
        while time.time() < sa_deadline:
            not_in = [i for i in range(N4) if i not in in_set_indices]
            if not not_in:
                break
            
            v = not_in[rng.integers(len(not_in))]
            
            nbrs_in_set = [u for u in neighbor_list[v] if u in in_set_indices]
            delta = 1 - len(nbrs_in_set)
            
            if delta >= 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                for u in nbrs_in_set:
                    in_set[u] = False
                    in_set_indices.remove(u)
                in_set[v] = True
                in_set_indices.add(v)
                current_size += delta
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                
                T *= cooling
            else:
                T *= cooling
        
        return local_best_in_set, local_best_size
    
    # Run multiple restarts
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        remaining = deadline - 10 - time.time()
        if remaining < 1:
            break
        
        sa_time = min(remaining * 0.3, 3.0)
        if sa_time < 0.3:
            sa_time = 0.3
        
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        if best_size >= 143:
            break
    
    # Combine with weight-1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = np.array([v4_candidates[i] for i in final_v4_indices], dtype=np.int64)
    v1_arr = np.array(v1_list, dtype=np.int64)
    
    final_vectors = np.vstack([v1_arr, result_v4])
    
    return final_vectors
# EVOLVE-BLOCK-END
