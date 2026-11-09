import random
import time
import math
from itertools import combinations

_START = time.time()
_LIMIT = 110.0  # Safety margin

# Precompute all possible triples and pair masks
ALL_TRIPLES = []
ALL_TRIPLE_MASKS = []
TRIPLE_PAIR_MASKS = []

pair_to_idx = {}
idx = 0
for i in range(16):
    for j in range(i + 1, 16):
        pair_to_idx[(i, j)] = idx
        idx += 1
NUM_PAIRS = idx  # 120

for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            ALL_TRIPLES.append((i, j, k))
            mask = (1 << i) | (1 << j) | (1 << k)
            ALL_TRIPLE_MASKS.append(mask)
            # Pair mask: which pairs this triple uses
            pmask = 0
            pmask |= (1 << pair_to_idx[(i, j)])
            pmask |= (1 << pair_to_idx[(i, k)])
            pmask |= (1 << pair_to_idx[(j, k)])
            TRIPLE_PAIR_MASKS.append(pmask)

NUM_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute for each triple, the list of triples that conflict with it (share a pair)
# This helps in quickly finding valid additions
TRIPLE_CONFLICTS = [set() for _ in range(NUM_TRIPLES)]
for i in range(NUM_TRIPLES):
    for j in range(NUM_TRIPLES):
        if i != j and (TRIPLE_PAIR_MASKS[i] & TRIPLE_PAIR_MASKS[j]):
            TRIPLE_CONFLICTS[i].add(j)

def _compute_alpha_fast(edges):
    """
    Compute independence number of hypergraph with 16 vertices.
    edges: list of ints (bitmasks)
    """
    n = 16
    all_vertices_mask = (1 << n) - 1
    
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in range(n):
            if e & (1 << v):
                vertex_edges[v].append(i)
                
    best = [0]
    
    def is_valid_add(v, S):
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            others = e ^ (1 << v)
            if (others & S) == others:
                return False
        return True

    def get_new_avail(S, v_bit, old_avail):
        new_S = S | v_bit
        new_avail = old_avail ^ v_bit
        final_avail = 0
        temp = new_avail
        while temp:
            u_bit = temp & -temp
            u = u_bit.bit_length() - 1
            temp ^= u_bit
            
            blocked = False
            for e_idx in vertex_edges[u]:
                e = edges[e_idx]
                rest = e ^ u_bit
                if (rest & new_S) == rest:
                    blocked = True
                    break
            
            if not blocked:
                final_avail |= u_bit
                
        return final_avail

    def dfs(S, avail):
        if S.bit_count() + avail.bit_count() <= best[0]:
            return
        
        if avail == 0:
            if S.bit_count() > best[0]:
                best[0] = S.bit_count()
            return
            
        v_bit = avail & -avail
        v = v_bit.bit_length() - 1
        
        dfs(S, avail ^ v_bit)
        
        if is_valid_add(v, S):
            new_avail = get_new_avail(S, v_bit, avail)
            dfs(S | v_bit, new_avail)
            
    dfs(0, all_vertices_mask)
    return best[0]

def _is_valid_triple_set(triples):
    """Check if the set of triples is linear (any two share at most 1 point)."""
    n = len(triples)
    for i in range(n):
        a, b, c = triples[i]
        for j in range(i + 1, n):
            x, y, z = triples[j]
            inter = 0
            if a == x or a == y or a == z: inter += 1
            if b == x or b == y or b == z: inter += 1
            if c == x or c == y or c == z: inter += 1
            if inter > 1:
                return False
    return True

def construct():
    rng = random.Random(42)
    
    def get_used_pairs(triples):
        pairs = set()
        for a, b, c in triples:
            pairs.add((a, b))
            pairs.add((a, c))
            pairs.add((b, c))
        return pairs

    def is_conflict(triple, used_pairs):
        a, b, c = triple
        if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
            return True
        return False

    def greedy_init():
        triples = []
        used_pairs = set()
        remaining = list(range(NUM_TRIPLES))
        rng.shuffle(remaining)
        
        for idx in remaining:
            t = ALL_TRIPLES[idx]
            if not is_conflict(t, used_pairs):
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                
        return triples

    def local_search(triples, time_budget):
        """
        Local search with multiple move types: add, remove, swap, and 2-opt.
        Uses pair masks for fast conflict checking.
        """
        current_triples = list(triples)
        current_masks = [ALL_TRIPLE_MASKS[ALL_TRIPLES.index(t)] for t in current_triples]
        current_alpha = _compute_alpha_fast(current_masks)
        best_triples = list(current_triples)
        best_alpha = current_alpha
        
        # Build pair mask for current set
        used_pairs_mask = 0
        for t in current_triples:
            idx = ALL_TRIPLES.index(t)
            used_pairs_mask |= TRIPLE_PAIR_MASKS[idx]
        
        T = 1.0
        cooling = 0.999
        start_time = time.time()
        
        while (time.time() - start_time) < time_budget:
            if time.time() - _START > _LIMIT:
                break
                
            move_type = rng.random()
            new_triples = None
            new_alpha = None
            delta = 0
            
            if move_type < 0.25:
                # Add a triple
                t_idx = rng.randrange(NUM_TRIPLES)
                t = ALL_TRIPLES[t_idx]
                if t in current_triples:
                    continue
                if (used_pairs_mask & TRIPLE_PAIR_MASKS[t_idx]) == 0:
                    new_triples = current_triples + [t]
                    new_masks = current_masks + [ALL_TRIPLE_MASKS[t_idx]]
                    new_alpha = _compute_alpha_fast(new_masks)
                    delta = new_alpha - current_alpha
            elif move_type < 0.5:
                # Remove a triple
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                t = current_triples[idx]
                t_idx = ALL_TRIPLES.index(t)
                new_triples = current_triples[:idx] + current_triples[idx+1:]
                new_masks = current_masks[:idx] + current_masks[idx+1:]
                new_alpha = _compute_alpha_fast(new_masks)
                delta = new_alpha - current_alpha
            elif move_type < 0.75:
                # Swap
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                t_remove = current_triples[idx]
                t_remove_idx = ALL_TRIPLES.index(t_remove)
                t_add_idx = rng.randrange(NUM_TRIPLES)
                t_add = ALL_TRIPLES[t_add_idx]
                if t_add_idx == t_remove_idx or t_add in current_triples:
                    continue
                
                temp_used = used_pairs_mask & ~TRIPLE_PAIR_MASKS[t_remove_idx]
                if (temp_used & TRIPLE_PAIR_MASKS[t_add_idx]) == 0:
                    new_triples = current_triples[:idx] + current_triples[idx+1:] + [t_add]
                    new_masks = current_masks[:idx] + current_masks[idx+1:] + [ALL_TRIPLE_MASKS[t_add_idx]]
                    new_alpha = _compute_alpha_fast(new_masks)
                    delta = new_alpha - current_alpha
            else:
                # 2-opt: remove two, add two
                if len(current_triples) < 2:
                    continue
                idx1 = rng.randrange(len(current_triples))
                idx2 = rng.randrange(len(current_triples))
                if idx1 == idx2:
                    continue
                
                # Remove both
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                t1_rem = current_triples[idx2]
                t2_rem = current_triples[idx1]
                t1_rem_idx = ALL_TRIPLES.index(t1_rem)
                t2_rem_idx = ALL_TRIPLES.index(t2_rem)
                
                temp_triples = current_triples[:idx2] + current_triples[idx2+1:idx1] + current_triples[idx1+1:]
                temp_used = used_pairs_mask & ~TRIPLE_PAIR_MASKS[t1_rem_idx] & ~TRIPLE_PAIR_MASKS[t2_rem_idx]
                
                # Add two random non-conflicting triples
                t1_add_idx = rng.randrange(NUM_TRIPLES)
                t1_add = ALL_TRIPLES[t1_add_idx]
                if t1_add in current_triples or (temp_used & TRIPLE_PAIR_MASKS[t1_add_idx]) != 0:
                    continue
                temp_used |= TRIPLE_PAIR_MASKS[t1_add_idx]
                
                t2_add_idx = rng.randrange(NUM_TRIPLES)
                t2_add = ALL_TRIPLES[t2_add_idx]
                if t2_add in current_triples or t2_add_idx == t1_add_idx or (temp_used & TRIPLE_PAIR_MASKS[t2_add_idx]) != 0:
                    continue
                
                new_triples = temp_triples + [t1_add, t2_add]
                new_masks = current_masks[:idx2] + current_masks[idx2+1:idx1] + current_masks[idx1+1:] + \
                           [ALL_TRIPLE_MASKS[t1_add_idx], ALL_TRIPLE_MASKS[t2_add_idx]]
                new_alpha = _compute_alpha_fast(new_masks)
                delta = new_alpha - current_alpha

            if new_triples is None:
                continue

            if delta < 0:
                current_triples = new_triples
                current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
                current_alpha = new_alpha
                # Update used_pairs_mask
                used_pairs_mask = 0
                for t in current_triples:
                    idx = ALL_TRIPLES.index(t)
                    used_pairs_mask |= TRIPLE_PAIR_MASKS[idx]
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            else:
                if T > 0:
                    if delta == 0:
                        if rng.random() < 0.5:
                            current_triples = new_triples
                            current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
                            current_alpha = new_alpha
                            used_pairs_mask = 0
                            for t in current_triples:
                                idx = ALL_TRIPLES.index(t)
                                used_pairs_mask |= TRIPLE_PAIR_MASKS[idx]
                    else:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            current_triples = new_triples
                            current_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples]
                            current_alpha = new_alpha
                            used_pairs_mask = 0
                            for t in current_triples:
                                idx = ALL_TRIPLES.index(t)
                                used_pairs_mask |= TRIPLE_PAIR_MASKS[idx]
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_triples, best_alpha

    def tabu_search(triples, time_budget):
        """
        Tabu search with a different neighborhood structure.
        """
        current_triples = list(triples)
        current_masks = [ALL_TRIPLE_MASKS[ALL_TRIPLES.index(t)] for t in current_triples]
        current_alpha = _compute_alpha_fast(current_masks)
        best_triples = list(current_triples)
        best_alpha = current_alpha
        
        tabu_list = []  # List of (triple, expiry)
        tabu_tenure = 20
        start_time = time.time()
        
        while (time.time() - start_time) < time_budget:
            if time.time() - _START > _LIMIT:
                break
            
            # Clean expired tabu entries
            now = time.time()
            tabu_list = [(t, exp) for t, exp in tabu_list if exp > now]
            
            # Generate neighbors
            neighbors = []
            
            # Try adding each possible triple
            used_pairs = get_used_pairs(current_triples)
            for t_idx in range(NUM_TRIPLES):
                t = ALL_TRIPLES[t_idx]
                if t in current_triples:
                    continue
                if (used_pairs_mask & TRIPLE_PAIR_MASKS[t_idx]) == 0:
                    # Check if t is tabu
                    if t in [tt for tt, _ in tabu_list]:
                        continue
                    new_triples = current_triples + [t]
                    new_masks = current_masks + [ALL_TRIPLE_MASKS[t_idx]]
                    new_alpha = _compute_alpha_fast(new_masks)
                    neighbors.append((new_alpha, new_triples, new_masks, t, 'add'))
            
            # Try removing each triple
            for idx, t in enumerate(current_triples):
                if t in [tt for tt, _ in tabu_list]:
                    continue
                new_triples = current_triples[:idx] + current_triples[idx+1:]
                new_masks = current_masks[:idx] + current_masks[idx+1:]
                new_alpha = _compute_alpha_fast(new_masks)
                neighbors.append((new_alpha, new_triples, new_masks, t, 'remove'))
            
            if not neighbors:
                break
            
            # Sort by alpha
            neighbors.sort(key=lambda x: x[0])
            
            # Pick best non-tabu move
            best_move = None
            for alpha, triples, masks, triple, move_type in neighbors:
                if triple not in [tt for tt, _ in tabu_list]:
                    best_move = (alpha, triples, masks, triple, move_type)
                    break
            
            if best_move is None:
                best_move = neighbors[0]
            
            alpha, new_triples, new_masks, triple, move_type = best_move
            
            # Accept if it improves or is the best available
            if alpha <= current_alpha or len(neighbors) == 1:
                current_triples = new_triples
                current_masks = new_masks
                current_alpha = alpha
                
                # Update used_pairs_mask
                used_pairs_mask = 0
                for t in current_triples:
                    idx = ALL_TRIPLES.index(t)
                    used_pairs_mask |= TRIPLE_PAIR_MASKS[idx]
                
                # Add to tabu list
                tabu_list.append((triple, time.time() + tabu_tenure * 0.01))
                
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
        
        return best_triples, best_alpha

    best_global_alpha = 16
    best_global_triples = None
    
    start_time = time.time()
    restarts = 0
    while (time.time() - start_time) < (_LIMIT - 10):
        init_triples = greedy_init()
        if not init_triples:
            continue
        
        # Alternate between local search and tabu search
        if restarts % 2 == 0:
            budget = min(5.0, max(1.0, _LIMIT - 15 - (time.time() - start_time)))
            opt_triples, alpha = local_search(init_triples, budget)
        else:
            budget = min(5.0, max(1.0, _LIMIT - 15 - (time.time() - start_time)))
            opt_triples, alpha = tabu_search(init_triples, budget)
        
        if alpha < best_global_alpha:
            best_global_alpha = alpha
            best_global_triples = opt_triples
            
        restarts += 1
        if best_global_alpha <= 5:
            break

    if best_global_triples is None:
        best_global_triples = greedy_init()
        if not _is_valid_triple_set(best_global_triples):
            best_global_triples = []
            
    return best_global_triples

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
    print(f"Alpha: {_compute_alpha_fast([(1<<a)|(1<<b)|(1<<c) for a,b,c in res])}")
