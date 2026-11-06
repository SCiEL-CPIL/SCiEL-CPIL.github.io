"""K(2,2,2)-free 3-graph on 16 vertices via precomputed conflict counting."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

_triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute all K(2,2,2) configurations
# Each config is a set of 8 triple indices
_config_triples = []       # list of frozensets of triple indices
_triple_to_configs = [[] for _ in range(N_TRIPLES)]  # for each triple, which configs it's in

for points in combinations(range(V), 6):
    p0, p1, p2, p3, p4, p5 = points
    pairings = [
        ((p0,p1),(p2,p3),(p4,p5)),
        ((p0,p1),(p2,p4),(p3,p5)),
        ((p0,p1),(p2,p5),(p3,p4)),
        ((p0,p2),(p1,p3),(p4,p5)),
        ((p0,p2),(p1,p4),(p3,p5)),
        ((p0,p2),(p1,p5),(p3,p4)),
        ((p0,p3),(p1,p2),(p4,p5)),
        ((p0,p3),(p1,p4),(p2,p5)),
        ((p0,p3),(p1,p5),(p2,p4)),
        ((p0,p4),(p1,p2),(p3,p5)),
        ((p0,p4),(p1,p3),(p2,p5)),
        ((p0,p4),(p1,p5),(p2,p3)),
        ((p0,p5),(p1,p2),(p3,p4)),
        ((p0,p5),(p1,p3),(p2,p4)),
        ((p0,p5),(p1,p4),(p2,p3)),
    ]
    for pair1, pair2, pair3 in pairings:
        trans = []
        for x in pair1:
            for y in pair2:
                for z in pair3:
                    trans.append(tuple(sorted((x, y, z))))
        indices = frozenset(_triple_to_idx[t] for t in trans)
        cfg_idx = len(_config_triples)
        _config_triples.append(indices)
        for idx in indices:
            _triple_to_configs[idx].append(cfg_idx)

N_CONFIGS = len(_config_triples)

def _is_addable(t_idx, counts):
    """Check if triple t_idx can be added without completing any K(2,2,2)."""
    for c in _triple_to_configs[t_idx]:
        if counts[c] == 7:
            return False
    return True

def _greedy(order, deadline):
    """Greedy construction. Returns (count, active_bool, counts)."""
    active = [False] * N_TRIPLES
    counts = [0] * N_CONFIGS
    cnt = 0
    for t_idx in order:
        if time.time() >= deadline:
            break
        if _is_addable(t_idx, counts):
            active[t_idx] = True
            cnt += 1
            for c in _triple_to_configs[t_idx]:
                counts[c] += 1
    return cnt, active, counts

def _local_search(active, counts, deadline, max_rounds=50):
    """Local search: try adding skipped triples, then try 1-for-1 swaps."""
    active = list(active)
    counts = list(counts)
    cnt = sum(active)
    
    for _ in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        
        # Phase 1: try adding any inactive triple
        for t_idx in range(N_TRIPLES):
            if active[t_idx]:
                continue
            if _is_addable(t_idx, counts):
                active[t_idx] = True
                cnt += 1
                for c in _triple_to_configs[t_idx]:
                    counts[c] += 1
                improved = True
        
        if not improved:
            break
        
        # Phase 2: try 1-for-1 swaps
        # For each inactive triple t, find an active triple t' in all blocking configs
        # such that removing t' and adding t keeps all counts <= 7
        if time.time() >= deadline:
            break
        
        # Precompute for speed: for each config, which triples are in it
        # We'll use a lighter approach: sample swaps
        inactive = [i for i in range(N_TRIPLES) if not active[i]]
        random.shuffle(inactive)
        
        for t_idx in inactive[:200]:  # limit swaps tried per round
            if time.time() >= deadline:
                break
            
            # Find blocking configs
            blockers = []
            for c in _triple_to_configs[t_idx]:
                if counts[c] == 7:
                    blockers.append(c)
            
            if not blockers:
                # Should have been added in phase 1, but just in case
                active[t_idx] = True
                cnt += 1
                for c in _triple_to_configs[t_idx]:
                    counts[c] += 1
                improved = True
                continue
            
            # Find candidate t' to remove: must be in ALL blocker configs
            # and removing it must not break any config that t_idx also affects
            # Actually simpler: t' must be in intersection of all blocker configs
            # and t' must be active
            
            # Get intersection of blocker configs' triple sets
            candidate_set = None
            for c in blockers:
                cfg_set = set(_config_triples[c])
                if candidate_set is None:
                    candidate_set = cfg_set
                else:
                    candidate_set &= cfg_set
            
            if not candidate_set:
                continue
            
            # Filter to active triples
            candidates = [t for t in candidate_set if active[t]]
            if not candidates:
                continue
            
            random.shuffle(candidates)
            for t_prime in candidates[:10]:
                # Check: after removing t_prime and adding t_idx, all counts <= 7
                # Configs affected by t_prime: _triple_to_configs[t_prime]
                # Configs affected by t_idx: _triple_to_configs[t_idx]
                # For each config in union, check new count <= 7
                
                valid = True
                # Check configs that contain t_prime
                for c in _triple_to_configs[t_prime]:
                    new_count = counts[c] - 1
                    # If t_idx is also in config c, new_count += 1
                    if c in _triple_to_configs[t_idx]:
                        new_count += 1
                    if new_count > 7:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    active[t_prime] = False
                    active[t_idx] = True
                    for c in _triple_to_configs[t_prime]:
                        counts[c] -= 1
                    for c in _triple_to_configs[t_idx]:
                        counts[c] += 1
                    improved = True
                    break
        
        if not improved:
            break
    
    return cnt, active, counts

def construct():
    start = time.time()
    deadline = start + 112  # Safety margin
    
    best_cnt = 0
    best_active = [False] * N_TRIPLES
    best_counts = [0] * N_CONFIGS
    
    # Strategy 1: Multiple randomized greedy passes
    while time.time() < deadline - 20:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        cnt, active, counts = _greedy(order, deadline - 20)
        if cnt > best_cnt:
            best_cnt = cnt
            best_active = active
            best_counts = counts
    
    # Strategy 2: Local search on best
    if time.time() < deadline - 10:
        cnt, active, counts = _local_search(best_active, best_counts, deadline - 10)
        if cnt > best_cnt:
            best_cnt = cnt
            best_active = active
            best_counts = counts
    
    # Strategy 3: More greedy passes with different orderings
    while time.time() < deadline - 5:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        cnt, active, counts = _greedy(order, deadline - 5)
        if cnt > best_cnt:
            best_cnt = cnt
            best_active = active
            best_counts = counts
    
    # Strategy 4: Final local search
    if time.time() < deadline:
        cnt, active, counts = _local_search(best_active, best_counts, deadline)
        if cnt > best_cnt:
            best_cnt = cnt
            best_active = active
    
    # Convert to list of triples
    result = [list(ALL_TRIPLES[i]) for i in range(N_TRIPLES) if best_active[i]]
    return result

random.seed(42)
