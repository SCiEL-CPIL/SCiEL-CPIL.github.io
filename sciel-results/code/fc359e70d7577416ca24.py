import time
import itertools

def _count_comparable(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(sets)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(sets[i]) < len(sets[j]):
                if all(x in sets[j] for x in sets[i]):
                    count += 1
            elif len(sets[j]) < len(sets[i]):
                if all(x in sets[i] for x in sets[j]):
                    count += 1
    return count

def _construct_base():
    """Construct a strong initial solution using a chain plus optimized extensions."""
    # Start with a full chain of 13 sets (sizes 0 through 12)
    chain = [list(range(i)) for i in range(13)]  # C_i = {0, 1, ..., i-1}
    
    sets = list(chain)
    
    # Add 27 more sets greedily to maximize comparable pairs
    # Candidate sets: all subsets of {0,...,11} not already in sets
    # For efficiency, generate candidates by size
    
    used = set(frozenset(s) for s in sets)
    best_sets = list(sets)
    
    deadline = time.time() + 80  # 80 second deadline for safety
    
    # Generate candidate sets by size, prioritizing those that fit in chain gaps
    candidates_by_size = {}
    for r in range(13):
        cands = []
        for combo in itertools.combinations(range(12), r):
            fs = frozenset(combo)
            if fs not in used:
                cands.append(list(combo))
        candidates_by_size[r] = cands
    
    # For each size, sort candidates by how well they fit the chain
    # A set S of size r is comparable with chain element C_r (size r) if S ⊂ C_r or C_r ⊂ S
    # Since |S| = |C_r| = r, they're comparable only if equal (already excluded)
    # S is comparable with C_{r-1} if S ⊃ C_{r-1} (i.e., S contains {0,...,r-2})
    # S is comparable with C_{r+1} if S ⊂ C_{r+1} (i.e., S ⊆ {0,...,r})
    
    for r in range(13):
        def score_candidate(S):
            # Count how many existing sets S is comparable with
            cnt = 0
            for existing in best_sets:
                if len(S) < len(existing):
                    if all(x in existing for x in S):
                        cnt += 1
                elif len(existing) < len(S):
                    if all(x in S for x in existing):
                        cnt += 1
            return cnt
        
        if candidates_by_size[r]:
            # Sort by score descending
            candidates_by_size[r].sort(key=score_candidate, reverse=True)
    
    # Greedily add sets
    added = 0
    while added < 27 and time.time() < deadline:
        best_score = -1
        best_set = None
        best_size = -1
        
        for r in range(13):
            if not candidates_by_size[r]:
                continue
            for idx, S in enumerate(candidates_by_size[r]):
                if time.time() > deadline:
                    break
                # Count comparable pairs with current best_sets
                cnt = 0
                for existing in best_sets:
                    if len(S) < len(existing):
                        if all(x in existing for x in S):
                            cnt += 1
                    elif len(existing) < len(S):
                        if all(x in S for x in existing):
                            cnt += 1
                if cnt > best_score:
                    best_score = cnt
                    best_set = S
                    best_size = r
                    best_idx = idx
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if best_set is None:
            break
        
        best_sets.append(best_set)
        used.add(frozenset(best_set))
        candidates_by_size[best_size].pop(best_idx)
        added += 1
    
    # Verify we have exactly 40 distinct sets
    assert len(best_sets) == 40
    assert len(set(frozenset(s) for s in best_sets)) == 40
    
    # Convert to lists of ints
    return [list(s) for s in best_sets]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline
    
    try:
        result = _construct_base()
        # Verify validity
        assert len(result) == 40
        seen = set()
        for s in result:
            assert isinstance(s, (list, tuple))
            for x in s:
                assert isinstance(x, int) and not isinstance(x, bool)
                assert 0 <= x <= 11
            fs = frozenset(s)
            assert fs not in seen
            seen.add(fs)
        return result
    except Exception:
        # Fallback: simple chain-based construction
        chain = [list(range(i)) for i in range(13)]
        # Add remaining sets from a fixed pattern
        extra = []
        for r in range(1, 12):
            for start in range(12 - r + 1):
                s = list(range(start, start + r))
                if frozenset(s) not in set(frozenset(c) for c in chain) and frozenset(s) not in set(frozenset(e) for e in extra):
                    extra.append(s)
                if len(chain) + len(extra) >= 40:
                    break
            if len(chain) + len(extra) >= 40:
                break
        result = chain + extra[:40 - len(chain)]
        while len(result) < 40:
            # Add more
            for r in range(13):
                for combo in itertools.combinations(range(12), r):
                    s = list(combo)
                    if frozenset(s) not in set(frozenset(x) for x in result):
                        result.append(s)
                        break
                if len(result) >= 40:
                    break
        result = result[:40]
        return [list(s) for s in result]
