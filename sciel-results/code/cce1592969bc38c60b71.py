import time
import itertools
import random
import numpy as np

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

def _build_comparability_matrix():
    """Build a 4096x4096 boolean matrix where comp[i,j]=True iff subsets i,j are comparable."""
    comp = np.zeros((4096, 4096), dtype=bool)
    for i in range(4096):
        # Enumerate all proper subsets of i
        sub = (i - 1) & i
        while True:
            comp[i, sub] = True
            comp[sub, i] = True
            if sub == 0:
                break
            sub = (sub - 1) & i
        # Enumerate all proper supersets of i
        # A superset j of i satisfies: j | i == j, i.e., all bits in i are in j
        # j = i | (x & ~i) for x in 0..4095, but we need j != i
        # Iterate over all subsets of the complement
        complement = (~i) & 0xFFF
        sub_comp = complement
        while True:
            j = i | sub_comp
            if j != i:
                comp[i, j] = True
                comp[j, i] = True
            if sub_comp == 0:
                break
            sub_comp = (sub_comp - 1) & complement
    return comp

def _construct_base():
    """Construct a strong initial solution using numpy-accelerated local search on comparability graph."""
    deadline = time.time() + 80
    
    # Build comparability matrix
    comp = _build_comparability_matrix()
    
    # Degree of each node
    deg = comp.sum(axis=1)
    node_order = np.argsort(deg)[::-1]
    
    # Greedy initial solution: pick highest-degree nodes
    selected = []
    used = np.zeros(4096, dtype=bool)
    
    for node in node_order:
        if len(selected) >= 40:
            break
        selected.append(int(node))
        used[node] = True
    
    def compute_score(sel):
        if len(sel) < 2:
            return 0
        sel_arr = np.array(sel)
        sub_comp = comp[np.ix_(sel_arr, sel_arr)]
        return int(sub_comp.sum() // 2)
    
    current_score = compute_score(selected)
    
    # Local search with numpy-accelerated edge counting
    improved = True
    max_iter = 500
    iter_count = 0
    
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            
            saved = selected[i]
            selected.pop(i)
            used[saved] = False
            
            sel_arr = np.array(selected)
            if len(selected) > 0:
                edges_all = comp[:, sel_arr].sum(axis=1)
            else:
                edges_all = np.zeros(4096)
            edges_all[used] = -1
            
            best_node = int(np.argmax(edges_all))
            
            if edges_all[best_node] > 0:
                selected.append(best_node)
                used[best_node] = True
                
                new_score = compute_score(selected)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    selected.pop()
                    used[best_node] = False
                    selected.insert(i, saved)
                    used[saved] = True
            else:
                selected.insert(i, saved)
                used[saved] = True
    
    # Convert to list of lists
    result = []
    for idx in selected:
        subset = []
        for bit in range(12):
            if (idx >> bit) & 1:
                subset.append(bit)
        result.append(subset)
    
    assert len(result) == 40
    assert len(set(frozenset(s) for s in result)) == 40
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    try:
        result = _construct_base()
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
        chain = [list(range(i)) for i in range(13)]
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
