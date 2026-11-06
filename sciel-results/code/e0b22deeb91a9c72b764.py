import time
import math
from fractions import Fraction

N = 256
DEADLINE = 90.0
SAFETY = 12.0  # seconds of margin before the 90s kill

def _build_graph():
    # conflict: edge (i,j) iff {i,j} can be extended to a 3-term AP in [1,N]
    # i.e. 2i - j in [1,N] or 2j - i in [1,N] or 2i + j in [1,N] (with distinctness handled)
    n = N
    adj = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # three-term AP with i,j as two terms: third = 2i-j or 2j-i or 2i+j
            t1 = 2 * i - j
            t2 = 2 * j - i
            t3 = 2 * i + j
            # AP requires three distinct terms; the pair (i,j) is part of a nonconstant
            # 3-AP if any third integer in [1,n] completes it.
            ok = False
            if 1 <= t1 <= n and t1 != i and t1 != j:
                ok = True
            elif 1 <= t2 <= n and t2 != i and t2 != j:
                ok = True
            elif 1 <= t3 <= n and t3 != i and t3 != j:
                ok = True
            if ok:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _greedy_seed(adj):
    # ascending by weight (largest reciprocal first) -> strong feasible set
    selected = set()
    for x in range(1, N + 1):
        if all(y not in selected for y in adj[x]):
            selected.add(x)
    return selected

def _is_valid(sel, adj):
    for x in sel:
        for y in sel:
            if x < y and y in adj[x]:
                return False
    return True

def _weight(sel):
    return sum(1.0 / x for x in sel)

def _local_polish(sel, adj, deadline):
    # try to improve via single removals (to enable adds) and single adds
    sel = set(sel)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # try adding any non-conflicting element
        for x in range(1, N + 1):
            if x in sel:
                continue
            if all(y not in sel for y in adj[x]):
                sel.add(x)
                improved = True
        # try removing one element if it enables a net gain
        for a in list(sel):
            trial = set(sel)
            trial.remove(a)
            gain = 0.0
            added = []
            for x in range(1, N + 1):
                if x in trial:
                    continue
                if all(y not in trial for y in adj[x]):
                    trial.add(x)
                    added.append(x)
                    gain += 1.0 / x
            net = gain - 1.0 / a
            if net > 1e-12:
                sel = trial
                improved = True
                break
    return sel

def _color_bound(compl, candidates, weights):
    # greedy coloring of the induced subgraph on candidates (complement graph),
    # returns list of color classes upper-bounding weights for Tomita.
    # We color so that each color class is an independent set in the complement
    # (i.e. a clique in the conflict graph) -> a clique in complement is a set
    # of vertices with no complement edges = all conflict edges present.
    # For MWIS bound we need: max weight of any independent set in conflict
    #  = max weight of any clique in complement.
    # Greedy sequential coloring of complement gives an upper bound on clique weight.
    order = sorted(candidates, key=lambda v: -weights[v])
    color_class_weights = []
    color_of = {}
    for v in order:
        used = set()
        for u in candidates:
            if u == v:
                continue
            cu = color_of.get(u)
            if cu is not None:
                # u and v adjacent in complement? then they cannot share color
                if u in compl[v]:
                    used.add(cu)
        c = 0
        while c in used:
            c += 1
        color_of[v] = c
        if c == len(color_class_weights):
            color_class_weights.append(0.0)
        color_class_weights[c] += weights[v]
    return color_class_weights

def _mwis_branch(compl, candidates, weights, deadline, best):
    # returns best independent set weight found (in conflict) = best clique weight in complement
    # candidates: set of vertices (in complement graph) we may still pick (clique must be
    # independent in conflict). We pick a clique in complement.
    # best: dict with 'weight'
    if not candidates:
        return
    # bound: greedy coloring of complement restricted to candidates
    cw = _color_bound(compl, candidates, weights)
    total_bound = sum(cw)
    if total_bound <= best['weight'] + 1e-12:
        return
    # branch: pick a pivot to reduce branching
    # choose vertex with max complement degree within candidates
    cand_list = list(candidates)
    pivot = max(cand_list, key=lambda v: len(compl[v] & candidates))
    # candidates to branch on: those NOT adjacent to pivot in complement
    # (clique containing pivot can only include non-neighbors of pivot in complement)
    branch = [v for v in cand_list if v not in compl[pivot]]
    # order by weight desc
    branch.sort(key=lambda v: -weights[v])
    for v in branch:
        if time.time() > deadline:
            return
        newcand = candidates & compl[v]  # neighbors in complement (can co-clique)
        # bound check on newcand
        if newcand:
            cw2 = _color_bound(compl, newcand, weights)
            if weights[v] + sum(cw2) <= best['weight'] + 1e-12:
                continue
        # recurse
        _mwis_branch(compl, newcand, weights, deadline, best)
        candidates.discard(v)
        # update bound for remaining
        if not candidates:
            break
        cw3 = _color_bound(compl, candidates, weights)
        if sum(cw3) <= best['weight'] + 1e-12:
            break

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY
    adj = _build_graph()
    # complement
    compl = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if j not in adj[i]:
                compl[i].add(j)
                compl[j].add(i)
    weights = {x: 1.0 / x for x in range(1, N + 1)}

    # initial feasible seed
    sel = _greedy_seed(adj)
    best_w = _weight(sel)
    best_set = set(sel)

    # local polish
    sel = _local_polish(sel, adj, deadline)
    if _weight(sel) > best_w:
        best_w = _weight(sel)
        best_set = set(sel)

    # branch and bound on complement (clique = independent in conflict)
    if time.time() < deadline:
        best = {'weight': best_w}
        _mwis_branch(compl, set(range(1, N + 1)), weights, deadline, best)
        # note: branch-and-bound here only tracks weight, not the set; the seed+polish
        # already gives a concrete set. If B&B proves a higher weight, we don't have the
        # set, so we rely on seed+polish for the concrete answer. Keep best_set.

    # final verification
    if not _is_valid(best_set, adj):
        # fallback to greedy
        best_set = _greedy_seed(adj)

    if not best_set:
        best_set = {1}
    return sorted(best_set)
