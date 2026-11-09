"""Rewritten program: stronger Max Cut via multi-restart local search
with 1-flip and fast 2-flip moves under an explicit wall-clock deadline."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_WALL_BUDGET_S = 150.0
_DEADLINE_FRACTION = 0.92  # safety margin


def _cut_size(adj, labels):
    a = labels.astype(np.int8)
    cross = (a[:, None] != a[None, :]) & np.triu(np.ones(adj.shape, dtype=bool), 1)
    return int(np.sum(cross))


def _local_search(adj, labels, n, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # --- 1-flip pass ---
        for v in range(n):
            if time.time() >= deadline:
                return labels
            lv = labels[v]
            same = adj[v][labels == lv].sum()
            other = adj[v][labels != lv].sum()
            if other > same:
                labels[v] = 1 - lv
                improved = True
        # --- 2-flip pass ---
        order = np.random.permutation(n)
        for i in range(n - 1):
            if time.time() >= deadline:
                return labels
            u = int(order[i])
            lu = labels[u]
            rest = order[i + 1:]
            cands = rest[labels[rest] != lu]
            for v in cands:
                v = int(v)
                # delta of flipping u and v
                # edges u-x (x != v): change = adj[u,x] * ( (1-lu)!=labels[x] - lu!=labels[x] )
                # edges v-x (x != u): change = adj[v,x] * ( (1-lv)!=labels[x] - lv!=labels[x] )
                # edge u-v: change = adj[u,v] * ( ((1-lu)!=(1-lv)) - (lu!=lv) )
                lv = labels[v]
                # vectorized
                du = ((1 - lu) != labels)
                bu = (lu != labels)
                dv = ((1 - lv) != labels)
                bv = (lv != labels)
                delta = int(np.sum(adj[u] * (du - bu)))
                delta -= int(adj[u, v] * (du[v] - bu[v]))  # remove x=v term from u-row
                delta += int(np.sum(adj[v] * (dv - bv)))
                delta -= int(adj[v, u] * (dv[u] - bv[u]))  # remove x=u term from v-row
                if adj[u, v]:
                    before_uv = 1 if lu != lv else 0
                    after_uv = 1 if (1 - lu) != (1 - lv) else 0
                    delta += after_uv - before_uv
                if delta > 0:
                    labels[u] = 1 - lu
                    labels[v] = 1 - lv
                    improved = True
    return labels


def partition(adjacency):
    start_time = time.time()
    deadline = start_time + _WALL_BUDGET_S * _DEADLINE_FRACTION
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]

    best_labels = None
    best_cut = -1

    base = np.arange(n) % 2
    improved = _local_search(adj, base.copy(), n, deadline)
    c = _cut_size(adj, improved)
    if c > best_cut:
        best_cut = c
        best_labels = improved.copy()

    rng = np.random.default_rng(0)
    while time.time() < deadline:
        labels = (rng.random(n) < 0.5).astype(np.int8)
        improved = _local_search(adj, labels, n, deadline)
        c = _cut_size(adj, improved)
        if c > best_cut:
            best_cut = c
            best_labels = improved.copy()

    if best_labels is None:
        best_labels = np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
