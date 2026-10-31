"""MST-aware greedy: penalize edges that disconnect remaining unvisited cities."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0
_STATE = {}  # (n, id) -> (unvisited_set, mst_cache_version)


def _compute_mst_edges(unvisited, d):
    """Return set of MST edge pairs among unvisited nodes using Prim's algorithm."""
    unv = list(unvisited)
    k = len(unv)
    if k <= 1:
        return set()
    idx = {v: i for i, v in enumerate(unv)}
    in_mst = [False] * k
    min_edge = [np.inf] * k
    min_edge[0] = 0.0
    parent = [-1] * k
    edges = set()
    for _ in range(k):
        # find min edge node
        best = -1
        best_val = np.inf
        for i in range(k):
            if not in_mst[i] and min_edge[i] < best_val:
                best_val = min_edge[i]
                best = i
        if best == -1:
            break
        in_mst[best] = True
        if parent[best] != -1:
            u, v = unv[parent[best]], unv[best]
            edges.add((min(u, v), max(u, v)))
        # update
        for j in range(k):
            if not in_mst[j]:
                dist = d[unv[best], unv[j]]
                if dist < min_edge[j]:
                    min_edge[j] = dist
                    parent[j] = best
    return edges


def _is_mst_bridge(node_a, node_b, unvisited, mst_edges):
    """Check if removing node_b from unvisited would make edge (node_a, node_b) a bridge in MST."""
    # Simpler check: is (node_a, node_b) in the MST? If so, it's critical.
    key = (min(node_a, node_b), max(node_a, node_b))
    return key in mst_edges


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    if unvisited.size == 0:
        return int(current_node)
    n = int(distance_matrix.shape[0])
    d = distance_matrix

    if time.time() > _DEADLINE:
        return int(unvisited[int(np.argmin(d[current_node, unvisited]))])

    # Get or create state for this instance
    state_key = (n, id(distance_matrix))
    if state_key not in _STATE:
        _STATE[state_key] = set(unvisited.tolist())
    unv_set = _STATE[state_key]

    # If unvisited changed, update set
    if len(unv_set) != len(unvisited):
        unv_set = set(unvisited.tolist())
        _STATE[state_key] = unv_set

    k = len(unv_set)
    if k <= 2:
        # Just pick nearest
        return int(unvisited[int(np.argmin(d[current_node, unvisited]))])

    # Compute MST edges among unvisited nodes
    mst_edges = _compute_mst_edges(unv_set, d)

    # Score each candidate: base distance + penalty if it's an MST bridge
    best_node = -1
    best_score = np.inf
    cur = int(current_node)
    for idx, node in enumerate(unvisited):
        node = int(node)
        dist = d[cur, node]
        # Penalty: if this edge is in the MST, removing this node disconnects the graph
        # We penalize taking MST edges early to avoid creating long bridges later
        if _is_mst_bridge(cur, node, unv_set, mst_edges):
            penalty = 0.3 * dist  # moderate penalty for MST edges
        else:
            penalty = 0.0
        score = dist + penalty
        if score < best_score:
            best_score = score
            best_node = node

    return int(best_node)
# EVOLVE-BLOCK-END
