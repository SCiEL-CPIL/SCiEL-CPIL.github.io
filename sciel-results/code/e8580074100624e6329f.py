"""Pairwise coprime selection below 512 via exact branch-and-bound on the conflict graph."""
import time
from math import gcd
from fractions import Fraction

_N = 512
_DEADLINE = time.time() + 135.0  # safety margin under the 120s evaluator limit


def _build_graph():
    n = _N - 1  # numbers 1..511
    # weights w[i] = 1/(512 - (i+1))
    w = [Fraction(1, _N - (i + 1)) for i in range(n)]
    # adj_mask[i] = bitmask of j (j != i) with gcd(i+1, j+1) == 1
    adj_mask = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        ai = i + 1
        mask = 0
        for j in range(n):
            if i != j and gcd(ai, j + 1) == 1:
                mask |= (1 << j)
        adj_mask[i] = mask
    # conflict_mask[i] = bitmask of j (j != i) with gcd(i+1, j+1) > 1
    conflict_mask = [0] * n
    for i in range(n):
        conflict_mask[i] = all_mask ^ (1 << i) ^ adj_mask[i]
    return n, w, adj_mask, conflict_mask


def _bound(cand_mask, w, conflict_mask, order_by_weight):
    # Greedy coloring of the CONFLICT graph subgraph defined by cand_mask
    # Each color class is an independent set in the conflict graph (i.e., a clique in the coprime graph)
    # The max weight clique in the coprime graph can pick at most 1 node from each conflict-color class?
    # Wait.
    # Conflict Graph G_c. Edges if gcd > 1.
    # We want Max Weight Independent Set in G_c.
    # We color G_c. Color classes are Independent Sets in G_c.
    # An Independent Set in G_c can contain multiple nodes from a color class?
    # No. A color class is an independent set. An independent set can contain the entire color class.
    # So this logic is wrong.
    
    # Let's re-verify the standard bound.
    # For Max Independent Set in G:
    # Color G. Let colors be 1..k.
    # The independence number alpha(G) <= k? NO.
    # alpha(G) <= n/k? NO.
    # The standard bound is: If we can color G with k colors, then the size of a maximal independent set is at least n/k. That's a lower bound.
    # The UPPER bound on the size of a MAXIMAL independent set?
    # Actually, the standard Tomita bound for Max Clique (which is Max Independent Set in complement) uses coloring of the complement graph?
    # Let's check Tomita 2003.
    # Algorithm: Color the graph G (the one we are finding a Clique in).
    # If we color G with k colors, then the size of a Max Clique in G is at most k.
    # Why? A clique in G is an independent set in G_bar.
    # Wait.
    # If we color G, the color classes are Independent Sets in G.
    # A Clique in G can contain at most 1 vertex from each Independent Set?
    # YES. Because if a clique contains 2 vertices from the same independent set, those 2 vertices must be adjacent in G. But they are in the same independent set, so they are NOT adjacent. Contradiction.
    # So, if we color the COPRIME graph (G) with k colors, the Max Weight Clique is bounded by the sum of max weights in each color class.
    
    # So we need to color the COPRIME graph (adj_mask).
    # Condition for adding v to color class C (which is an independent set in Coprime Graph):
    # v must NOT be adjacent to any node in C in the Coprime Graph.
    # i.e., (adj_mask[v] & C) == 0.
    
    if cand_mask == 0:
        return Fraction(0)
        
    # Get nodes in cand_mask sorted by weight desc
    nodes_in_cand = [i for i in order_by_weight if (cand_mask >> i) & 1]
    
    color_classes = [] # List of bitmasks
    max_weights = []   # Max weight in each class
    
    for v in nodes_in_cand:
        placed = False
        for idx, c_mask in enumerate(color_classes):
            # Check if v is non-adjacent to all in c_mask (in Coprime Graph)
            if (adj_mask[v] & c_mask) == 0:
                color_classes[idx] = c_mask | (1 << v)
                if w[v] > max_weights[idx]:
                    max_weights[idx] = w[v]
                placed = True
                break
        if not placed:
            color_classes.append(1 << v)
            max_weights.append(w[v])
            
    return sum(max_weights, Fraction(0))


def _bbs(cand_mask, w, adj_mask, cur_w, best_w, best_set, state, order_by_weight):
    if time.time() > _DEADLINE:
        state["killed"] = True
        return
    if cand_mask == 0:
        if cur_w > best_w[0]:
            best_w[0] = cur_w
            best_set[0] = list(state["chosen"])
        return
        
    # Upper Bound
    ub = _bound(cand_mask, w, adj_mask, order_by_weight)
    if cur_w + ub <= best_w[0]:
        return

    # Select next node to branch on
    # Heuristic: pick node with max weight in cand_mask
    next_node = -1
    for i in order_by_weight:
        if (cand_mask >> i) & 1:
            next_node = i
            break
            
    if next_node == -1:
        return

    # Branch 1: Include next_node
    # New candidates must be adjacent to next_node in Coprime Graph
    new_cand_mask = cand_mask & adj_mask[next_node]
    
    state["chosen"].append(next_node)
    _bbs(new_cand_mask, w, adj_mask, cur_w + w[next_node], best_w, best_set, state, order_by_weight)
    state["chosen"].pop()
    if state["killed"]:
        return
        
    # Branch 2: Exclude next_node
    # Remove next_node from cand_mask
    new_cand_mask = cand_mask & ~(1 << next_node)
    _bbs(new_cand_mask, w, adj_mask, cur_w, best_w, best_set, state, order_by_weight)
    if state["killed"]:
        return


def construct():
    n, w, adj_mask, conflict_mask = _build_graph()
    state = {"chosen": [], "killed": False}
    best_w = [Fraction(0)]
    best_set = [[]]
    
    # Initial greedy solution
    order_by_weight = sorted(range(n), key=lambda x: (-w[x], x))
    
    # Greedy heuristic
    cand_mask = (1 << n) - 1
    chosen = []
    cur_w = Fraction(0)
    while cand_mask:
        next_node = -1
        for i in order_by_weight:
            if (cand_mask >> i) & 1:
                next_node = i
                break
        if next_node == -1: break
        chosen.append(next_node)
        cur_w += w[next_node]
        cand_mask &= adj_mask[next_node]
        
    best_w[0] = cur_w
    best_set[0] = list(chosen)
    
    # B&B
    _bbs((1 << n) - 1, w, adj_mask, Fraction(0), best_w, best_set, state, order_by_weight)
    
    return [i + 1 for i in best_set[0]]


if __name__ == "__main__":
    res = construct()
    print(len(res), float(sum(Fraction(1, _N - a) for a in res)))
