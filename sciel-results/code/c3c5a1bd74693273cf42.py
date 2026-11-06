"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] sharing a factor with 255
    # and must include 255
    candidates = []
    for i in range(1, 256):
        if gcd(i, 255) > 1:
            candidates.append(i)
    
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    
    # Build adjacency: edge between i and j if gcd(candidates[i], candidates[j]) > 1
    # Use bitsets for speed
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Maximum clique via branch and bound with greedy coloring
    # We need a clique that includes 255 (find its index)
    idx_255 = candidates.index(255)
    
    # Greedy coloring for upper bound
    def greedy_color(cand_set):
        """Color candidates in cand_set (a set of indices) greedily for upper bound."""
        colors = {}
        color_classes = []
        for v in sorted(cand_set):
            placed = False
            for ci, cls in enumerate(color_classes):
                # Check if v is adjacent to all in cls
                can_place = True
                for u in cls:
                    if not (adj[v] & (1 << u)):
                        can_place = False
                        break
                if can_place:
                    cls.append(v)
                    colors[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                colors[v] = len(color_classes) - 1
        return colors, len(color_classes)
    
    best_clique = set([idx_255])
    
    # Branch and bound: try to extend clique from {255}
    # Candidates that can join: those adjacent to 255
    # Since 255 = 3*5*17, all candidates share a factor with 255, so all are adjacent
    
    # Order candidates by degree (descending) for better pruning
    # Exclude 255 from the candidate list for branching
    branch_cands = [i for i in range(n) if i != idx_255]
    
    # Sort by degree descending
    branch_cands.sort(key=lambda i: bin(adj[i]).count('1'), reverse=True)
    
    # Use a recursive search with bitset pruning
    # State: current clique (as set), candidate set (as set of indices)
    
    import sys
    sys.setrecursionlimit(10000)
    
    best_size = 1
    best_set = {idx_255}
    
    def search(clique_set, cand_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        
        if not cand_set:
            if len(clique_set) > best_size:
                best_size = len(clique_set)
                best_set = set(clique_set)
            return
        
        # Upper bound: current clique size + greedy color bound on candidates
        # that are all adjacent to all in clique
        # For simplicity, use |clique| + |cand_set| as loose bound
        if len(clique_set) + len(cand_set) <= best_size:
            return
        
        # Pick vertex with highest degree in cand_set
        # Use first in sorted order
        v = list(cand_set)[0]
        
        # Branch: include v
        new_clique = clique_set | {v}
        # New candidates: those in cand_set that are adjacent to v and to all in clique
        new_cand = set()
        for u in cand_set:
            if u != v and (adj[v] & (1 << u)):
                # Check u is adjacent to all in clique
                ok = True
                for c in new_clique:
                    if not (adj[u] & (1 << c)):
                        ok = False
                        break
                if ok:
                    new_cand.add(u)
        search(new_clique, new_cand)
        
        # Branch: exclude v
        cand_set.discard(v)
        search(clique_set, cand_set)
    
    # Start search
    initial_cands = set(branch_cands)
    search({idx_255}, initial_cands)
    
    # Convert best set back to values
    result = sorted([candidates[i] for i in best_set])
    
    # Verify 255 is in result
    if 255 not in result:
        result.append(255)
    
    return result
# EVOLVE-BLOCK-END
