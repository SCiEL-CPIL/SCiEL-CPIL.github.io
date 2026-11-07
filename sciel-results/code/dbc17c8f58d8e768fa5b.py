import time
import random
import sys

def construct():
    n = 15
    deadline = time.time() + 50  # Leave 10s buffer for Python overhead/evaluation
    
    # Initial graph: K_{7,8}
    # Partitions: A = 0..6, B = 7..14
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    best_edges = list(edges)
    best_score = len(best_edges)
    
    def get_neighbors(edges_set):
        neighbors = {i: set() for i in range(n)}
        for u, v in edges_set:
            neighbors[u].add(v)
            neighbors[v].add(u)
        return neighbors

    def is_valid(edges_set):
        neighbors = get_neighbors(edges_set)
        deg = [len(neighbors[i]) for i in range(n)]
        
        # Check all paths s-u-v-t
        # Iterate over edges (u, v)
        for u, v in edges_set:
            # s is neighbor of u, s != v
            for s in neighbors[u]:
                if s == v:
                    continue
                # t is neighbor of v, t != u, t != s
                for t in neighbors[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def try_improve(edges_set):
        # Local search: try adding random non-edges
        # If adding fails, try swapping
        # We return a new set if improved
        current_count = len(edges_set)
        
        # Generate candidates for addition
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        non_edges = [e for e in all_possible if e not in edges_set]
        
        random.shuffle(non_edges)
        
        for e in non_edges:
            new_edges = edges_set | {e}
            if is_valid(new_edges):
                return new_edges
        
        # Try swaps: remove one, add one
        # Limit swaps to avoid explosion
        edge_list = list(edges_set)
        random.shuffle(edge_list)
        
        for e_remove in edge_list[:50]: # Only try removing from a subset to save time
            non_edges_after_remove = [e for e in non_edges if e != e_remove] # Optimization: non_edges already excludes existing
            # Actually non_edges is static for this iteration, but e_remove is in edges_set, so it's not in non_edges.
            # We need to add e_remove back to the pool? No, we are adding a different edge.
            # So we try to add any non-edge.
            random.shuffle(non_edges)
            for e_add in non_edges[:20]:
                if e_add == e_remove: continue
                new_edges = (edges_set - {e_remove}) | {e_add}
                if is_valid(new_edges):
                    # Check if we actually increased count? No, swap keeps count same.
                    # We only want to accept if it leads to a state where we can add more?
                    # Or just accept to escape local minima?
                    # Since objective is max edges, a swap doesn't increase score.
                    # But it might allow future additions.
                    # For strict improvement, we should only accept if we can add an edge immediately?
                    # Let's stick to strict improvement for now, or accept swaps with low probability.
                    # Actually, if we are stuck, a swap might help.
                    # But to keep it simple and fast, let's just try to add.
                    # If we can't add, we are stuck.
                    pass
        
        return None

    while time.time() < deadline:
        # Try to add an edge
        # Pick a random non-edge
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        current_set = set(best_edges)
        non_edges = [e for e in all_possible if e not in current_set]
        
        if not non_edges:
            break # Complete graph, but likely invalid
        
        random.shuffle(non_edges)
        
        improved = False
        for e in non_edges:
            candidate = current_set | {e}
            if is_valid(candidate):
                best_edges = list(candidate)
                best_score = len(best_edges)
                improved = True
                break
        
        if not improved:
            # Try a random restart or mutation?
            # For now, just break or try swaps.
            # Let's try a few random swaps to escape local optimum
            current_set = set(best_edges)
            edge_list = list(current_set)
            if len(edge_list) > 0:
                e_rem = random.choice(edge_list)
                non_edges_now = [e for e in all_possible if e not in current_set]
                if non_edges_now:
                    e_add = random.choice(non_edges_now)
                    if e_add != e_rem:
                        candidate = (current_set - {e_rem}) | {e_add}
                        if is_valid(candidate):
                            # Accept swap only if it allows an immediate addition?
                            # Or just accept to diversify?
                            # Let's accept if it doesn't reduce count (it doesn't) and hope for the best.
                            # Actually, accepting a swap doesn't change score.
                            # It might help find a better local optimum.
                            # Let's do it with some probability.
                            if random.random() < 0.1:
                                best_edges = list(candidate)
            
            # If we are stuck for a while, maybe restart from K_{7,8} with different split?
            # But K_{7,8} is symmetric.
            # Let's just continue trying to add.
            pass

    # Return best found
    # Ensure types are correct
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
