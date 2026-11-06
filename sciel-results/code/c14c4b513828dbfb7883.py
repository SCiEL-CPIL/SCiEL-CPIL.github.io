import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit

    # Use a near-turbo graph: random graph with density ~0.514
    # G(n, 0.5) has expected 390 edges, we need 401, so density ~ 401/780 ≈ 0.514
    
    # Strategy: Start with a random graph, then use simulated annealing / local search
    # to minimize the maximum book size.
    
    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return edges

    def build_masks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= 1 << v
            masks[v] |= 1 << u
        return masks

    def max_book(masks, edges):
        max_b = 0
        for u, v in edges:
            common = masks[u] & masks[v]
            b = common.bit_count()
            if b > max_b:
                max_b = b
        return max_b

    def book_of_edge(masks, u, v):
        return (masks[u] & masks[v]).bit_count()

    # Initial graph
    edges = random_graph()
    masks = build_masks(edges)
    current_score = max_book(masks, edges)
    best_score = current_score
    best_edges = set(edges)

    # Precompute all non-edges for faster random selection
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(edges)
    non_edges = [p for p in all_pairs if p not in edge_set]

    no_improve = 0
    step = 0

    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(list(edges))
        
        # Pick a random non-edge to add
        x, y = random.choice(non_edges)
        
        # Perform swap
        edges.discard((u, v))
        edges.add((x, y))
        
        # Update masks
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        masks[x] |= 1 << y
        masks[y] |= 1 << x
        
        # Update non_edges list
        non_edges.remove((x, y))
        non_edges.append((u, v))
        
        new_score = max_book(masks, edges)
        
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(edges)
                no_improve = 0
            else:
                # Accept with some probability if equal, to explore
                if random.random() < 0.3:
                    no_improve += 1
                else:
                    # Revert
                    edges.discard((x, y))
                    edges.add((u, v))
                    masks[x] &= ~(1 << y)
                    masks[y] &= ~(1 << x)
                    masks[u] |= 1 << v
                    masks[v] |= 1 << u
                    non_edges.remove((u, v))
                    non_edges.append((x, y))
                    no_improve += 1
        else:
            # Revert
            edges.discard((x, y))
            edges.add((u, v))
            masks[x] &= ~(1 << y)
            masks[y] &= ~(1 << x)
            masks[u] |= 1 << v
            masks[v] |= 1 << u
            non_edges.remove((u, v))
            non_edges.append((x, y))
            no_improve += 1
        
        step += 1
        
        # If stuck for too long, restart
        if no_improve > 5000:
            edges = random_graph()
            masks = build_masks(edges)
            current_score = max_book(masks, edges)
            edge_set = set(edges)
            non_edges = [p for p in all_pairs if p not in edge_set]
            no_improve = 0

    return [[u, v] for u, v in best_edges]
