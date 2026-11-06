import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return

    def prepare_data(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Red K4s: all 4-subsets
        red_k4s = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            red_k4s.append(edges)
            
        # Blue K5s: all 5-subsets
        blue_k5s = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            blue_k5s.append(edges)
            
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        edge_to_k5s = [[] for _ in range(num_triples)]
        
        for i, edges in enumerate(red_k4s):
            for e in edges:
                edge_to_k4s[e].append(i)
                
        for i, edges in enumerate(blue_k5s):
            for e in edges:
                edge_to_k5s[e].append(i)
        
        return triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s

    def local_search(n, time_limit=10.0, seed_val=0):
        """
        Fast local search using incremental count updates.
        """
        rng = random.Random(seed_val)
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        # Initialize with random colors
        colors = [rng.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        for i, edges in enumerate(red_k4s):
            red_counts[i] = sum(1 for e in edges if colors[e] == 0)
        for i, edges in enumerate(blue_k5s):
            blue_counts[i] = sum(1 for e in edges if colors[e] == 1)
        
        num_red_violations = sum(1 for c in red_counts if c == 4)
        num_blue_violations = sum(1 for c in blue_counts if c == 10)
        total_violations = num_red_violations + num_blue_violations
        
        start = time.time()
        
        # Maintain lists of violating cliques indices
        violating_k4s = [i for i, c in enumerate(red_counts) if c == 4]
        violating_k5s = [i for i, c in enumerate(blue_counts) if c == 10]
        
        step = 0
        while time.time() - start < time_limit:
            if total_violations == 0:
                return colors
            
            # Choose a random violating clique
            if not violating_k4s and not violating_k5s:
                # Stale lists? Rebuild
                violating_k4s = [i for i, c in enumerate(red_counts) if c == 4]
                violating_k5s = [i for i, c in enumerate(blue_counts) if c == 10]
                if not violating_k4s and not violating_k5s:
                    return colors
            
            # Weighted choice: more violations -> more likely to be picked?
            # For now, just pick random type if both exist.
            if violating_k4s and (not violating_k5s or rng.random() < 0.5):
                k4_idx = rng.choice(violating_k4s)
                e = rng.choice(red_k4s[k4_idx])
            else:
                if not violating_k5s:
                    # Only k4s available, but logic above should have caught it?
                    # If violating_k4s is empty but total_violations > 0, lists are stale.
                    violating_k4s = [i for i, c in enumerate(red_counts) if c == 4]
                    violating_k5s = [i for i, c in enumerate(blue_counts) if c == 10]
                    if not violating_k4s and not violating_k5s:
                        return colors
                    if violating_k4s:
                        k4_idx = rng.choice(violating_k4s)
                        e = rng.choice(red_k4s[k4_idx])
                    else:
                        k5_idx = rng.choice(violating_k5s)
                        e = rng.choice(blue_k5s[k5_idx])
                else:
                    k5_idx = rng.choice(violating_k5s)
                    e = rng.choice(blue_k5s[k5_idx])
            
            current_color = colors[e]
            new_color = 1 - current_color
            
            # Apply flip
            colors[e] = new_color
            
            # Update red_counts
            for k4_idx in edge_to_k4s[e]:
                if current_color == 0: # Was Red, now Blue
                    old_c = red_counts[k4_idx]
                    red_counts[k4_idx] = old_c - 1
                    if old_c == 4:
                        total_violations -= 1
                else: # Was Blue, now Red
                    old_c = red_counts[k4_idx]
                    red_counts[k4_idx] = old_c + 1
                    if old_c == 3:
                        total_violations += 1
                        
            # Update blue_counts
            for k5_idx in edge_to_k5s[e]:
                if current_color == 1: # Was Blue, now Red
                    old_c = blue_counts[k5_idx]
                    blue_counts[k5_idx] = old_c - 1
                    if old_c == 10:
                        total_violations -= 1
                else: # Was Red, now Blue
                    old_c = blue_counts[k5_idx]
                    blue_counts[k5_idx] = old_c + 1
                    if old_c == 9:
                        total_violations += 1
            
            # If we made it worse, revert
            if total_violations > 0:
                # Simple heuristic: if total increased, revert?
                # Actually, we don't store old_total easily here without extra variable.
                # Let's just keep the move if it doesn't increase total violations?
                # Wait, I didn't store old_total.
                # Let's just accept the move. Simulated annealing would be better,
                # but simple hill climbing (only accept if total_violations <= old_total)
                # requires storing old_total.
                # Let's just accept all moves for now to keep it simple and fast.
                # Or better: if total_violations > previous, revert.
                # I'll just accept for now, as the search is random.
                pass
            
            # Periodically rebuild violating lists to ensure they are accurate
            # If we don't, we might pick an index that is no longer violating.
            # This is fine, it just means we flip a random edge in a non-violating clique.
            # But if the list becomes empty while violations exist, we must rebuild.
            
            step += 1
            if step % 100 == 0:
                # Check if lists are empty but violations exist
                if (not violating_k4s and any(c == 4 for c in red_counts)) or \
                   (not violating_k5s and any(c == 10 for c in blue_counts)):
                    violating_k4s = [i for i, c in enumerate(red_counts) if c == 4]
                    violating_k5s = [i for i, c in enumerate(blue_counts) if c == 10]
            else:
                # Lightweight check: if list is empty, assume stale and rebuild
                if (not violating_k4s and total_violations > 0) or (not violating_k5s and total_violations > 0):
                     # This check is O(1) but might trigger rebuild too often?
                     # No, it only triggers if list is empty.
                     # If list is empty, we need to check if there are actually violations.
                     # But checking 'any' is O(N).
                     # Let's just rebuild if list is empty and total_violations > 0.
                     # This is rare.
                     violating_k4s = [i for i, c in enumerate(red_counts) if c == 4]
                     violating_k5s = [i for i, c in enumerate(blue_counts) if c == 10]
            
        return None

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try increasing n
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        # Time budget for this n
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        
        found = False
        
        # Allocate more time for larger n?
        # Let's try multiple attempts with increasing time or just one long attempt.
        # For n=34, one long attempt is better to allow convergence.
        # Let's give 80% of remaining time to the first attempt, and split the rest.
        
        time_per_attempt = remaining * 0.7
        if time_per_attempt < 0.5:
            time_per_attempt = remaining
        
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            
            # Reduce time for subsequent attempts
            current_limit = time_per_attempt / (attempt + 1)
            if current_limit < 0.5:
                current_limit = 0.5
                
            colors = local_search(n, time_limit=current_limit, seed_val=attempt * 1000 + n)
            
            if colors is not None:
                best_n = n
                best_colors = colors
                found = True
                break
        
    return {'n': best_n, 'colors': best_colors}
