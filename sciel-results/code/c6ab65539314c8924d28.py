import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    best_alpha = 17
    best_edges = []
    
    all_triples = list(itertools.combinations(range(16), 3))
    triple_set = set(all_triples)
    
    def is_linear(edge_list, new_edge):
        s_new = set(new_edge)
        for e in edge_list:
            if len(s_new & set(e)) > 1:
                return False
        return True
        
    def calc_alpha(edge_list):
        if not edge_list:
            return 16
        edge_masks = [sum(1 << v for v in e) for e in edge_list]
        # Check from largest k down
        for k in range(16, 3, -1): # Alpha must be at least 4? No, check all.
            for combo in itertools.combinations(range(16), k):
                mask = sum(1 << v for v in combo)
                independent = True
                for em in edge_masks:
                    if (mask & em) == em:
                        independent = False
                        break
                if independent:
                    return k
        return 3 # Fallback, shouldn't happen if k goes down to 1
        
    def generate_maximal():
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        current_edges = []
        for t in shuffled:
            if is_linear(current_edges, t):
                current_edges.append(t)
        return current_edges
        
    def local_search(edges):
        current_edges = list(edges)
        current_alpha = calc_alpha(current_edges)
        temperature = 1.0
        cooling = 0.995
        no_improve_count = 0
        
        while time.time() - start_time < timeout:
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_edges = list(current_edges)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 500:
                    break
                    
            # Pick random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            old_edge = current_edges[idx]
            temp_edges = current_edges[:idx] + current_edges[idx+1:]
            
            # Find valid new edges
            # Sample a few random candidates to save time, or iterate all?
            # Iterating all 560 is fast enough.
            candidates = []
            for t in all_triples:
                if t == old_edge: continue
                if is_linear(temp_edges, t):
                    candidates.append(t)
            
            if not candidates:
                # Cannot swap? Try adding if possible? 
                # But it's maximal.
                # Restart local search
                break
                
            new_edge = random.choice(candidates)
            new_edges = temp_edges + [new_edge]
            new_alpha = calc_alpha(new_edges)
            
            delta = new_alpha - current_alpha
            if delta < 0:
                current_edges = new_edges
                current_alpha = new_alpha
            elif delta == 0:
                if random.random() < 0.5:
                    current_edges = new_edges
            else:
                if random.random() < pow(temperature, delta):
                    current_edges = new_edges
                    current_alpha = new_alpha
            
            temperature *= cooling
            if temperature < 0.01:
                break
                
        return current_edges, current_alpha

    # Main Loop
    # Run multiple restarts if time permits
    while time.time() - start_time < timeout:
        edges = generate_maximal()
        edges, alpha = local_search(edges)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges
            
        # If we found alpha <= 4, we can stop early as it's likely optimal
        if best_alpha <= 4:
            break
            
    return best_edges
