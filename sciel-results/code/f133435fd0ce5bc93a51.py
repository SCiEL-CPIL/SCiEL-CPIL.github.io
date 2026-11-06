import numpy as np
import random

N = 64

def construct():
    points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    best_set = []
    beam_width = 20
    max_depth = 50 # Max points to add
    
    # We will run this beam search a few times with different random initializations if needed,
    # but starting from empty is deterministic if we break ties randomly.
    
    for run in range(10):
        random.seed(run)
        beam = [([], {})] # List of (selected_list, dist_sets)
        
        for step in range(max_depth):
            next_beam = []
            seen_states = set() # To avoid duplicate states? States are defined by set of points.
            
            for selected, dist_sets in beam:
                # Find all valid points
                valid_points = []
                for p in points:
                    if p in selected:
                        continue
                    valid = True
                    dists_from_p = set()
                    for a in selected:
                        d_pa = dist_sq(p, a)
                        if d_pa in dists_from_p:
                            valid = False
                            break
                        dists_from_p.add(d_pa)
                        if d_pa in dist_sets[a]:
                            valid = False
                            break
                    if valid:
                        valid_points.append(p)
                
                if not valid_points:
                    continue
                
                # Pick top beam_width points from valid_points
                # To add diversity, we can sample or pick random ones.
                # If we pick the "best" ones, we need a metric.
                # Let's just pick random valid points to maintain diversity.
                if len(valid_points) > beam_width:
                    chosen = random.sample(valid_points, beam_width)
                else:
                    chosen = valid_points
                
                for p in chosen:
                    new_selected = selected + [p]
                    new_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                    new_dist_sets[p] = set()
                    for a in selected:
                        d = dist_sq(p, a)
                        new_dist_sets[p].add(d)
                        new_dist_sets[a].add(d)
                    
                    # Create a unique key for the state to avoid duplicates in beam
                    key = frozenset(new_selected)
                    if key not in seen_states:
                        seen_states.add(key)
                        next_beam.append((new_selected, new_dist_sets))
            
            if not next_beam:
                break
            
            # Sort next_beam by size (descending) and keep top beam_width
            # All have same size (step+1), so we just keep first beam_width
            # Or randomize if we want diversity
            random.shuffle(next_beam)
            beam = next_beam[:beam_width]
            
            # Update best
            if len(beam[0][0]) > len(best_set):
                best_set = beam[0][0]
        
        # Update best after run
        if len(beam[0][0]) > len(best_set):
            best_set = beam[0][0]
            
    return best_set
