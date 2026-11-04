import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    D = 11
    
    # Generate all supports (combinations of 4 indices from 11)
    supports = list(itertools.combinations(range(D), 4))
    num_supports = len(supports)  # 330
    
    # Precompute support masks for fast intersection checks
    support_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks.append(mask)
    
    # Build conflict graph adjacency as bitmasks
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict
    conflict_masks = [0] * num_supports
    for i in range(num_supports):
        for j in range(i + 1, num_supports):
            inter = support_masks[i] & support_masks[j]
            if bin(inter).count('1') >= 3:
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
    
    # Helper to count bits in an integer (for popcount)
    # Python 3.10+ has int.bit_count(), but we'll use bin().count('1') for compatibility if needed,
    # though bit_count is faster. Let's assume modern Python or use a fast method.
    # Given the environment, bin().count('1') is safe.
    
    # Simulated Annealing
    def simulated_annealing():
        # Initial state: Random selection or greedy start
        # Greedy start is often better for SA initialization
        current_selection = [False] * num_supports
        # Greedy initialization: pick nodes with lowest degree in remaining graph?
        # Simpler: Random initialization with some density, or just empty and let SA build?
        # Let's try a greedy heuristic first to get a decent start.
        
        # Greedy MIS approximation
        remaining = set(range(num_supports))
        selected = []
        while remaining:
            # Pick node with min degree in remaining subgraph
            min_deg = float('inf')
            min_node = -1
            for node in remaining:
                deg = bin(conflict_masks[node] & (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF if num_supports < 32 else (1 << num_supports) - 1)).count('1')
                # Actually, we need to mask with remaining.
                # This is slow in Python. Let's use a simpler random start or a very fast greedy.
                pass
            
            # Faster Greedy:
            # Just pick random nodes if no conflict? No, let's use a simple greedy:
            # Iterate nodes, if no conflict with already selected, add it.
            # This is O(N^2) but N=330, so 100k ops. Fast.
            selected = []
            for i in range(num_supports):
                # Check if i conflicts with any in selected
                conflict = False
                for s_idx in selected:
                    if (conflict_masks[i] >> s_idx) & 1:
                        conflict = True
                        break
                if not conflict:
                    selected.append(i)
            
            for s in selected:
                current_selection[s] = True
                
        current_score = len(selected)
        best_selection = current_selection[:]
        best_score = current_score
        
        # SA Parameters
        T = 100.0
        T_min = 0.1
        alpha = 0.9995  # Cooling rate
        # Number of iterations per temperature? Or just total time budget?
        # We have ~140 seconds.
        
        # Precompute list of indices for faster iteration?
        # N=330.
        
        while time.time() < deadline and T > T_min:
            # Perform one move: flip a random bit? Or swap?
            # Flipping a single bit is standard for binary SA.
            i = random.randint(0, num_supports - 1)
            
            # Calculate delta score
            # If we flip i:
            # If current_selection[i] is True, we remove it. Score decreases by 1.
            #   But we might be able to add others? No, SA step is usually single flip.
            #   Removing i always decreases score by 1.
            #   Wait, if we remove i, we don't gain anything immediately in the score count,
            #   but we might allow adding others in *future* steps.
            #   Standard SA for MIS:
            #   Move: Add a node (if valid) -> Score +1.
            #   Move: Remove a node -> Score -1.
            #   Move: Swap?
            
            # Let's stick to single bit flips.
            if current_selection[i]:
                # Removing i
                new_score = current_score - 1
                delta = new_score - current_score # -1
            else:
                # Adding i
                # Check if i conflicts with any currently selected node
                conflict = False
                for j in range(num_supports):
                    if current_selection[j] and ((conflict_masks[i] >> j) & 1):
                        conflict = True
                        break
                if conflict:
                    # Cannot add, invalid move. Skip or penalize?
                    # Usually, we only consider valid moves or use penalty.
                    # Let's skip invalid moves to keep state valid.
                    continue
                new_score = current_score + 1
                delta = 1
            
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # delta is -1
                p = random.random()
                accept = p < (2.718281828 ** (delta / T)) # exp(-1/T)
            
            if accept:
                current_selection[i] = not current_selection[i]
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selection = current_selection[:]
            
            T *= alpha
            
            # Optimization: If T is very small, delta -1 is almost never accepted.
            # We can speed up by checking if we can just do local search (greedy improvement)
            # when T is low.
            if T < 1.0:
                # Greedy improvement pass
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(num_supports):
                        if not current_selection[i]:
                            conflict = False
                            for j in range(num_supports):
                                if current_selection[j] and ((conflict_masks[i] >> j) & 1):
                                    conflict = True
                                    break
                            if not conflict:
                                current_selection[i] = True
                                current_score += 1
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_selection = current_selection[:]
                                break # Restart loop to check again? Or just continue?
                                # Breaking and restarting ensures we don't miss cascading additions.
        
        return best_selection

    best_selection = simulated_annealing()
    
    # Generate vectors from best_selection
    vectors = []
    for i in range(num_supports):
        if best_selection[i]:
            s = supports[i]
            for cfg in range(16):
                v = np.zeros(D, dtype=np.int8)
                for k in range(4):
                    sgn = 1 if ((cfg >> k) & 1) else -1
                    v[s[k]] = sgn
                vectors.append(v)
    
    if not vectors:
        # Fallback
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for s_i in (-1, 1):
                    for s_j in (-1, 1):
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = s_i
                        v[j] = s_j
                        vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
