import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def backtrack_with_propagation(n, deadline, max_nodes=2000000):
        """
        Backtracking search with constraint propagation (forward checking).
        Assigns colors to triples in lexicographic order.
        For each assignment, checks if it creates an immediate violation.
        Uses randomization for tie-breaking to explore different regions.
        """
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        
        if num_triples == 0:
            return []
        
        # Precompute: for each triple, which quads does it belong to (for red K4 check)
        # A triple (a,b,c) is in quad (a,b,c,d) for each d > c
        # Also in quads where it's not the "first three": e.g., (a,b,d) with c<d, etc.
        # Actually, triple (a,b,c) with a<b<c appears in:
        #   quads (a,b,c,d) for d > c
        #   quads (a,b,d,c) -> no, must be sorted: (a,b,c,d) only if d>c
        #   Wait: quad containing triple (a,b,c): the 4th vertex can be any vertex not in {a,b,c}
        #   But the quad is sorted. So quad is {a,b,c,x} sorted.
        #   If x < a: quad = (x,a,b,c) -> triples in quad: (x,a,b),(x,a,c),(x,b,c),(a,b,c)
        #   If a < x < b: quad = (a,x,b,c) -> triples: (a,x,b),(a,x,c),(a,b,c),(x,b,c)
        #   If b < x < c: quad = (a,b,x,c) -> triples: (a,b,x),(a,b,c),(a,x,c),(b,x,c)
        #   If x > c: quad = (a,b,c,x) -> triples: (a,b,c),(a,b,x),(a,c,x),(b,c,x)
        
        # For red K4: triple (a,b,c) being red means we need the other 3 triples in the quad to also be red.
        # For blue K5: triple (a,b,c) being blue means we need the other 9 triples in each quint containing it to also be blue.
        
        # Strategy: maintain arrays tracking "danger levels"
        # For each quad: count of red triples assigned so far (out of 4)
        # For each quint: count of blue triples assigned so far (out of 10)
        
        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))
        
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # For each triple, precompute which quads and quints it belongs to
        triple_quads = [[] for _ in range(num_triples)]
        triple_quints = [[] for _ in range(num_triples)]
        
        for qi, quad in enumerate(quad_list):
            a, b, c, d = quad
            quads_triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            for t in quads_triples:
                triple_quads[triple_to_idx[t]].append(qi)
        
        for qi, quint in enumerate(quint_list):
            for t in combinations(quint, 3):
                triple_quints[triple_to_idx[t]].append(qi)
        
        # State
        color = [-1] * num_triples  # -1 = unassigned
        quad_red_count = [0] * len(quad_list)
        quad_red_assigned = [0] * len(quad_list)  # how many of 4 triples are assigned
        quint_blue_count = [0] * len(quint_list)
        quint_blue_assigned = [0] * len(quint_list)  # how many of 10 triples are assigned
        
        # For backtracking, we need to undo assignments
        # We'll use a stack of (triple_idx, old_color, quad_changes, quint_changes)
        
        nodes_visited = 0
        solution = []
        found = [False]
        
        def assign(triple_idx, col):
            """Assign color col to triple_idx. Returns list of changes for undo, or None if infeasible."""
            changes = []
            t = triple_list[triple_idx]
            
            # Check red K4 constraint: for each quad containing this triple,
            # if we assign red, check if all other 3 are already red (violation)
            # If we assign blue, it's safe for red K4
            if col == 0:  # red
                for qi in triple_quads[triple_idx]:
                    if quad_red_assigned[qi] == 4 - 1:  # 3 others already assigned
                        if quad_red_count[qi] == 3:
                            # All other 3 are red -> violation
                            return None
            # Check blue K5 constraint: for each quint containing this triple,
            # if we assign blue, check if all other 9 are already blue (violation)
            if col == 1:  # blue
                for qi in triple_quints[triple_idx]:
                    if quint_blue_assigned[qi] == 10 - 1:  # 9 others already assigned
                        if quint_blue_count[qi] == 9:
                            # All other 9 are blue -> violation
                            return None
            
            # Also check: if assigning red, and quad already has 3 reds among assigned, that's a violation
            # Wait, I need to be more careful. Let me re-check.
            # If I assign red to triple t in quad q:
            #   New red count = old red count + 1
            #   New assigned count = old assigned count + 1
            #   Violation if new red count == 4 and new assigned count == 4
            #   i.e., old red count == 3 and old assigned count == 3
            # This is what I have above.
            
            # If I assign blue to triple t in quint q:
            #   New blue count = old blue count + 1
            #   New assigned count = old assigned count + 1
            #   Violation if new blue count == 10 and new assigned count == 10
            #   i.e., old blue count == 9 and old assigned count == 9
            # This is what I have above.
            
            # Apply the assignment
            color[triple_idx] = col
            
            for qi in triple_quads[triple_idx]:
                old_red = quad_red_count[qi]
                old_assigned = quad_red_assigned[qi]
                quad_red_assigned[qi] = old_assigned + 1
                if col == 0:
                    quad_red_count[qi] = old_red + 1
                changes.append(('quad', qi, old_red, old_assigned))
            
            for qi in triple_quints[triple_idx]:
                old_blue = quint_blue_count[qi]
                old_assigned = quint_blue_assigned[qi]
                quint_blue_assigned[qi] = old_assigned + 1
                if col == 1:
                    quint_blue_count[qi] = old_blue + 1
                changes.append(('quint', qi, old_blue, old_assigned))
            
            return changes
        
        def unassign(triple_idx, changes):
            """Undo an assignment."""
            color[triple_idx] = -1
            for change in changes:
                if change[0] == 'quad':
                    _, qi, old_red, old_assigned = change
                    quad_red_count[qi] = old_red
                    quad_red_assigned[qi] = old_assigned
                else:
                    _, qi, old_blue, old_assigned = change
                    quint_blue_count[qi] = old_blue
                    quint_blue_assigned[qi] = old_assigned
        
        def search(idx, depth):
            nonlocal nodes_visited, found
            if found[0]:
                return True
            
            if time.time() > deadline - 5:
                return False
            
            if idx == num_triples:
                found[0] = True
                return True
            
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return False
            
            t = triple_list[idx]
            
            # Try both colors, with randomization
            order = [0, 1]
            if random.random() < 0.5:
                order = [1, 0]
            
            for col in order:
                changes = assign(idx, col)
                if changes is not None:
                    if search(idx + 1, depth + 1):
                        return True
                    unassign(idx, changes)
            
            return False
        
        # Use a heuristic: assign triples in an order that maximizes constraint propagation
        # Actually, lexicographic order is fine for now. Let's try it.
        
        search(0, 0)
        
        if found[0]:
            return color
        return None

    def heuristic_backtrack(n, deadline, max_nodes=500000):
        """
        Heuristic backtracking: pick the next triple to assign based on which 
        has the most constraints (most quads/quints it belongs to that are "close to violation").
        """
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        
        if num_triples == 0:
            return []
        
        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))
        
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        triple_quads = [[] for _ in range(num_triples)]
        triple_quints = [[] for _ in range(num_triples)]
        
        for qi, quad in enumerate(quad_list):
            a, b, c, d = quad
            quads_triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            for t in quads_triples:
                triple_quads[triple_to_idx[t]].append(qi)
        
        for qi, quint in enumerate(quint_list):
            for t in combinations(quint, 3):
                triple_quints[triple_to_idx[t]].append(qi)
        
        color = [-1] * num_triples
        assigned = [False] * num_triples
        quad_red_count = [0] * len(quad_list)
        quad_red_assigned = [0] * len(quad_list)
        quint_blue_count = [0] * len(quint_list)
        quint_blue_assigned = [0] * len(quint_list)
        
        nodes_visited = 0
        found = [False]
        
        def get_unassigned():
            return [i for i in range(num_triples) if not assigned[i]]
        
        def select_next(unassigned_list):
            """Select the next triple to assign using a heuristic."""
            best_idx = None
            best_score = -1
            # Sample up to 50 candidates for speed
            candidates = unassigned_list
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)
            for idx in candidates:
                score = 0
                for qi in triple_quads[idx]:
                    # Score higher if quad is close to red violation
                    remaining = 4 - quad_red_assigned[qi]
                    if remaining <= 2:
                        score += 10 - remaining * 5
                for qi in triple_quints[idx]:
                    remaining = 10 - quint_blue_assigned[qi]
                    if remaining <= 3:
                        score += 20 - remaining * 8
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx
        
        def assign(idx, col):
            changes = []
            if col == 0:
                for qi in triple_quads[idx]:
                    if quad_red_assigned[qi] == 3 and quad_red_count[qi] == 3:
                        return None
            if col == 1:
                for qi in triple_quints[idx]:
                    if quint_blue_assigned[qi] == 9 and quint_blue_count[qi] == 9:
                        return None
            
            color[idx] = col
            assigned[idx] = True
            
            for qi in triple_quads[idx]:
                old_red = quad_red_count[qi]
                old_assigned = quad_red_assigned[qi]
                quad_red_assigned[qi] = old_assigned + 1
                if col == 0:
                    quad_red_count[qi] = old_red + 1
                changes.append(('quad', qi, old_red, old_assigned))
            
            for qi in triple_quints[idx]:
                old_blue = quint_blue_count[qi]
                old_assigned = quint_blue_assigned[qi]
                quint_blue_assigned[qi] = old_assigned + 1
                if col == 1:
                    quint_blue_count[qi] = old_blue + 1
                changes.append(('quint', qi, old_blue, old_assigned))
            
            return changes
        
        def unassign(idx, changes):
            color[idx] = -1
            assigned[idx] = False
            for change in changes:
                if change[0] == 'quad':
                    _, qi, old_red, old_assigned = change
                    quad_red_count[qi] = old_red
                    quad_red_assigned[qi] = old_assigned
                else:
                    _, qi, old_blue, old_assigned = change
                    quint_blue_count[qi] = old_blue
                    quint_blue_assigned[qi] = old_assigned
        
        def search():
            nonlocal nodes_visited, found
            if found[0]:
                return True
            if time.time() > deadline - 5:
                return False
            if nodes_visited > max_nodes:
                return False
            
            unassigned = get_unassigned()
            if not unassigned:
                found[0] = True
                return True
            
            nodes_visited += 1
            idx = select_next(unassigned)
            
            order = [0, 1]
            if random.random() < 0.5:
                order = [1, 0]
            
            for col in order:
                changes = assign(idx, col)
                if changes is not None:
                    if search():
                        return True
                    unassign(idx, changes)
            
            return False
        
        search()
        
        if found[0]:
            return color
        return None

    # Strategy: Try backtracking for various n, starting from smaller
    targets = [20, 25, 30, 34]
    
    for n in targets:
        if time.time() > deadline - 20:
            break
        
        current_deadline = time.time() + min(30, deadline - time.time() - 10)
        
        # Try simple backtracking first
        result = backtrack_with_propagation(n, current_deadline, max_nodes=1000000)
        if result is not None:
            if is_valid(n, result):
                best_n = n
                best_colors = result
                continue
        
        # Try heuristic backtracking
        if time.time() < deadline - 20:
            current_deadline = time.time() + min(30, deadline - time.time() - 10)
            result = heuristic_backtrack(n, current_deadline, max_nodes=500000)
            if result is not None:
                if is_valid(n, result):
                    best_n = n
                    best_colors = result
                    continue

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
