import random
import time

def construct():
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    elem_to_aps = [[] for _ in range(n + 1)]
    
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for x in ap:
                elem_to_aps[x].append(ap_idx)
                
    num_aps = len(aps)
    
    class State:
        __slots__ = ['selected', 'ap_counts']
        def __init__(self):
            self.selected = set()
            self.ap_counts = [0] * num_aps
            
        def is_valid_add(self, x):
            for ap_idx in elem_to_aps[x]:
                if self.ap_counts[ap_idx] == 3:
                    return False
            return True
            
        def add(self, x):
            self.selected.add(x)
            for ap_idx in elem_to_aps[x]:
                self.ap_counts[ap_idx] += 1
                
        def remove(self, x):
            self.selected.remove(x)
            for ap_idx in elem_to_aps[x]:
                self.ap_counts[ap_idx] -= 1
                
        def size(self):
            return len(self.selected)

    def greedy_init(random_order):
        state = State()
        for x in random_order:
            if state.is_valid_add(x):
                state.add(x)
        return state

    def local_search(state, time_limit):
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            
            # Phase 1: Add any possible element
            for x in range(1, n + 1):
                if x not in state.selected:
                    if state.is_valid_add(x):
                        state.add(x)
                        improved = True
            
            if not improved:
                # State is maximal. Try to swap.
                if state.size() == 0:
                    break
                
                # Try 1 random swap
                y = random.choice(list(state.selected))
                state.remove(y)
                
                added_count = 0
                for x in range(1, n + 1):
                    if x not in state.selected:
                        if state.is_valid_add(x):
                            state.add(x)
                            added_count += 1
                            
                if added_count == 0:
                    # Revert
                    state.add(y)
                # If added_count >= 1, we are in a new state.
                # We don't mark improved=True here because size didn't increase.
                # But we want to continue searching from this new state.
                # However, if we just swap and size is same, we might loop.
                # To avoid loops, we can limit swaps or rely on randomization.
                # Let's just let it run. The "Add" phase will run again.
                # If no add is possible, we swap again.
                # This is a random walk on maximal sets.
                # We need to break out eventually.
                # The outer while loop in construct() will restart if time runs out.
                # But we should probably limit the number of swaps in local_search to prevent infinite loops of 0-gain swaps.
                # Actually, if we do 0-gain swaps, we are just exploring.
                # Let's add a counter for swaps without improvement.
                
        return state

    # Modify local_search to handle swap limits
    def local_search_limited(state, max_swaps=20):
        swaps_done = 0
        improved = True
        while improved and swaps_done < max_swaps:
            improved = False
            
            # Phase 1: Add any possible element
            for x in range(1, n + 1):
                if x not in state.selected:
                    if state.is_valid_add(x):
                        state.add(x)
                        improved = True
                        swaps_done = 0 # Reset swap counter on improvement? No, just keep going.
            
            if not improved:
                # State is maximal. Try to swap.
                if state.size() == 0:
                    break
                
                y = random.choice(list(state.selected))
                state.remove(y)
                swaps_done += 1
                
                added_count = 0
                for x in range(1, n + 1):
                    if x not in state.selected:
                        if state.is_valid_add(x):
                            state.add(x)
                            added_count += 1
                            
                if added_count == 0:
                    state.add(y)
                    
        return state

    best_state = None
    best_size = 0
    start_time = time.time()
    time_limit = 55.0
    
    while time.time() - start_time < time_limit:
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        state = greedy_init(order)
        state = local_search_limited(state)
        
        if state.size() > best_size:
            best_size = state.size()
            best_state = state
            
    return sorted(best_state.selected)
