    def local_search_add(S, reachable):
        """Try to add elements from complement that can be safely added."""
        S_set = set(S)
        all_elements = set(range(1, 101))
        complement = list(all_elements - S_set)
        current_S = list(S)
        current_reachable = set(reachable)
        improved = True
        
        while improved:
            improved = False
            random.shuffle(complement)
            # Iterate over a copy to allow safe removal
            for x in list(complement):
                if can_add(current_reachable, x):
                    current_S.append(x)
                    S_set.add(x)
                    new_sums = set()
                    for s in current_reachable:
                        ns = s + x
                        if ns <= 200:
                            new_sums.add(ns)
                    current_reachable = current_reachable | new_sums
                    if x in complement:
                        complement.remove(x)
                    improved = True
        return current_S, current_reachable
