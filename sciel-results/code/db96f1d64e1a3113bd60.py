"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def max_sum_free_subset(elements):
        """
        Find a large sum-free subset using greedy approach.
        Process elements by decreasing absolute value.
        An element can be added if it doesn't create a+b=c violation.
        """
        if not elements:
            return []

        # Sort by decreasing absolute value for better greedy results
        sorted_elems = sorted(elements, key=lambda x: -abs(x))

        selected = []
        selected_set = set()

        for elem in sorted_elems:
            # Check if adding elem would violate sum-free property
            # Violation 1: exists a, b in selected such that a + b == elem
            # Violation 2: exists a in selected such that a + elem == b for some b in selected
            #              i.e., elem = b - a for some a, b in selected
            # Violation 3: elem + elem == b for some b in selected (i.e., 2*elem in selected)
            #              This is covered by violation 2 if we consider a=elem, but elem not yet in set
            # Actually, we need to check:
            # For sum-free: no a+b=c where a,b,c all in final set
            # If we add elem, then:
            #   - For any a,b in selected: a+b != elem
            #   - For any a in selected: a+elem != b for any b in selected, i.e., elem != b-a for any a,b in selected
            #   - elem+elem != b for any b in selected, i.e., 2*elem not in selected

            can_add = True

            # Check: is elem = a + b for any a, b in selected?
            for a in selected:
                if (elem - a) in selected_set:
                    can_add = False
                    break

            if can_add:
                # Check: is elem = b - a for any a, b in selected? (i.e., a + elem = b)
                for b in selected:
                    if (b - elem) in selected_set:
                        can_add = False
                        break

            if can_add:
                # Check: is 2*elem in selected? (elem + elem = c)
                if (2 * elem) in selected_set:
                    can_add = False

            if can_add:
                selected.append(elem)
                selected_set.add(elem)

        return selected

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        subset = max_sum_free_subset(elems)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
