def apply_cycle(cycle, element):
    # cycle is a tuple like (0, 1, 2)
    # element is an int
    # If element is in cycle, return next
    # Else return element
    try:
        idx = cycle.index(element)
        return cycle[(idx + 1) % len(cycle)]
    except ValueError:
        return element

def permute_block(block, perm):
    # perm is a tuple of cycles
    # Apply all cycles
    b = list(block)
    for cycle in perm:
        b = [apply_cycle(cycle, x) for x in b]
    return tuple(sorted(b))
