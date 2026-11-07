"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def update(state, item):
    """
    Update the state with a new item.
    State is a bytes object of length 400, where state[i] is the count of item i.
    """
    # Convert to bytearray for mutation
    b = bytearray(state)
    # Increment the count for the item
    # Byte overflow will wrap around, but counts are expected to be < 256
    b[item] = (b[item] + 1) % 256
    return bytes(b)

def query(state, item):
    """
    Query the estimated frequency of an item.
    State is a bytes object of length 400.
    """
    # Direct lookup
    return state[item]
# EVOLVE-BLOCK-END
