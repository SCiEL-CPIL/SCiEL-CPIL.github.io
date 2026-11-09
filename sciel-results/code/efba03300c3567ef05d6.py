"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80
    
    def is_valid_prefix(seq):
        """Check if seq has all distinct subset sums."""
        n = len(seq)
        sums = {0}
        for x in seq:
            sums = sums | {s + x for s in sums}
        return len(sums) == (1 << n)
    
    # Greedy: pick smallest valid candidate at each step
    seq = []
    current_sums = {0}
    for i in range(12):
        used = set(seq)
        candidate = 1
        while True:
            if candidate not in used:
                new_sums = current_sums | {s + candidate for s in current_sums}
                if len(new_sums) == (1 << (i + 1)):
                    seq.append(candidate)
                    current_sums = new_sums
                    break
            candidate += 1
    
    return seq
# EVOLVE-BLOCK-END
