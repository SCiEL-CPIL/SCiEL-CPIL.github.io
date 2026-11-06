<<<<<<< SEARCH
    # Final attempt: try to find a coloring with fewer colors using structured patterns
    if time.time() < deadline - 2.0:
=======
    def old_violation_check(idx, colors, role, new_color):
        """Check if AP idx would be violated with new_color at role."""
        p = ap_pos[idx]
        c = [colors[pi] for pi in p]
        c[role] = new_color
        return ap_violation(c[0], c[1], c[2], c[3])

    # Final attempt: try to find a coloring with fewer colors using structured patterns
    if time.time() < deadline - 2.0:
>>>>>>> REPLACE

