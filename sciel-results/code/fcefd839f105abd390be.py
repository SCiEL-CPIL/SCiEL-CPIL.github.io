            index = np.arange(dim)
            c_bit = (index >> (qubits - 1 - control)) & 1
            # ...
            c_val = (index & c_mask) > 0
            new_index = index ^ (c_val * mask)
            matrix = matrix[new_index, :]
