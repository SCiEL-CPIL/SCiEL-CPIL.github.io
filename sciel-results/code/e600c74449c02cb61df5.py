def dispatch(ready_operations, time):
        if not ready_operations:
            return 0

        best_index = 0
        best_machine = None
        best_dur = float('inf')
        best_rem_work = float('inf')
        best_rem_ops = float('inf')

        for i, op in enumerate(ready_operations):
            machine_options = op["machine_options"]
            if not machine_options:
                continue # or handle error

            # Find best machine for this op
            m_dur = min(machine_options, key=machine_options.get)
            d = machine_options[m_dur]
            
            rw = op["remaining_work"]
            ro = op["remaining_ops"]

            # Lexicographic comparison
            if d < best_dur:
                best_dur = d
                best_rem_work = rw
                best_rem_ops = ro
                best_index = i
                best_machine = m_dur
            elif d == best_dur:
                if rw < best_rem_work:
                    best_rem_work = rw
                    best_rem_ops = ro
                    best_index = i
                    best_machine = m_dur
                elif rw == best_rem_work:
                    if ro < best_rem_ops:
                        best_rem_ops = ro
                        best_index = i
                        best_machine = m_dur

        if best_machine is None:
            return 0
        return (best_index, best_machine)
