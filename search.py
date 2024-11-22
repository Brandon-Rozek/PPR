"""
Search algorithms for finding possibilistic plans
"""
from typing import Optional
from queue import Queue
from pp import *

def search_single_gamma_acceptable(problem: PosPlanningProblem, gamma: float) -> Optional[List[PosAction]]:
    """
    Return a single plan whose necessity value of reaching the goal
    is equal to or above gamma.
    """
    search_queue: Queue[Tuple[PosDist, List[PosAction]]] = Queue()
    search_queue.put((problem.initial_distribution, []))

    seen: Set[PosDist] = set()
    seen.add(problem.initial_distribution)

    while not search_queue.empty():
        pladist, plan = search_queue.get()

        goal_necessity = compute_nec_from_pos(problem.goal_states, pladist)
        if goal_necessity >= gamma:
            return plan

        for action in problem.actions:
            # Apply action on current pladist
            next_pladist = compute_posdist_from_curpos_action(pladist, action)
            if next_pladist not in seen:
                search_queue.put((next_pladist, plan + [action]))
                seen.add(next_pladist)

    # No gamma acceptable plan found
    return None

