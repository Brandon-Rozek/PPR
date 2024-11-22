"""
Representation of Possibilistic Planning as defined in
"Possibilistic Planning: Representation and Complexity" by
Célia Da Costa Pereira, Frédérick Garcia, Jérôme Lange & Roger Martin-Clouaire
in ECP 1997.

Slight adaption to match a more STRIPS-style formalism.
Absense of proposition implies falsity
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import List, Set, Tuple, FrozenSet, Dict

@dataclass
class Proposition:
    name: str

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"({self.name})"

State = FrozenSet[Proposition]
StateSpace = FrozenSet[State]


Plausibility = float

class PosDist:
    def __init__(self, state_space: StateSpace):
        self.state_space = state_space
        self.plausibilities: Dict[State, Plausibility] = {}

    def __getitem__(self, state: State):
        return self.plausibilities.get(state, 0)

    def __setitem__(self, state: State, p: Plausibility):
        self.plausibilities[state] = p

    def is_valid(self) -> bool:
        # Ensure that the keys of the plausibility map
        # are within the state space
        if not all(k in self.state_space for k in self.plausibilities.keys()):
            return False

        # Ensure that the possibility distribution is normalized
        # i.e. the maximum plausibility value is equal to 1
        if not max(self.plausibilities.values()) == 1:
            return False

        # Ensure that the plausibility values live between 0 and 1
        for p in self.plausibilities.values():
            if p < 0 or p > 1:
                return False

        return True

    def non_zero_states(self):
        """
        Return a generator of the states
        that have a non-zero plausibility value
        """
        return (s for (s, p) in self.plausibilities.items() if p > 0)

    def __str__(self):
        """
        Only print the states and their plausibilities when
        its non-zero.
        """
        return str({kv for kv in self.plausibilities.items() if kv[1] > 0})
        # return str(self.plausibilities)

@dataclass
class PosEffect:
    positive_discriminants: Set[Proposition]
    negative_discriminants: Set[Proposition]
    # (plausibility, add effects, delete effects)
    consequents: List[Tuple[Plausibility, Set[Proposition], Set[Proposition]]]

    def is_valid_pos_dist(self) -> bool:
        # Ensure that the plausibility values live
        # between 0 and 1
        for c in self.consequents:
            if c[0] < 0 or c[0] > 1:
                return False

        # Ensure that the possibility distribution is normalized
        # i.e the maximum plausibility value is 1
        return max(self.consequents, key=lambda c: c[0])[0] == 1

def apply_consequent(eij: Tuple[Set[Proposition], Set[Proposition]], s: State) -> State:
    """
    State resulting from the change on s caused by eij.
    Called "Res" in the paper.
    """
    eij_pos = eij[0]
    eij_neg = eij[1]
    return frozenset(eij_pos | {l for l in s if l not in eij_neg})


def satisfies(state: State, effect: PosEffect) -> bool:
    """
    Returns whether a state satisfies the discriminent of a
    Possibilistic Effect e_i.
    """
    pos_satisfied = all((pd in state for pd in effect.positive_discriminants))
    if not pos_satisfied:
        return False

    neg_satisfied = all((nd not in state for nd in effect.negative_discriminants))
    return neg_satisfied

@dataclass
class PosAction:
    effects: List[PosEffect]

    def is_valid(self, state_space: StateSpace) -> bool:
        # Every effect must have a valid possibilistic distribution
        valid_pos_dist = all((e.is_valid_pos_dist() for e in self.effects))

        if not valid_pos_dist:
            return False

        # Make sure only one effect can fire at a given state
        for state in state_space:
            num_satisfied = 0
            for effect in self.effects:
                if satisfies(state, effect):
                    num_satisfied += 1

                if num_satisfied > 1:
                    break

            if num_satisfied != 1:
                return False

        return True

    def find_applicable_effect(self, state: State) -> PosEffect:
        """
        Returns the first PosEffect whose discriminent is
        satisfied by the state provided.
        """
        for effect in self.effects:
            if satisfies(state, effect):
                return effect
        raise ValueError("PosAction is not valid")

def compute_posdist_from_state_action(s: State, action: PosAction, state_space: Set[State]) -> PosDist:
    """
    Returns a PosDist representing a vector where each element is
    π[s′ ∣ s, a] for each s′ in the state space.

    Derived from definition 3 from the paper describing
    π[s′ ∣ s, a].
    """
    next_dist = PosDist(state_space)
    # Enforces that s ∈ S(t_i)
    effect = action.find_applicable_effect(s)
    for (plausibility, add_effects, del_effects) in effect.consequents:
        # Compute s′ = Res(eik, s)
        next_state = apply_consequent((add_effects, del_effects), s)
        # Keep the highest plausibilty consequent
        next_dist[next_state] = max(next_dist[next_state], plausibility)

    return next_dist

def compute_posdist_from_curpos_action(dist: PosDist, action: PosAction) -> PosDist:
    """
    Returns a PosDist representing a vector where each element is
    π[sN ∣ π_last, a] for each sN in the state space.

    Derived from the second equation from definition 3
    describing π[Goals ∣ π_init, <a_i>].
    """
    next_dist = PosDist(dist.state_space)
    # NOTE: If π_last(s0) = 0 then π[sN ∣ s0, a] = 0.
    # This is the default behavior of PosDist meaning we
    # can focus only on non-zero states.
    for s0 in dist.non_zero_states():
        # Compute π[sN ∣ s0, a] for every state sN
        next_dist_s = compute_posdist_from_state_action(s0, action, dist.state_space)
        # NOTE: If π[sN ∣ s0, a] = 0 then we can skip s0 for determining max_s0 min(π[sN ∣ s0, a], π_last(s0))
        for sN in next_dist_s.non_zero_states():
            # Keep in mind the plausibility of the current
            # state when determining the next state's
            p = min(next_dist_s[sN], dist[s0])
            # Maximize over π[sN ∣ s0, a] for all s0s considered so far
            next_dist[sN] = max(next_dist[sN], p)
    return next_dist

def compute_necdist_from_pos_action(dist: PosDist, action: PosAction) -> PosDist:
    """
    Returns a PosDist representing a vector where each element is
    N[sN ∣ π_last, a] for each sN in the state space.

    Derived from the third equation from definition 3
    describing N[Goals ∣ π_init, <a_i>].
    """
    state_space = dist.state_space
    next_dist = PosDist(state_space)

    # First assign everything with a necessity of 1
    # and the algorithm will iteratively take the min
    # of this and π[s ∣ s0, a] for every s0
    for s in state_space:
        next_dist[s] = 1

    for s0 in state_space:
        # Compute π[sNC ∣ s0, a] for every state sNC
        next_dist_s = compute_posdist_from_state_action(s0, action, state_space)
        for sN, sNC in product(state_space, state_space):
            # Only look at the compliment of sN
            if sN != sNC:
                p = max(1 - dist[s0], 1 - next_dist_s[sNC])
                next_dist[sN] = min(next_dist[sN], p)

    return next_dist

def compute_nec_from_pos_action(states: List[State], dist: PosDist, action: PosAction) -> Plausibility:
    """
    Returns the value of N[states ∣ π_last, a] as a necessity value.

    Derived from the third equation from definition 3
    describing N[Goals ∣ π_init, <a_i>].
    """
    state_space = dist.state_space
    nvalue = 1

    for s0 in state_space:
        # Compute π[sNC ∣ s0, a] for every state sNC
        next_dist_s = compute_posdist_from_state_action(s0, action, state_space)
        for sNC in state_space:
            # Only consider the compliment of states
            if sNC not in states:
                p = max(1 - dist[s0], 1 - next_dist_s[sNC])
                nvalue = min(nvalue, p)

    return nvalue

class PosPlanningProblem:
    def __init__(self, initial_distribution: PosDist, actions: List[PosAction], goal: Set[Proposition]):
        self.initial_distribution = initial_distribution
        self.actions  = actions
        self.goal = goal

    def is_valid(self) -> bool:
        state_space = self.initial_distribution.state_space
        if not self.initial_distribution.is_valid():
            print("Not valid initial distributoin")
            return False

        for action in self.actions:
            if not action.is_valid(state_space):
                print("Invalid action found")
                return False

        for s in state_space:
            if self.goal < s:
                return True
        return False
