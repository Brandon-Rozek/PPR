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
from typing import List, Set, Tuple, FrozenSet, Dict

@dataclass
class Proposition:
    name: str
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return f"({self.name})"

State = FrozenSet[Proposition]


Plausibility = float
PosDist = Dict[State, Plausibility]

def new_pos_dist():
    return defaultdict(lambda : 0)

def is_valid_pos_dist(dist: PosDist) -> bool:
    return max(dist.values()) == 1

@dataclass
class PossEffect:
    positive_discriminants: Set[Proposition]
    negative_discriminants: Set[Proposition]
    # (plausibility, add effects, delete effects)
    consequents: List[Tuple[Plausibility, Set[Proposition], Set[Proposition]]]

    def is_valid_pos_dist(self) -> bool:
        return max(self.consequents, key=lambda c: c[0])[0] == 1

def apply_consequent(eij: Tuple[Set[Proposition], Set[Proposition]], s: State) -> State:
    """
    State resulting from the change on s caused by eij.
    Called "Res" in the paper.
    """
    eij_pos = eij[0]
    eij_neg = eij[1]
    return frozenset(eij_pos | {l for l in s if l not in eij_neg})


def satisfies(state: State, effect: PossEffect) -> bool:
    pos_satisfied = all((pd in state for pd in effect.positive_discriminants))
    if not pos_satisfied:
        return False
    
    neg_satisfied = all((nd not in state for nd in effect.negative_discriminants))
    return neg_satisfied

@dataclass
class PosAction:
    effects: List[PossEffect]

    def is_valid(self, state_space: List[State]) -> bool:
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
    
    def find_applicable_effect(self, state: State) -> PossEffect:
        """
        Returns the first PosEffect whose discriminent is
        satisfied by the state provided.
        """
        for effect in self.effects:
            if satisfies(state, effect):
                return effect
        raise ValueError("PosAction is not valid")

# Apply an action to a PosDist
# PosDist x PosAction -> PosDist

def compute_next_pos_from_state_action(s: State, action: PosAction) -> PosDist:
    """
    Definition 3 from the paper describing π[s′ ∣ s, a]

    """
    # Start off with every next state having 0 plausibility
    next_dist = new_pos_dist()
    # Enforces that s ∈ S(t_i)
    effect = action.find_applicable_effect(s)
    for (plausibility, add_effects, del_effects) in effect.consequents:
        next_state = apply_consequent((add_effects, del_effects), s)
        # Keep the highest plausibilty consequent
        next_dist[next_state] = max(next_dist[next_state], plausibility)

    return next_dist

def compute_next_pos_from_pos_action(dist: PosDist, action: PosAction) -> PosDist:
    """
    Second equation from definition 3 describing π[s′ ∣ π_init, a]
    """
    # Start off with every next state having 0 plausibility
    next_dist = new_pos_dist()
    for state in dist:
        # For every state s we're going to compute π[s′ ∣ s, a]
        next_dist_s = compute_next_pos_from_state_action(state, action)
        for next_state in next_dist_s:
            # Keep in mind the plausibility of the current
            # state when determining the next state
            p = min(next_dist_s[next_state], dist[state])
            # Maximize over the state space in dist
            next_dist[next_state] = max(next_dist[next_state], p)
    return next_dist
