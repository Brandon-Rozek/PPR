from pp import *

state1: State = frozenset({
    Proposition("at-0"),
    Proposition("alive")
})

state2: State = frozenset({
    Proposition("at-0")
    # Not Alive
})

state3: State = frozenset({
    Proposition("at-1"),
    Proposition("alive")
})

state4: State = frozenset({
    Proposition("at-1"),
    # Not Alive
})

state5: State = frozenset({
    Proposition("at-2"),
    Proposition("alive")
})

state6: State = frozenset({
    Proposition("at-2"),
    # Not Alive
})


state_space: StateSpace = {state1, state2, state3, state4, state5, state6}
initial_distribution = PosDist(state_space)
initial_distribution[state1] = 1

print("Initial Plausibility Distribution", initial_distribution)
print("Is Valid?", initial_distribution.is_valid())
print("")

move_0_1 = PosAction([
    # No-op if dead
    PossEffect(set(), {Proposition("alive")}, [(1, set(), set())]),
    # No-op if not adjacent
    PossEffect({Proposition("alive")}, {Proposition("at-0")}, [(1, set(), set())]),
    # Otherwise move with a small plausibility of dying
    PossEffect({Proposition("alive"), Proposition("at-0")}, set(), [
        (1, {Proposition("at-1")}, {Proposition("at-0")}),
        (0.25, set(), {Proposition("alive")})
    ])
])

move_1_2 = PosAction([
    # No-op if dead
    PossEffect(set(), {Proposition("alive")}, [(1, set(), set())]),
    # No-op if not adjacent
    PossEffect({Proposition("alive")}, {Proposition("at-1")}, [(1, set(), set())]),
    # Otherwise move
    PossEffect({Proposition("alive"), Proposition("at-1")}, set(), [(1, {Proposition("at-2")}, {Proposition("at-1")})])
])

print("move_0_1 valid?", move_0_1.is_valid(state_space))
print("move_1_2 valid?", move_1_2.is_valid(state_space))
print("")

# Applying an action

pdist2 = compute_next_pos_from_pos_action(initial_distribution, move_0_1)
ndist2 = compute_next_nec_from_pos_action(initial_distribution, move_0_1)
print("Time 2 Possibility:", pdist2)
print("Time 2 Necessity:", ndist2)
print("")

pdist3 = compute_next_pos_from_pos_action(pdist2, move_1_2)
ndist3 = compute_next_nec_from_pos_action(pdist2, move_1_2)
print("Time 3 Plausibility:", pdist3)
print("Time 3 Necessity:", ndist3)

# Defining a planning problem

# Finding a plan