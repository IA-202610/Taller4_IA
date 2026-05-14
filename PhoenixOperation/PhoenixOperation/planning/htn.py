from __future__ import annotations

from collections import deque

from planning.pddl import Action, Problem, apply_action, is_applicable
from planning.domain import MOVE, PICKUP, PUTDOWN, RESCUE, SETUP_SUPPLIES


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    queue = deque([list(hlas)])

    while queue:
        plan = queue.popleft()

        if is_plan_primitive(plan):
            state = problem.initial_state
            for action in plan:
                if not is_applicable(state, action):
                    break
                state = apply_action(state, action)
            else:
                if problem.isGoalState(state):
                    return plan
            continue

        for i, step in enumerate(plan):
            if not is_primitive(step):
                for refinement in step.refinements:
                    queue.append(plan[:i] + list(refinement) + plan[i + 1:])
                break

    return []


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    robot = problem.objects["robots"][0]
    cells = problem.objects["cells"]
    supplies = problem.objects["supplies"]
    patients = problem.objects["patients"]
    medical_posts = problem.objects["medical_posts"]

    if not supplies or not patients or not medical_posts:
        return []

    adj: dict = {c: [] for c in cells}
    supply_pos: dict = {}
    patient_pos: dict = {}
    robot_pos = None

    for fluent in problem.initial_state:
        if fluent[0] == "Adjacent":
            adj[fluent[1]].append(fluent[2])
        elif fluent[0] == "At":
            if fluent[1] == robot:
                robot_pos = fluent[2]
            elif fluent[1] in supplies:
                supply_pos[fluent[1]] = fluent[2]
            elif fluent[1] in patients:
                patient_pos[fluent[1]] = fluent[2]

    if robot_pos is None:
        return []

    def shortest_path(start, end):
        if start == end:
            return [start]
        visited = {start}
        q: deque = deque([[start]])
        while q:
            path = q.popleft()
            for nxt in adj[path[-1]]:
                if nxt not in visited:
                    new_path = path + [nxt]
                    if nxt == end:
                        return new_path
                    visited.add(nxt)
                    q.append(new_path)
        return []

    navigate: dict = {}
    for fc in cells:
        for tc in cells:
            path = shortest_path(fc, tc)
            if len(path) <= 1:
                refs: list = [[]] if fc == tc else []
            else:
                moves = [
                    MOVE.ground({"r": robot, "from_cell": path[i], "to_cell": path[i + 1]})
                    for i in range(len(path) - 1)
                ]
                refs = [moves]
            navigate[(fc, tc)] = HLA(f"Navigate({fc},{tc})", refinements=refs)

    missions: list[HLA] = []
    current_pos = robot_pos

    for s, p in zip(supplies, patients):
        m = medical_posts[0]
        s_pos = supply_pos.get(s)
        p_pos = patient_pos.get(p)
        if s_pos is None or p_pos is None:
            continue

        pickup_s = PICKUP.ground({"r": robot, "obj": s, "loc": s_pos})
        setup = SETUP_SUPPLIES.ground({"r": robot, "s": s, "loc": m})
        prepare_hla = HLA(f"PrepareSupplies({s},{m})", refinements=[
            [navigate[(current_pos, s_pos)], pickup_s, navigate[(s_pos, m)], setup]
        ])

        pickup_p = PICKUP.ground({"r": robot, "obj": p, "loc": p_pos})
        putdown_p = PUTDOWN.ground({"r": robot, "obj": p, "loc": m})
        extract_hla = HLA(f"ExtractPatient({p},{m})", refinements=[
            [navigate[(m, p_pos)], pickup_p, navigate[(p_pos, m)], putdown_p]
        ])

        rescue = RESCUE.ground({"r": robot, "p": p, "loc": m})
        mission = HLA(f"FullRescueMission({s},{p},{m})", refinements=[
            [prepare_hla, extract_hla, rescue]
        ])
        missions.append(mission)
        current_pos = m

    return missions
