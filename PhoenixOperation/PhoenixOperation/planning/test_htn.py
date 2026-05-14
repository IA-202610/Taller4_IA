import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from world.rescue_layout import RescueLayout
from planning.problems import SimpleRescueProblem
from planning.htn import build_htn_hierarchy, hierarchicalSearch


def load_layout(name: str) -> RescueLayout | None:
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    layouts_dir = os.path.join(project_root, "layouts")
    filename = name if name.endswith(".lay") else name + ".lay"
    for root, _dirs, files in os.walk(layouts_dir):
        if filename in files:
            with open(os.path.join(root, filename)) as f:
                return RescueLayout([line.rstrip("\n") for line in f])
    return None


def run_test(layout_name: str) -> None:
    layout = load_layout(layout_name)
    if layout is None:
        print(f"Layout '{layout_name}' not found.")
        return

    print(f"=== Layout: {layout_name} ===")
    print(f"  Robot:         {layout.robot_position}")
    print(f"  Supplies:      {layout.supplies}")
    print(f"  Patients:      {layout.patients}")
    print(f"  Medical posts: {layout.medical_posts}")

    if not layout.patients or not layout.supplies or not layout.medical_posts:
        print("  Missing required objects (need R, T, S, M in layout).")
        return

    problem = SimpleRescueProblem(layout)
    hlas = build_htn_hierarchy(problem)

    if not hlas:
        print("  No HLAs built.")
        return

    print(f"\nTop-level tasks: {hlas}")
    plan = hierarchicalSearch(problem, hlas)

    if plan:
        print(f"\nPlan found ({len(plan)} actions):")
        for i, action in enumerate(plan, 1):
            print(f"  {i:2}. {action.name}")
    else:
        print("\nNo plan found.")


if __name__ == "__main__":
    layout_name = sys.argv[1] if len(sys.argv) > 1 else "tinyTest"
    run_test(layout_name)
