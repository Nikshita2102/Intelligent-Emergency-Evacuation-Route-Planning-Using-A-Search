"""
Intelligent Emergency Evacuation Route Planning Using A* Search

Multi-agent grid evacuation using Space-Time A* with prioritized
reservations. Agents plan routes to available emergency exits while
avoiding fire, collisions, and occupied cells.

Usage:
    python evacuation.py
    python evacuation.py normal
    python evacuation.py exit_blocked
    python evacuation.py bottleneck
    python evacuation.py trapped
"""

import heapq
import sys
from collections import deque


# ============================================================
# BUILDING GRID
# ============================================================

GRID = [
    "####################",
    "#..........#.......#",
    "#..........#.......#",
    "#..........D.......#",
    "#####.######.#######",
    "#..........#.......#",
    "#..........#.......#",
    "E..........#.......E",
    "####################",
]

H = len(GRID)
W = len(GRID[0])

# Agent can wait or move in four directions
MOVES = [
    (0, 0),      # Wait
    (1, 0),      # Down
    (-1, 0),     # Up
    (0, 1),      # Right
    (0, -1),     # Left
]

MAX_T = 80
INF = 10**9
ALGO = "astar"   # "astar" or "greedy"


# ============================================================
# EXITS AND AGENTS
# ============================================================

EXITS = [
    (r, c)
    for r in range(H)
    for c in range(W)
    if GRID[r][c] == "E"
]

BASE_AGENTS = [
    (2, 3),      # Agent 0
    (6, 4),      # Agent 1
    (2, 15),     # Agent 2
    (6, 16),     # Agent 3
]


# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = {

    # Fire source, fire spread time, extra agents
    "normal": (
        [(5, 4)],
        2,
        []
    ),

    "exit_blocked": (
        [(7, 2)],
        2,
        []
    ),

    "bottleneck": (
        [(5, 4)],
        3,
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 1),
            (1, 3)
        ]
    ),

    "trapped": (
        [(4, 5), (3, 11)],
        999,
        []
    ),

    # Fast fire (1 cell per step): some people cannot be saved
    "fast_fire": ([(5, 4)], 1, []),
    "door_fire": ([(3, 11)], 1, []),
    "crowd_fire": (
        [(3, 11)], 1,
        [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]
    ),
}


# ============================================================
# FIRE SIMULATION
# ============================================================

def fire_times(fires, spread):
    """
    Calculate the time at which each cell catches fire.

    Fire spreads deterministically from the initial fire cells.
    """

    fire_time = {}
    queue = deque()

    # Initial fire cells
    for fire in fires:
        fire_time[fire] = 0
        queue.append(fire)

    while queue:

        current = queue.popleft()
        r, c = current

        for dr, dc in MOVES[1:]:
            nr = r + dr
            nc = c + dc

            next_cell = (nr, nc)

            # Check grid boundary
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            # Fire cannot spread through walls
            if GRID[nr][nc] == "#":
                continue

            # Already processed
            if next_cell in fire_time:
                continue

            fire_time[next_cell] = (
                fire_time[current] + spread
            )

            queue.append(next_cell)

    return fire_time


# ============================================================
# MANHATTAN HEURISTIC
# ============================================================

def heuristic(position):
    """
    Manhattan distance to the nearest emergency exit.
    """

    return min(
        abs(position[0] - exit_pos[0])
        + abs(position[1] - exit_pos[1])
        for exit_pos in EXITS
    )


# ============================================================
# A* SEARCH
# ============================================================

def a_star(start, fire_time, reserved_vertices, reserved_edges):
    """
    Space-Time A* search.

    State = (position, time)

    Returns:
        path
        number of expanded nodes
    """

    counter = 0

    # Priority queue:
    # (f, counter, g, position, time)
    priority_queue = [
        (
            heuristic(start),
            counter,
            0,
            start,
            0
        )
    ]

    # Parent information for reconstructing path
    parent = {
        (start, 0): None
    }

    # Best cost found for each (position, time)
    best_g = {
        (start, 0): 0
    }

    expanded = 0

    while priority_queue:

        _, _, g, position, time = heapq.heappop(
            priority_queue
        )

        state = (position, time)

        # Ignore outdated queue entries
        if g > best_g[state]:
            continue

        expanded += 1

        # ----------------------------------------------------
        # GOAL TEST
        # ----------------------------------------------------

        if position in EXITS:

            path = []
            current = state

            while current is not None:
                path.append(current[0])
                current = parent[current]

            path.reverse()

            return path, expanded

        # ----------------------------------------------------
        # TIME LIMIT
        # ----------------------------------------------------

        if time >= MAX_T:
            continue

        # ----------------------------------------------------
        # EXPAND NEIGHBOURS
        # ----------------------------------------------------

        for dr, dc in MOVES:

            nr = position[0] + dr
            nc = position[1] + dc

            next_position = (nr, nc)

            # Check boundaries
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            # Check wall
            if GRID[nr][nc] == "#":
                continue

            next_time = time + 1

            # ------------------------------------------------
            # FIRE CHECK
            # ------------------------------------------------

            if fire_time.get(
                next_position,
                INF
            ) <= next_time:
                continue

            # ------------------------------------------------
            # AGENT COLLISION CHECK
            # ------------------------------------------------

            if (
                next_position,
                next_time
            ) in reserved_vertices:
                continue

            # ------------------------------------------------
            # SWAP COLLISION CHECK
            # ------------------------------------------------

            if (
                next_position,
                position,
                time
            ) in reserved_edges:
                continue

            # ------------------------------------------------
            # FIRE DANGER PENALTY
            # ------------------------------------------------

            remaining_fire_time = (
                fire_time.get(
                    next_position,
                    INF
                ) - next_time
            )

            danger_penalty = 0

            if remaining_fire_time <= 3:
                danger_penalty = 3

            # ------------------------------------------------
            # NEW COST
            # ------------------------------------------------

            new_g = g + 1 + danger_penalty

            next_state = (
                next_position,
                next_time
            )

            if new_g < best_g.get(
                next_state,
                INF
            ):

                best_g[next_state] = new_g

                parent[next_state] = state

                counter += 1

                h = heuristic(next_position)

                f = new_g + h if ALGO == "astar" else h

                heapq.heappush(
                    priority_queue,
                    (
                        f,
                        counter,
                        new_g,
                        next_position,
                        next_time
                    )
                )

    # No route found
    return None, expanded


# ============================================================
# RESERVATION-BASED MULTI-AGENT PLANNING
# ============================================================

def plan_all_agents(agents, fire_time, order):
    """
    Plan routes sequentially according to agent priority.

    Each successful agent reserves its cells and movements so
    that later agents do not collide with it.
    """

    reserved_vertices = set()
    reserved_edges = set()

    paths = {}
    failed = []

    total_expanded = 0
    expansion_info = {}

    for agent_id in order:

        start = agents[agent_id]

        path, expanded = a_star(
            start,
            fire_time,
            reserved_vertices,
            reserved_edges
        )

        total_expanded += expanded

        expansion_info[agent_id] = expanded

        # ----------------------------------------------------
        # NO ROUTE
        # ----------------------------------------------------

        if path is None:

            failed.append(agent_id)

            continue

        # ----------------------------------------------------
        # STORE PATH
        # ----------------------------------------------------

        paths[agent_id] = path

        # ----------------------------------------------------
        # RESERVE CELLS
        # ----------------------------------------------------

        for t, position in enumerate(path):

            reserved_vertices.add(
                (position, t)
            )

            # Reserve movement edge
            if t < len(path) - 1:

                next_position = path[t + 1]

                reserved_edges.add(
                    (
                        position,
                        next_position,
                        t
                    )
                )

    return (
        paths,
        failed,
        total_expanded,
        expansion_info
    )


# ============================================================
# RETRY MECHANISM
# ============================================================

def solve(agents, fire_time):
    """
    Perform prioritized planning.

    If an agent fails because it is blocked by higher-priority
    agents, retry by moving failed agents to the front.
    """

    order = list(range(len(agents)))

    # First attempt
    (
        paths,
        failed,
        total_expanded,
        expansion_info
    ) = plan_all_agents(
        agents,
        fire_time,
        order
    )

    solve.first_fail = failed[:]
    solve.retried = 0

    # Retry failed agents
    for attempt in range(3):

        if not failed:
            break

        # Move failed agents to the front
        new_order = (
            failed
            + [
                i
                for i in order
                if i not in failed
            ]
        )

        (
            new_paths,
            new_failed,
            new_expanded,
            new_info
        ) = plan_all_agents(
            agents,
            fire_time,
            new_order
        )

        total_expanded += new_expanded
        solve.retried += 1

        # Keep better result
        if len(new_paths) > len(paths):

            paths = new_paths
            failed = new_failed
            expansion_info = new_info
            order = new_order

    return (
        paths,
        failed,
        total_expanded,
        expansion_info
    )


# ============================================================
# GRID RENDERING
# ============================================================

def render(time, paths, fire_time):
    """
    Display the building state at a particular time.
    """

    rows = []

    positions = {}

    for agent_id, path in paths.items():

        if time < len(path):

            positions[path[time]] = agent_id

    for r in range(H):

        line = ""

        for c in range(W):

            cell = (r, c)

            # Wall
            if GRID[r][c] == "#":

                line += "#"

            # Agent
            elif cell in positions:

                line += str(
                    positions[cell]
                )

            # Fire
            elif fire_time.get(
                cell,
                INF
            ) <= time:

                line += "*"

            # Exit
            elif cell in EXITS:

                line += "E"

            # Floor
            else:

                line += "."

        rows.append(line)

    return "\n".join(rows)


# ============================================================
# MAIN PROGRAM
# ============================================================

def parse_args():
    """python evacuation.py [scenario] [astar|greedy]  (flags start with --)"""
    global ALGO
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) > 1:
        if pos[1] not in ("astar", "greedy"):
            print("Algorithm must be astar or greedy")
            sys.exit(1)
        ALGO = pos[1]
    return pos[0] if pos else "normal"


def solve_scenario(name, algo, custom_agents=None):
    """Run the planner and return plain JSON-able data (used by --serve)."""
    global ALGO
    ALGO = algo
    fires, spread, extra = SCENARIOS[name]
    agents = custom_agents if custom_agents is not None else BASE_AGENTS + extra
    ft = fire_times(fires, spread)
    paths, failed, total, info = solve(agents, ft)
    return {
        "agents": [list(a) for a in agents],
        "ft": {f"{r},{c}": t for (r, c), t in ft.items()},
        "paths": {str(i): [list(q) for q in pth] for i, pth in paths.items()},
        "failed": failed,
        "info": {str(i): n for i, n in info.items()},
        "total": total,
        "firstFail": solve.first_fail,
        "retried": solve.retried,
        "T": (max(len(x) for x in paths.values()) - 1) if paths else 0,
        "saved": len(paths),
        "src": "Python",
    }


EXPECTED_SAVED = {
    "normal": 4, "exit_blocked": 4, "bottleneck": 10, "trapped": 3,
    "fast_fire": 3, "door_fire": 3, "crowd_fire": 5,
}


def run_tests():
    """python evacuation.py --test : run every scenario with both algorithms."""
    print(f"{'scenario':<14}{'algo':<8}{'saved':<8}{'nodes':<8}{'time':<6}{'failed':<16}result")
    ok = True
    for name, want in EXPECTED_SAVED.items():
        for algo in ("astar", "greedy"):
            r = solve_scenario(name, algo)
            good = r["saved"] == want
            ok &= good
            print(f"{name:<14}{algo:<8}{str(r['saved']) + '/' + str(len(r['agents'])):<8}"
                  f"{r['total']:<8}{r['T']:<6}{str(r['failed']):<16}{'PASS' if good else 'FAIL'}")
    print("\nfailed = ids of people who had NO safe route (fire reached every path before they could exit); [] = everyone saved")
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


def serve(port=8000):
    """Serve evacuation_explorer.html and /api/solve from this planner."""
    import json, os
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    here = os.path.dirname(os.path.abspath(__file__))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/solve":
                q = parse_qs(u.query)
                sc = q.get("scenario", ["normal"])[0]
                al = q.get("algo", ["astar"])[0]
                if sc not in SCENARIOS or al not in ("astar", "greedy"):
                    return self._send(400, b'{"error":"bad input"}', "application/json")
                custom = None
                raw = q.get("agents", [None])[0]
                if raw is not None:
                    try:
                        custom = [tuple(int(v) for v in a.split(",")) for a in raw.split(";") if a]
                        ok = (len(custom) <= 10 and len(set(custom)) == len(custom)
                              and all(len(a) == 2 and 0 <= a[0] < H and 0 <= a[1] < W
                                      and GRID[a[0]][a[1]] not in "#E" for a in custom))
                    except ValueError:
                        ok = False
                    if not ok:
                        return self._send(400, b'{"error":"bad agents"}', "application/json")
                return self._send(200, json.dumps(solve_scenario(sc, al, custom)).encode(), "application/json")
            if u.path in ("/", "/index.html"):
                try:
                    with open(find_html(), "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    pass
            self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    def find_html():
        import glob
        c = glob.glob(os.path.join(here, "evacuation_explorer*.html"))
        if not c:
            raise OSError("missing html")
        return max(c, key=os.path.getmtime)

    try:
        find_html()
    except OSError:
        print("ERROR: evacuation_explorer.html must be in the same folder as evacuation.py")
        return
    for port in range(port, port + 10):
        try:
            server = HTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        print("No free port found")
        return
    url = f"http://127.0.0.1:{port}"
    print("=" * 52)
    print(f"  EVACUATION PLANNER (WEB) RUNNING:  {url}")
    print("  (opening your browser; press Ctrl+C here to stop)")
    print("=" * 52)
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main():

    # --------------------------------------------------------
    # GET SCENARIO
    # --------------------------------------------------------

    scenario = parse_args()

    # Check scenario
    if scenario not in SCENARIOS:

        print(
            "Invalid scenario."
        )

        print(
            "Available scenarios:"
        )

        print(
            " | ".join(SCENARIOS)
        )

        return

    # --------------------------------------------------------
    # LOAD SCENARIO
    # --------------------------------------------------------

    fires, spread, extra_agents = (
        SCENARIOS[scenario]
    )

    agents = (
        BASE_AGENTS
        + extra_agents
    )

    # --------------------------------------------------------
    # FIRE FORECAST
    # --------------------------------------------------------

    fire_time = fire_times(
        fires,
        spread
    )

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    print(
        f"Scenario = {scenario}"
    )

    print(
        f"Algorithm = {ALGO}"
    )

    print(
        f"Agents = {len(agents)}"
    )

    print(
        f"Exits = {len(EXITS)}"
    )

    print()

    # --------------------------------------------------------
    # RUN A*
    # --------------------------------------------------------

    (
        paths,
        failed,
        total_expanded,
        expansion_info
    ) = solve(
        agents,
        fire_time
    )

    # --------------------------------------------------------
    # DISPLAY AGENT RESULTS
    # --------------------------------------------------------

    for agent_id in range(len(agents)):

        if agent_id in paths:

            path = paths[agent_id]

            print(
                f"Agent {agent_id}: "
                f"{agents[agent_id]} -> "
                f"exit {path[-1]} "
                f"in {len(path) - 1} steps "
                f"[expanded "
                f"{expansion_info[agent_id]}]"
            )

        else:

            print(
                f"Agent {agent_id}: "
                f"NO SAFE ROUTE "
                f"(trapped) "
                f"[expanded "
                f"{expansion_info[agent_id]}]"
            )

    # --------------------------------------------------------
    # EVACUATION SUMMARY
    # --------------------------------------------------------

    if paths:

        evacuation_time = (
            max(
                len(path)
                for path in paths.values()
            ) - 1
        )

        print()

        print(
            f"Evacuation complete at "
            f"t={evacuation_time}"
        )

        print(
            f"Saved: "
            f"{len(paths)}/{len(agents)}"
        )

        print(
            f"Total nodes expanded = "
            f"{total_expanded}"
        )

        print()

        # ----------------------------------------------------
        # SNAPSHOTS
        # ----------------------------------------------------

        snapshot_times = sorted(
            set(
                [
                    0,
                    evacuation_time // 3,
                    (2 * evacuation_time) // 3,
                    evacuation_time
                ]
            )
        )

        for time in snapshot_times:

            print(
                f"--- t={time} ---"
            )

            print(
                render(
                    time,
                    paths,
                    fire_time
                )
            )

            print()

    else:

        print()

        print(
            "Nobody could be evacuated."
        )


# ============================================================
# TKINTER USER INTERFACE
# ============================================================

def launch_ui():
    """Launch a graphical interface for the A* evacuation simulator."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    import threading

    class EvacuationUI:
        def __init__(self, root):
            self.root = root
            self.root.title("Intelligent Emergency Evacuation — A* Search")
            self.root.geometry("1180x760")
            self.root.minsize(1000, 680)

            self.paths = {}
            self.fire_time = {}
            self.expansion_info = {}
            self.failed = []
            self.evacuation_time = 0
            self.current_time = 0
            self.running = False

            self.bg = "#0f172a"
            self.panel = "#111827"
            self.panel2 = "#1f2937"
            self.text = "#e5e7eb"
            self.muted = "#94a3b8"
            self.accent = "#38bdf8"
            self.success = "#22c55e"
            self.danger = "#ef4444"
            self.wall = "#334155"
            self.floor = "#e2e8f0"
            self.exit_color = "#22c55e"
            self.agent_colors = [
                "#f59e0b", "#60a5fa", "#a78bfa", "#f472b6",
                "#14b8a6", "#fb7185", "#84cc16", "#c084fc",
                "#f97316", "#2dd4bf"
            ]

            self._build_ui()
            self._update_scenario_description()
            self.root.after(100, self.run_simulation)

        def _build_ui(self):
            self.root.configure(bg=self.bg)

            header = tk.Frame(self.root, bg=self.bg)
            header.pack(fill="x", padx=20, pady=(16, 8))

            tk.Label(
                header,
                text="INTELLIGENT EMERGENCY EVACUATION",
                font=("Segoe UI", 20, "bold"),
                fg="white",
                bg=self.bg
            ).pack(anchor="w")

            tk.Label(
                header,
                text="Multi-agent Space-Time A* • Fire-aware routing • Cooperative reservations",
                font=("Segoe UI", 10),
                fg=self.muted,
                bg=self.bg
            ).pack(anchor="w", pady=(3, 0))

            controls = tk.Frame(self.root, bg=self.panel, highlightthickness=1,
                                highlightbackground="#263244")
            controls.pack(fill="x", padx=20, pady=8)

            tk.Label(
                controls, text="Scenario", font=("Segoe UI", 10, "bold"),
                fg=self.text, bg=self.panel
            ).grid(row=0, column=0, padx=(14, 6), pady=12, sticky="w")

            self.scenario_var = tk.StringVar(value="normal")
            self.scenario_box = ttk.Combobox(
                controls,
                textvariable=self.scenario_var,
                values=list(SCENARIOS),
                state="readonly",
                width=18
            )
            self.scenario_box.grid(row=0, column=1, padx=6, pady=12)
            self.scenario_box.bind("<<ComboboxSelected>>",
                                   lambda e: self._update_scenario_description())

            self.algo_var = tk.StringVar(value="astar")
            ttk.Combobox(
                controls, textvariable=self.algo_var,
                values=["astar", "greedy"], state="readonly", width=8
            ).grid(row=0, column=5, padx=6, pady=12)

            self.run_btn = tk.Button(
                controls, text="▶  Run A*", command=self.run_simulation,
                bg=self.accent, fg="#082f49", activebackground="#7dd3fc",
                font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=7
            )
            self.run_btn.grid(row=0, column=2, padx=8)

            self.play_btn = tk.Button(
                controls, text="▶  Play", command=self.toggle_play,
                bg=self.panel2, fg=self.text, activebackground="#334155",
                font=("Segoe UI", 10, "bold"), relief="flat", padx=14, pady=7
            )
            self.play_btn.grid(row=0, column=3, padx=4)

            self.reset_btn = tk.Button(
                controls, text="↺  Reset", command=self.reset_view,
                bg=self.panel2, fg=self.text, activebackground="#334155",
                font=("Segoe UI", 10, "bold"), relief="flat", padx=14, pady=7
            )
            self.reset_btn.grid(row=0, column=4, padx=4)

            self.desc_var = tk.StringVar()
            tk.Label(
                controls, textvariable=self.desc_var, font=("Segoe UI", 9),
                fg=self.muted, bg=self.panel, anchor="w"
            ).grid(row=1, column=0, columnspan=5, padx=14, pady=(0, 12), sticky="w")

            body = tk.Frame(self.root, bg=self.bg)
            body.pack(fill="both", expand=True, padx=20, pady=(4, 16))

            left = tk.Frame(body, bg=self.panel, highlightthickness=1,
                            highlightbackground="#263244")
            left.pack(side="left", fill="both", expand=True, padx=(0, 8))

            self.canvas = tk.Canvas(
                left, bg="#020617", highlightthickness=0
            )
            self.canvas.pack(fill="both", expand=True, padx=12, pady=12)
            self.canvas.bind("<Configure>", lambda e: self.draw_grid())

            right = tk.Frame(body, bg=self.panel, width=330, highlightthickness=1,
                             highlightbackground="#263244")
            right.pack(side="right", fill="y")
            right.pack_propagate(False)

            tk.Label(
                right, text="SIMULATION STATUS", font=("Segoe UI", 11, "bold"),
                fg="white", bg=self.panel
            ).pack(anchor="w", padx=16, pady=(16, 10))

            self.status_var = tk.StringVar(value="Ready")
            tk.Label(
                right, textvariable=self.status_var, font=("Segoe UI", 9),
                fg=self.muted, bg=self.panel, wraplength=295, justify="left"
            ).pack(anchor="w", padx=16, pady=(0, 12))

            stats = tk.Frame(right, bg=self.panel2)
            stats.pack(fill="x", padx=14, pady=4)

            self.stat_labels = {}
            for i, key in enumerate([
                ("time", "EVACUATION TIME"),
                ("saved", "SAVED"),
                ("expanded", "NODES EXPANDED"),
                ("agents", "AGENTS"),
            ]):
                card = tk.Frame(stats, bg=self.panel2)
                card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=7)
                tk.Label(card, text=key[1], font=("Segoe UI", 7, "bold"),
                         fg=self.muted, bg=self.panel2).pack()
                val = tk.Label(card, text="—", font=("Segoe UI", 15, "bold"),
                               fg="white", bg=self.panel2)
                val.pack(pady=(2, 0))
                self.stat_labels[key[0]] = val

            stats.grid_columnconfigure(0, weight=1)
            stats.grid_columnconfigure(1, weight=1)

            tk.Label(
                right, text="AGENT RESULTS", font=("Segoe UI", 10, "bold"),
                fg="white", bg=self.panel
            ).pack(anchor="w", padx=16, pady=(16, 7))

            table_frame = tk.Frame(right, bg=self.panel)
            table_frame.pack(fill="both", expand=True, padx=14)

            columns = ("agent", "exit", "steps", "expanded", "status")
            self.tree = ttk.Treeview(table_frame, columns=columns,
                                     show="headings", height=12)
            headings = {
                "agent": "Agent", "exit": "Exit", "steps": "Steps",
                "expanded": "Expanded", "status": "Status"
            }
            widths = {"agent": 45, "exit": 65, "steps": 48,
                      "expanded": 65, "status": 72}
            for col in columns:
                self.tree.heading(col, text=headings[col])
                self.tree.column(col, width=widths[col], anchor="center")
            self.tree.pack(side="left", fill="both", expand=True)

            scroll = ttk.Scrollbar(table_frame, orient="vertical",
                                   command=self.tree.yview)
            scroll.pack(side="right", fill="y")
            self.tree.configure(yscrollcommand=scroll.set)

            bottom = tk.Frame(right, bg=self.panel)
            bottom.pack(fill="x", padx=14, pady=12)

            self.time_var = tk.IntVar(value=0)
            self.time_scale = tk.Scale(
                bottom, from_=0, to=1, orient="horizontal",
                variable=self.time_var, command=self.on_time_change,
                bg=self.panel, fg=self.text, troughcolor=self.panel2,
                highlightthickness=0, showvalue=False
            )
            self.time_scale.pack(fill="x")

            self.time_label = tk.Label(
                bottom, text="t = 0", font=("Segoe UI", 9, "bold"),
                fg=self.accent, bg=self.panel
            )
            self.time_label.pack(anchor="e")

        def _update_scenario_description(self):
            descriptions = {
                "normal": "4 agents • fire at (5,4) • fire spreads every 2 time steps",
                "exit_blocked": "4 agents • left exit becomes unsafe • planner redirects agents",
                "bottleneck": "10 agents • single doorway bottleneck • fire spreads every 3 time steps",
                "trapped": "4 agents • upper-left room is sealed • demonstrates NO SAFE ROUTE",
                "fast_fire": "4 agents • fire spreads every step • person 0 is lost",
                "door_fire": "4 agents • fast fire at middle door • person 2 is cut off",
                "crowd_fire": "10 agents • fast fire at middle door • about half are lost"
            }
            self.desc_var.set(descriptions[self.scenario_var.get()])

        def run_simulation(self):
            if self.running:
                return
            self.running = True
            self.play_btn.config(text="▶  Play")
            self.run_btn.config(state="disabled")
            self.status_var.set("Running Space-Time A*...")
            self.root.update_idletasks()

            scenario = self.scenario_var.get()

            algo_choice = self.algo_var.get()

            def worker():
                global ALGO
                ALGO = algo_choice
                try:
                    fires, spread, extra_agents = SCENARIOS[scenario]
                    agents = BASE_AGENTS + extra_agents
                    fire_time = fire_times(fires, spread)
                    paths, failed, total_expanded, expansion_info = solve(
                        agents, fire_time
                    )

                    self.root.after(
                        0, lambda: self._simulation_done(
                            agents, fire_time, paths, failed,
                            total_expanded, expansion_info
                        )
                    )
                except Exception as exc:
                    self.root.after(
                        0, lambda: self._simulation_error(str(exc))
                    )

            threading.Thread(target=worker, daemon=True).start()

        def _simulation_done(self, agents, fire_time, paths, failed,
                             total_expanded, expansion_info):
            self.running = False
            self.run_btn.config(state="normal")
            self.fire_time = fire_time
            self.paths = paths
            self.failed = failed
            self.expansion_info = expansion_info
            self.evacuation_time = (
                max((len(p) for p in paths.values()), default=1) - 1
            )
            self.current_time = 0

            saved = len(paths)
            self.stat_labels["time"].config(text=f"t={self.evacuation_time}")
            self.stat_labels["saved"].config(
                text=f"{saved}/{len(agents)}",
                fg=self.success if saved == len(agents) else self.danger
            )
            self.stat_labels["expanded"].config(text=str(total_expanded))
            self.stat_labels["agents"].config(text=str(len(agents)))

            self.time_scale.config(to=max(1, self.evacuation_time))
            self.time_var.set(0)

            for item in self.tree.get_children():
                self.tree.delete(item)

            for agent_id in range(len(agents)):
                if agent_id in paths:
                    path = paths[agent_id]
                    exit_pos = path[-1]
                    steps = len(path) - 1
                    status = "Reached"
                else:
                    exit_pos = "-"
                    steps = "-"
                    status = "Blocked"

                self.tree.insert(
                    "", "end",
                    values=(
                        f"A{agent_id}",
                        exit_pos,
                        steps,
                        expansion_info.get(agent_id, 0),
                        status
                    )
                )

            if failed:
                self.status_var.set(
                    f"Completed with {len(failed)} blocked agent(s). "
                    f"Use the Trapped scenario to demonstrate NO SAFE ROUTE."
                )
            else:
                self.status_var.set(
                    f"All {len(agents)} agents have a safe plan. "
                    f"Use Play or drag the timeline to inspect movement."
                )

            self.draw_grid()

        def _simulation_error(self, message):
            self.running = False
            self.run_btn.config(state="normal")
            self.status_var.set("Simulation error.")
            messagebox.showerror("Simulation Error", message)

        def draw_grid(self):
            if not self.canvas.winfo_exists():
                return

            self.canvas.delete("all")
            if not self.fire_time:
                return

            rows = H
            cols = W
            cw = max(1, self.canvas.winfo_width() / cols)
            ch = max(1, self.canvas.winfo_height() / rows)
            size = min(cw, ch)
            offset_x = (self.canvas.winfo_width() - size * cols) / 2
            offset_y = (self.canvas.winfo_height() - size * rows) / 2
            t = self.current_time

            positions = {}
            for agent_id, path in self.paths.items():
                if t < len(path):
                    positions[path[t]] = agent_id

            for r in range(rows):
                for c in range(cols):
                    x1 = offset_x + c * size
                    y1 = offset_y + r * size
                    x2 = x1 + size
                    y2 = y1 + size
                    cell = (r, c)

                    if GRID[r][c] == "#":
                        fill = self.wall
                    elif cell in positions:
                        fill = self.agent_colors[
                            positions[cell] % len(self.agent_colors)
                        ]
                    elif self.fire_time.get(cell, INF) <= t:
                        fill = self.danger
                    elif cell in EXITS:
                        fill = self.exit_color
                    elif GRID[r][c] == "D":
                        fill = "#facc15"
                    else:
                        fill = self.floor

                    self.canvas.create_rectangle(
                        x1 + 1, y1 + 1, x2 - 1, y2 - 1,
                        fill=fill, outline="#64748b"
                    )

                    if cell in positions:
                        self.canvas.create_text(
                            (x1+x2)/2, (y1+y2)/2,
                            text=str(positions[cell]),
                            fill="#111827",
                            font=("Segoe UI", max(7, int(size * .42)), "bold")
                        )
                    elif cell in EXITS:
                        self.canvas.create_text(
                            (x1+x2)/2, (y1+y2)/2,
                            text="E", fill="white",
                            font=("Segoe UI", max(7, int(size * .38)), "bold")
                        )
                    elif self.fire_time.get(cell, INF) <= t:
                        self.canvas.create_text(
                            (x1+x2)/2, (y1+y2)/2,
                            text="*", fill="white",
                            font=("Segoe UI", max(7, int(size * .38)), "bold")
                        )

            self.time_label.config(text=f"t = {t}")

        def on_time_change(self, value):
            try:
                self.current_time = int(float(value))
            except ValueError:
                self.current_time = 0
            self.draw_grid()

        def reset_view(self):
            self.running = False
            self.current_time = 0
            self.time_var.set(0)
            self.draw_grid()

        def toggle_play(self):
            if not self.paths:
                return
            self.running = not self.running
            if self.running:
                self.play_btn.config(text="❚❚  Pause")
                self._animate()
            else:
                self.play_btn.config(text="▶  Play")

        def _animate(self):
            if not self.running:
                return
            if self.current_time >= self.evacuation_time:
                self.running = False
                self.play_btn.config(text="▶  Play")
                return

            self.current_time += 1
            self.time_var.set(self.current_time)
            self.draw_grid()
            self.root.after(450, self._animate)

    root = tk.Tk()
    app = EvacuationUI(root)
    root.mainloop()


# ============================================================
# COMMAND-LINE / UI ENTRY POINT
# ============================================================

def launch():
    """
    Usage:
        python evacuation.py
        python evacuation.py normal
        python evacuation.py --ui
    """
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    elif len(sys.argv) == 1 or "--demo" in sys.argv:
        run_tests()
        print()
        serve()
    elif "--serve" in sys.argv:
        serve()
    elif "--ui" in sys.argv or "--gui" in sys.argv:
        launch_ui()
    else:
        main()

# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    launch()