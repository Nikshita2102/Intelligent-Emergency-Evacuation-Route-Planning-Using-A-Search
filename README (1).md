# Intelligent Emergency Evacuation Route Planning (A* and Greedy)

Multi-agent evacuation while fire spreads. Space-time search with cooperative reservations.
Python 3.8+, standard library only.

## Run
    python evacuation.py                              # BEST: runs tests, then opens the web demo
    python evacuation.py [scenario] [astar|greedy]   # text output
    python evacuation.py --test                       # run all test cases (PASS/FAIL table)
    python evacuation.py --ui                         # Tkinter window
    python evacuation.py --serve                      # web explorer at http://127.0.0.1:8000

Scenarios: normal, exit_blocked, bottleneck, trapped, fast_fire, door_fire, crowd_fire

## Files
- evacuation.py: planner, CLI, tests, Tkinter UI, local server (/api/solve)
- evacuation_explorer.html: web UI; uses the Python planner when served, else its built-in JS copy
- Keep both files in the same folder.
