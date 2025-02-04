import gin

from solvers import solver_sokoban
from solvers import bfs_solver_int
from custom.run_only import bfs_solver_rubik


def configure_solver(goal_generator_class):
    return gin.external_configurable(
        goal_generator_class, module='solvers'
    )


BestFSSolverSokoban = configure_solver(solver_sokoban.BestFSSolverSokoban)
BestFSSolverINT = configure_solver(bfs_solver_int.BestFSSolverINT)
BestFSSolverRubik = configure_solver(bfs_solver_rubik.BestFSSolverRubik)
