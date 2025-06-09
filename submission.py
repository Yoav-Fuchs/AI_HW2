import time
from turtledemo.penrose import start

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

class OutOfTime(Exception):
    pass

# TODO: section a : 3

def smart_heuristic(env: WarehouseEnv, robot_id):
    robot = env.robots[robot_id]
    total_value = 0.0

    # Base utility from current credit
    total_value += robot.credit * 10

    # Bonus if at a delivery location
    if robot.package and robot.position == robot.package.destination:
        total_value += manhattan_distance(robot.package.position, robot.package.destination) * 2

    # -------------------------------
    # Case 1: Robot has a package
    # -------------------------------
    if robot.package:
        dest = robot.package.destination
        d_to_dest = manhattan_distance(robot.position, dest)
        total_value -= d_to_dest

    # -------------------------------
    # Case 2: Robot does not have a package
    # -------------------------------
    else:
        enemy = env.get_robot((robot_id + 1) % 2)
        best_package_value = float('inf')
        packages = env.packages
        count = 0
        for package in packages:
            if not package.on_board:
                continue

            d_to_package = manhattan_distance(robot.position, package.position)
            d_package_to_dest = manhattan_distance(package.position, package.destination)
            total_trip = d_to_package + d_package_to_dest
            best_package_value = min(best_package_value, total_trip)

        total_value -= (2 * best_package_value)

    return total_value





class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def rb_minimax(self ,env: WarehouseEnv, agent_id: int, depth: int, current_turn: int, start_time, time_limit):
        if time.time() - start_time + 0.05 >= time_limit:
            raise OutOfTime
        if env.done() or depth == 0:
            return smart_heuristic(env, agent_id)

        # Get the player whose move it is now
        acting_agent = current_turn

        operators = env.get_legal_operators(acting_agent)
        children = [env.clone() for _ in operators]

        for child, op in zip(children, operators):
            child.apply_operator(acting_agent, op)

        next_turn = (current_turn + 1) % 2

        if acting_agent == agent_id:
            value = float('-inf')
            for child in children:
                v = self.rb_minimax(child, agent_id, depth - 1, next_turn, start_time, time_limit)
                value = max(value, v)
            return value
        else:
            value = float('inf')
            for child in children:
                v = self.rb_minimax(child, agent_id, depth - 1, next_turn, start_time, time_limit)
                value = min(value, v)
            return value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        best_op = operators[0] if operators else None  # fallback default
        depth = 1

        try:
            while True:
                current_best_op = None
                best_score = float('-inf')

                for op in operators:
                    if time.time() - start_time >= time_limit - 0.05:
                        raise OutOfTime

                    child = env.clone()
                    child.apply_operator(agent_id, op)
                    next_turn = (agent_id + 1) % 2

                    score = self.rb_minimax(child, agent_id, depth=depth, current_turn=next_turn, start_time=start_time,
                                            time_limit=time_limit)

                    if score > best_score:
                        best_score = score
                        current_best_op = op

                # Only update best_op after fully completing this depth
                best_op = current_best_op
                depth += 1

        except OutOfTime:
            pass

        return best_op


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)