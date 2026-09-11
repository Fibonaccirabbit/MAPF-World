"""Shared MAPF observation encoding and bounded cost-to-go caching."""

from collections import OrderedDict, deque

import numpy as np
from pydantic import BaseModel

# Adapted from MAPF-GPT tokenizer/tokenizer.py and tokenizer/cost2go.cpp,
# and MAPF-World world/Inference.py.


MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
COORDS = list(range(-20, 21)) + [-80, -40, 40]
VOCAB = {value: index for index, value in enumerate(COORDS)}


class ObservationEncoder:
    """Fixed 11x11 cost map + 13 agents x 10 fields + 5 padding tokens.

    History records executed displacements. Call reset between episodes.
    `clip` clamps coordinate values; `buckets` groups out-of-range coordinates.
    """

    def __init__(self, cache_mb=128, coordinate_encoding="clip"):
        if cache_mb <= 0 or coordinate_encoding not in ("clip", "buckets"):
            raise ValueError("Invalid distance cache or coordinate encoding")
        self.cache_bytes = int(cache_mb * 1024**2)
        self.coordinate_encoding = coordinate_encoding
        self.reset()

    def reset(self):
        self.grid = self.previous = self.history = None
        self.cache = OrderedDict()
        self.cached_bytes = 0

    def distance(self, position):
        key = tuple(position)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = np.full(self.grid.shape, -1, dtype=np.int32)
        if self.grid[key] != 0:
            raise ValueError("Agent or goal occupies an obstacle")
        result[key] = 0
        queue = deque([key])
        height, width = result.shape
        while queue:
            x, y = queue.popleft()
            step = result[x, y] + 1
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width:
                    if self.grid[nx, ny] == 0 and result[nx, ny] < 0:
                        result[nx, ny] = step
                        queue.append((nx, ny))
        if result.nbytes <= self.cache_bytes:
            while self.cache and self.cached_bytes + result.nbytes > self.cache_bytes:
                self.cached_bytes -= self.cache.popitem(last=False)[1].nbytes
            self.cache[key] = result
            self.cached_bytes += result.nbytes
        return result

    def coord(self, value):
        value = int(value)
        if self.coordinate_encoding == "clip":
            return VOCAB[max(-20, min(20, value))]
        return VOCAB[
            value if abs(value) <= 20 else (40 if value > 20 else (-40 if value >= -40 else -80))
        ]

    def encode(self, observations):
        if not observations:
            raise ValueError("No observations supplied")
        grid = np.asarray(observations[0]["global_obstacles"], dtype=np.int8)
        positions = np.asarray([obs["global_xy"] for obs in observations], dtype=int)
        goals = np.asarray([obs["global_target_xy"] for obs in observations], dtype=int)
        count = len(positions)
        if self.grid is None:
            self.grid = grid.copy()
            self.history = np.full((count, 5), 44, dtype=np.int8)
        elif not np.array_equal(grid, self.grid) or len(self.previous) != count:
            raise ValueError("Observation map/agent count changed; reset the encoder first")
        if self.previous is not None:
            delta = positions - self.previous
            matches = (delta[:, None, :] == MOVES[None, :, :]).all(axis=-1)
            if not matches.any(axis=1).all():
                raise ValueError("Non-adjacent executed movement in observation history")
            self.history[:, :-1] = self.history[:, 1:]
            self.history[:, -1] = matches.argmax(axis=1) + 45
        self.previous = positions.copy()
        greedy = np.empty(count, dtype=np.int8)
        output = np.full((count, 256), 66, dtype=np.int8)
        for index, (position, goal) in enumerate(zip(positions, goals)):
            distances = self.distance(goal)
            x, y = position
            if x < 5 or y < 5 or x + 5 >= grid.shape[0] or y + 5 >= grid.shape[1]:
                raise ValueError("Expected POGEMA observations with a five-cell border")
            center = distances[x, y]
            patch = distances[x - 5 : x + 6, y - 5 : y + 6]
            relative = patch - center
            values = np.where(
                patch < 0, -80, np.where(relative > 20, 40, np.where(relative < -20, -40, relative))
            )
            output[index, :121] = [VOCAB[int(v)] for v in values.flat]
            bits = 0
            for dx, dy in MOVES[1:]:
                bits = 2 * bits + int(0 <= distances[x + dx, y + dy] < center)
            greedy[index] = 50 + bits
        for index, position in enumerate(positions):
            distances = self.distance(position)
            candidates = np.flatnonzero((np.abs(positions - position).max(axis=1) <= 5))
            candidates = [j for j in candidates if distances[tuple(positions[j])] >= 0]
            # Sort reachable neighbors by geodesic distance.
            ordered = sorted(
                candidates, key=lambda j: (int(distances[tuple(positions[j])]), int(j))
            )[:13]
            for slot, other in enumerate(ordered):
                offset = 121 + slot * 10
                output[index, offset : offset + 4] = [
                    self.coord(v)
                    for v in (*list(positions[other] - position), *list(goals[other] - position))
                ]
                output[index, offset + 4 : offset + 9] = self.history[other]
                output[index, offset + 9] = greedy[other]
        return output


# Adapted from MAPF-GPT tokenizer/parameters.py; see root references.bib.


class InputParameters(BaseModel):
    num_agents: int = 13
    num_previous_actions: int = 5
    agents_radius: int = 5
    cost2go_value_limit: int = 20
    cost2go_radius: int = 5
    context_size: int = 256
    mask_greed_action: bool = False
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    prediction_horizon: int = 1


# Adapted from MAPF-GPT tokenizer/tokenizer.py; see root references.bib.


class Encoder:
    def __init__(self, cfg: InputParameters):
        self.cfg = cfg
        self.coord_range = list(range(-cfg.cost2go_value_limit, cfg.cost2go_value_limit + 1)) + [
            -cfg.cost2go_value_limit * 4,
            -cfg.cost2go_value_limit * 2,
            cfg.cost2go_value_limit * 2,
        ]
        self.actions_range = ["n", "w", "u", "d", "l", "r"]
        self.next_action_range = [format(i, "04b") for i in range(16)]  # 0000 to 1111

        self.vocab = {
            token: idx
            for idx, token in enumerate(
                self.coord_range + self.actions_range + self.next_action_range + ["!"]
            )
        }  # '!' is a trash symbol
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, observation):
        agents_indices = []

        def clamp_value(value, max_abs_value=20):
            return max(-max_abs_value, min(max_abs_value, value))

        for agent in observation["agents"]:
            coord_indices = [
                self.vocab[clamp_value(agent["relative_pos"][0])],
                self.vocab[clamp_value(agent["relative_pos"][1])],
                self.vocab[clamp_value(agent["relative_goal"][0])],
                self.vocab[clamp_value(agent["relative_goal"][1])],
            ]
            actions_indices = [self.vocab[action] for action in agent["previous_actions"]]
            next_action_indices = [self.vocab[agent["next_action"]]]

            agent_obs = coord_indices + actions_indices + next_action_indices
            agents_indices.extend(agent_obs)
        if len(observation["agents"]) < self.cfg.num_agents:
            agents_indices.extend(
                [
                    self.vocab["!"]
                    for _ in range(
                        (self.cfg.num_agents - len(observation["agents"]))
                        * (5 + self.cfg.num_previous_actions)
                    )
                ]
            )
        cost2go_indices = [self.vocab[v] for v in np.array(observation["cost2go"]).flatten()]

        result = (
            cost2go_indices
            + agents_indices
            + [
                self.vocab["!"]
                for _ in range(self.cfg.context_size - len(cost2go_indices) - len(agents_indices))
            ]
        )
        if any(
            [
                self.cfg.mask_actions_history,
                self.cfg.mask_cost2go,
                self.cfg.mask_goal,
                self.cfg.mask_greed_action,
            ]
        ):
            result = self.mask(result)
        return result

    def mask(self, input):
        cost2go_size = (self.cfg.cost2go_radius * 2 + 1) ** 2
        if self.cfg.mask_actions_history:
            for i in range(self.cfg.num_agents):
                input[
                    cost2go_size + i * (5 + self.cfg.num_previous_actions) + 4 : cost2go_size
                    + i * (5 + self.cfg.num_previous_actions)
                    + 4
                    + self.cfg.num_previous_actions
                ] = [self.vocab["!"] for _ in range(self.cfg.num_previous_actions)]
        if self.cfg.mask_cost2go:
            traversable_cell = self.vocab[0]
            blocked_cell = self.vocab[-self.cfg.cost2go_value_limit * 4]
            for i in range(cost2go_size):
                if input[i] != blocked_cell:
                    input[i] = traversable_cell
        if self.cfg.mask_goal:
            for i in range(self.cfg.num_agents):
                input[cost2go_size + i * (5 + self.cfg.num_previous_actions) + 2] = self.vocab["!"]
                input[cost2go_size + i * (5 + self.cfg.num_previous_actions) + 3] = self.vocab["!"]
        if self.cfg.mask_greed_action:
            for i in range(self.cfg.num_agents):
                input[
                    cost2go_size
                    + i * (5 + self.cfg.num_previous_actions)
                    + 4
                    + self.cfg.num_previous_actions
                ] = self.vocab["!"]
        return input

    def decode(self, idx):
        if any(
            [
                self.cfg.mask_actions_history,
                self.cfg.mask_cost2go,
                self.cfg.mask_goal,
                self.cfg.mask_greed_action,
            ]
        ):
            idx = self.mask(idx)
        agents_info_size = 4 + self.cfg.num_previous_actions + 1
        cost2go_size = (self.cfg.cost2go_radius * 2 + 1) ** 2
        agents = []
        for i in range(self.cfg.num_agents):
            agent_indices = idx[
                cost2go_size + i * agents_info_size : cost2go_size + (i + 1) * agents_info_size
            ]

            relative_pos = (
                self.inverse_vocab[agent_indices[0]],
                self.inverse_vocab[agent_indices[1]],
            )
            relative_goal = (
                self.inverse_vocab[agent_indices[2]],
                self.inverse_vocab[agent_indices[3]],
            )
            previous_actions = [self.inverse_vocab[a] for a in agent_indices[4:-1]]
            next_action = self.inverse_vocab[agent_indices[-1]]

            agent = {
                "relative_pos": relative_pos,
                "relative_goal": relative_goal,
                "previous_actions": previous_actions,
                "next_action": next_action,
            }
            agents.append(agent)

        cost2go_indices = idx[:cost2go_size]
        cost2go = [self.inverse_vocab[v] for v in cost2go_indices]
        cost2go_size = self.cfg.cost2go_radius * 2 + 1
        cost2go = np.array(cost2go).reshape(cost2go_size, cost2go_size)
        observation = {"agents": agents, "cost2go": cost2go}

        return observation


# MAPF-GPT cost2go.cpp API with a bounded, lazy distance-matrix cache.


class DistanceMatrices:
    def __init__(self, grid):
        self.encoder = ObservationEncoder(cache_mb=128)
        self.encoder.grid = np.asarray(grid, dtype=np.int8)

    def __getitem__(self, position):
        return self.encoder.distance(tuple(position))


def precompute_cost2go(grid, offset=5):
    return DistanceMatrices(grid)


def get_cost_matrix(grid, x, y):
    return precompute_cost2go(grid)[x, y]


def generate_cost2go_obs(matrix, position, offset=5, limit=20, only_obstacles=False):
    if offset == 0:
        return []
    x, y = position
    matrix = np.asarray(matrix)
    patch = matrix[x - offset : x + offset + 1, y - offset : y + offset + 1]
    if only_obstacles:
        return (patch < 0).astype(int).ravel().tolist()
    relative = patch - matrix[x, y]
    values = np.where(
        patch < 0,
        -limit * 4,
        np.where(relative > limit, limit * 2, np.where(relative < -limit, -limit * 2, relative)),
    )
    return values.ravel().tolist()
