// Adapted from MAPF-World lacam/main.cpp and MAPF-GPT's LaCAM bridge.
// Solver implementation: https://github.com/Kei18/lacam3 (external checkout).
// Copyright (c) 2024 National Institute of Advanced Industrial Science and Technology (AIST).
// Upstream portions are distributed under the MIT license in the root LICENSE.
#include <iostream>
#include <lacam.hpp>

extern "C" {
const char* run_lacam(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec);
}

const char* run_lacam(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec)
{
  std::string map_content(map_content_cstr);
  std::string scene_content(scene_content_cstr);

  const int seed = 0;
  const int verbose = 0;

  const bool flg_no_all = false;
  const bool flg_no_star = false;
  const bool flg_no_swap = false;
  const bool flg_no_multi_thread = true;
  const int pibt_num = 10;
  const bool flg_no_refiner = false;
  const int refiner_num = 4;
  const bool flg_no_scatter = false;
  const int scatter_margin = 10;
  const float random_insert_prob1 = 0.001f;
  const float random_insert_prob2 = 0.01f;
  const bool flg_random_insert_init_node = false;
  const float recursive_rate = 0.2f;
  const int recursive_time_limit = 1000;
  const int checkpoints_duration = 5000;

  // Construct the graph directly from the supplied grid.
  Graph graph;
  std::istringstream map_stream(map_content);
  std::string line, word;
  while (std::getline(map_stream, line)) {
    std::istringstream field(line);
    field >> word;
    if (word == "height") field >> graph.height;
    if (word == "width") field >> graph.width;
    if (word == "map") break;
  }
  if (graph.width <= 0 || graph.height <= 0 || N <= 0) return "ERROR_MAP";
  graph.U.resize(graph.width * graph.height, nullptr);
  for (int y = 0; y < graph.height; ++y) {
    if (!std::getline(map_stream, line) || line.size() < static_cast<size_t>(graph.width))
      return "ERROR_MAP";
    for (int x = 0; x < graph.width; ++x) {
      if (line[x] != '.') continue;
      auto v = new Vertex(graph.V.size(), y * graph.width + x, x, y);
      graph.V.push_back(v);
      graph.U[v->index] = v;
    }
  }
  for (auto v : graph.V) {
    for (auto delta : {std::pair<int, int>{-1, 0}, {1, 0}, {0, 1}, {0, -1}}) {
      const int x = v->x + delta.first, y = v->y + delta.second;
      if (x >= 0 && x < graph.width && y >= 0 && y < graph.height) {
        auto u = graph.U[y * graph.width + x];
        if (u != nullptr) v->neighbor.push_back(u);
      }
    }
  }
  Config starts, goals;
  std::istringstream scene_stream(scene_content);
  std::getline(scene_stream, line);
  while (std::getline(scene_stream, line)) {
    int id, width, height, sx, sy, gx, gy;
    std::string name;
    std::istringstream fields(line);
    if (!(fields >> id >> name >> width >> height >> sx >> sy >> gx >> gy))
      return "ERROR_SCENE";
    if (sx < 0 || sx >= graph.width || gx < 0 || gx >= graph.width ||
        sy < 0 || sy >= graph.height || gy < 0 || gy >= graph.height)
      return "ERROR_SCENE";
    auto s = graph.U[sy * graph.width + sx], g = graph.U[gy * graph.width + gx];
    if (s == nullptr || g == nullptr) return "ERROR_SCENE";
    starts.push_back(s);
    goals.push_back(g);
  }
  auto ins = Instance(&graph, starts, goals, N);
  ins.delete_graph_after_used = false;
  if (!ins.is_valid(1)) return "ERROR_SCENE";

  Planner::FLG_SWAP = !flg_no_swap && !flg_no_all;
  Planner::FLG_STAR = !flg_no_star && !flg_no_all;
  Planner::FLG_MULTI_THREAD = !flg_no_multi_thread && !flg_no_all;
  Planner::PIBT_NUM = flg_no_all ? 1 : pibt_num;
  Planner::FLG_REFINER = !flg_no_refiner && !flg_no_all;
  Planner::REFINER_NUM = refiner_num;
  Planner::FLG_SCATTER = !flg_no_scatter && !flg_no_all;
  Planner::SCATTER_MARGIN = scatter_margin;
  Planner::RANDOM_INSERT_PROB1 = flg_no_all ? 0 : random_insert_prob1;
  Planner::RANDOM_INSERT_PROB2 = flg_no_all ? 0 : random_insert_prob2;
  Planner::FLG_RANDOM_INSERT_INIT_NODE =
      flg_random_insert_init_node && !flg_no_all;
  Planner::RECURSIVE_RATE = flg_no_all ? 0 : recursive_rate;
  Planner::RECURSIVE_TIME_LIMIT = flg_no_all ? 0 : recursive_time_limit;
  Planner::CHECKPOINTS_DURATION = checkpoints_duration;

  const auto deadline = Deadline(time_limit_sec * 1000);
  const auto solution = solve(ins, verbose - 1, &deadline, seed);
  const auto comp_time_ms = deadline.elapsed_ms();

  if (solution.empty()) {
    info(1, verbose, &deadline, "failed to solve");
    return "ERROR_EMPTY";
  }

  if (!is_feasible_solution(ins, solution, verbose)) {
    info(0, verbose, &deadline, "invalid solution");
    return "ERROR_SOLUTION";
  }

  print_stats(verbose, &deadline, ins, solution, comp_time_ms);

  auto get_x = [&](int k) { return k % ins.G->width; };
  auto get_y = [&](int k) { return k / ins.G->width; };

  std::ostringstream result_string;
  for (size_t t = 0; t < solution.size(); ++t) {
    auto C = solution[t];
    for (auto v : C) {
      result_string << get_x(v->index) << "," << get_y(v->index) << "|";
    }
    result_string << "\n";
  }

  static std::string result;
  result = result_string.str();

  return result.c_str();
}
