#pragma once

#include <vector>

#include "TemporallyExtendedModel.h"


class SimpleInterface {
public:
  void append(int action, int observation, double reward) {
    this->actions.push_back(action);
    this->observations.push_back(observation);
    this->rewards.push_back(reward);
  };
  void clear() {
    this->actions.clear();
    this->observations.clear();
    this->rewards.clear();
  };

  void fit() { model.optimize(); };

private:
  TemporallyExtendedModel model;

  std::vector<int> actions;
  std::vector<int> observations;
  std::vector<double> rewards;
};
