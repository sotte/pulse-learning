#include <vector>
#include "TemporallyExtendedModel.h"


class SimpleInterface {
public:
  void append(int action, int observation, double reward)
  {
    this->_actions.push_back(action);
    this->_observations.push_back(observation);
    this->_rewards.push_back(reward);
  };

  void clear()
  {
    this->_actions.clear();
    this->_observations.clear();
    this->_rewards.clear();
  };

  void fit()
  {
    TemporallyExtendedModel::data_t data;
    for (size_t i = 0; i < _actions.size(); i++) {
      TemporallyExtendedModel::DataPoint data_point(
          _actions.at(i), _observations.at(i), _rewards.at(i));
      data.push_back(data_point);
    }
    _model.set_data(data);
    _model.optimize();
  };

private:
  TemporallyExtendedModel _model;

  std::vector<int> _actions;
  std::vector<int> _observations;
  std::vector<double> _rewards;
};
