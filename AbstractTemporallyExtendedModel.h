#ifndef ABSTRACT_TEMPORALLY_EXTENDED_MODEL_H_
#define ABSTRACT_TEMPORALLY_EXTENDED_MODEL_H_

#include <vector>
#include <tuple>

class AbstractTemporallyExtendedModel {
public:
    typedef int action_t;
    typedef int observation_t;
    typedef double reward_t;
    typedef std::tuple<action_t,observation_t,reward_t> data_point_t;
    typedef std::vector<data_point_t> data_t;

    const int action_n;
    const int observation_n;

    /**
     * The constructor takes the number of actions and observations. Actions and
     * observations have to be greater than or equal to zero and smaller than \p
     * action_n and observation_n, respectively. */
    AbstractTemporallyExtendedModel(int action_n, int observation_n):
        action_n(action_n), observation_n(observation_n)
    {}
    virtual ~AbstractTemporallyExtendedModel() = default;
    /**
     * Set the strength of the L1-regularization. */
    virtual AbstractTemporallyExtendedModel & set_regularization(double) = 0;
    /**
     * Set the data to be used for learning the model. */
    virtual AbstractTemporallyExtendedModel & set_data(const data_t &) = 0;
    /**
     * Set the horizon extension. This is the number of time steps the model is
     * extended into the past with each iteration. */
    virtual AbstractTemporallyExtendedModel & set_horizon_extension(int n) = 0;
    /**
     * Set the maximum extension of the model. Data lying more than \p n steps
     * in the past will not be taken into accout for learning the model or
     * making predictions. */
    virtual AbstractTemporallyExtendedModel & set_maximum_horizon(int n) = 0;
    /**
     * Optimize the model. */
    virtual double optimize() = 0;
    /**
     * Get a prediction. The data are assumed to represent the past except for
     * the last data point. The function return a probability for the
     * observation and reward from the last data point given the action from the
     * last data point and the history from the rest of the data. */
    virtual double get_prediction(const data_t & data) = 0;
    /**
     * If in the inner loop the objective gradient is below this value
     * optimization is stopped. */
    virtual AbstractTemporallyExtendedModel & set_gradient_threshold(double) = 0;
    /**
     * If in the inner loop the largest change of the parameters is below this
     * value optimization is stopped. */
    virtual AbstractTemporallyExtendedModel & set_parameter_threshold(double) = 0;
    /**
     * If the number of iterations in the inner loop reaches this value
     * optimization is stopped. */
    virtual AbstractTemporallyExtendedModel & set_max_inner_loop_iterations(int) = 0;
    /**
     * If the number of iterations in the outer loop reaches this value
     * optimization is stopped. */
    virtual AbstractTemporallyExtendedModel & set_max_outer_loop_iterations(int) = 0;
    /**
     * If in the outer loop the likelihood changes less than this value
     * optimization is stopped. */
    virtual AbstractTemporallyExtendedModel & set_likelihood_threshold(double) = 0;
};

#endif /* ABSTRACT_TEMPORALLY_EXTENDED_MODEL_H_ */
