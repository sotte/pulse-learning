#ifndef TEMPORALLY_EXTENDED_MODEL_H_
#define TEMPORALLY_EXTENDED_MODEL_H_

#include "AbstractTemporallyExtendedModel.h"

class TemporallyExtendedModel: public AbstractTemporallyExtendedModel {
public:
    TemporallyExtendedModel(int action_n, int observation_n);
    virtual ~TemporallyExtendedModel() = default;
    virtual AbstractTemporallyExtendedModel & set_regularization(double) override;
    virtual AbstractTemporallyExtendedModel & set_data(const data_t &) override;
    virtual AbstractTemporallyExtendedModel & set_horizon_extension(int n) override;
    virtual AbstractTemporallyExtendedModel & set_maximum_horizon(int n) override;
    virtual double optimize() override;
    virtual double get_prediction(const data_t & data) override;
    virtual AbstractTemporallyExtendedModel & set_gradient_threshold(double) override;
    virtual AbstractTemporallyExtendedModel & set_parameter_threshold(double) override;
    virtual AbstractTemporallyExtendedModel & set_max_inner_loop_iterations(int) override;
    virtual AbstractTemporallyExtendedModel & set_max_outer_loop_iterations(int) override;
    virtual AbstractTemporallyExtendedModel & set_likelihood_threshold(double) override;
};

#endif /* TEMPORALLY_EXTENDED_MODEL_H_ */
