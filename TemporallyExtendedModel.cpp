#include "TemporallyExtendedModel.h"

#include <iostream>

using std::cout;
using std::endl;

TemporallyExtendedModel::TemporallyExtendedModel(int action_n, int observation_n):
    AbstractTemporallyExtendedModel(action_n,observation_n)
{}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_regularization(double) {
    cout << "set_regularization()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_data(const data_t &) {
    cout << "set_data()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_horizon_extension(int) {
    cout << "set_horizon_extension()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_maximum_horizon(int) {
    cout << "set_maximum_horizon()" << endl;
}

double TemporallyExtendedModel::optimize() {
    cout << "optimize()" << endl;
}

double TemporallyExtendedModel::get_prediction(const data_t &) {
    cout << "get_prediction()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_gradient_threshold(double) {
    cout << "set_gradient_threshold()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_parameter_threshold(double) {
    cout << "set_parameter_threshold()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_max_inner_loop_iterations(int) {
    cout << "set_max_inner_loop_iterations()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_max_outer_loop_iterations(int) {
    cout << "set_max_outer_loop_iterations()" << endl;
}

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_likelihood_threshold(double) {
    cout << "set_likelihood_threshold()" << endl;
}
