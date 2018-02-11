#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as numpy
from pprint import pprint
import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MultiRegression:

    def __init__(self):
        self.__y_list = []
        self.__x1_list = []
        self.__x2_list = []

        # independent variables, one dimension array
        self.__dependent_list= numpy.array([])

        # independent variables, two dimension array
        self.__explanation_vars_matrix = numpy.array([])

        # partial regression coefficient, tow dimension array
        self.__partial_regression_coefficient_list_ = numpy.array([])

        # multiple correlation coefficient
        self.__multiple_correlation_coefficient = 0.0

    def get_teacher_data(self):
        y = [45, 38, 41, 34, 59, 47, 35, 43, 54, 52]
        for var in y:
            self.__y_list.append(var)

        x1 = [11, 34, 14, 3, 51, 93, 78, 1, 6, 9]
        for var in x1:
            self.__x1_list.append(var)

        x2 = [221, 324, 614, 53, 511, 953, 738, 14, 61, 99]
        for var in x2:
            self.__x2_list.append(var)

    def learn(self):
        # set values
        explanation_vars_matrix =  [self.__x1_list, self.__x2_list]
        self.__set_dependent_vars(self.__y_list)
        self.__set_explanation_vars(explanation_vars_matrix)

        # calculate partial regression coefficient
        self.__calc_partial_regression_coefficient()

        # print the outage
        print(("partial regression coefficient: %s") % (self.__partial_regression_coefficient_list))

    def __set_dependent_vars(self, dependent_vars):
        self.__dependent_list = numpy.array(dependent_vars)

    def __set_explanation_vars(self, explanation_vars_matrix):
        # copy independent vars matrix
        matrix = explanation_vars_matrix

        # get deta length
        length = len(explanation_vars_matrix[0])

        # create [1.0, 1.0, ... , 1.0] list
        tmp_list = numpy.ones((1, length)).tolist()

        # insert [1.0, 1.0, ... , 1.0] to matrix
        matrix.extend(tmp_list)
        self.__explanation_vars_matrix = numpy.array(matrix)

    def __calc_partial_regression_coefficient(self):
        matrix = self.__explanation_vars_matrix
        vector = self.__dependent_list
        temp_matrixA = numpy.dot(matrix, matrix.T)
        temp_matrixB = numpy.dot(matrix, vector)
        self.__partial_regression_coefficient_list = numpy.linalg.solve(temp_matrixA, temp_matrixB)

if __name__ == '__main__':
    pprint('start learning')
    pprint(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    obj = MultiRegression()
    obj.get_teacher_data()
    obj.learn()
    pprint(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    print("end learning")
