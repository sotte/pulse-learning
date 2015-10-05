# from libcpp.vector cimport vector

cdef extern from "SimpleInterface.cpp":
    cdef cppclass SimpleInterface:
        SimpleInterface() except +
        void append(int action, int observation, double reward)
        void clear()
        void fit()


cdef class PyPulse:
    cdef SimpleInterface *thisptr

    def __cinit__(self):
        self.thisptr = new SimpleInterface()

    def __dealloc__(self):
        del self.thisptr

    def append(self, action, observation, reward):
        self.thisptr.append(action, observation, reward)

    def clear(self):
        self.thisptr.clear()

    def fit(self):
        print('Fitting...')
        self.thisptr.fit()
        print('Done')
