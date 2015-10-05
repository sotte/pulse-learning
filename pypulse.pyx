cdef extern from "SimpleInterface.cpp":
    cdef cppclass SimpleInterface:
        SimpleInterface() except +
        void clear()


cdef class PyPulse:
    cdef SimpleInterface *thisptr

    def __cinit__(self):
        self.thisptr = new SimpleInterface()

    def __dealloc__(self):
        del self.thisptr

    def clear(self):
        self.thisptr.clear()
