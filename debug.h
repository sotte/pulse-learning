#ifndef DEBUG_H_
#define DEBUG_H_

#include <iostream>
#include <assert.h>

#ifndef DEBUG_STRING
#define DEBUG_STRING __FILE__ << ": "
#endif

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif
#ifndef FORCE_DEBUG_LEVEL
#define FORCE_DEBUG_LEVEL 0
#endif

#ifdef DEBUG

#include <vector>

namespace { // anonymous namespace for encapsulation
    class DEBUG_INDENTATION {
    public:
        static std::vector<bool> close_indentation;
        DEBUG_INDENTATION() {
            #pragma omp critical
            close_indentation.push_back(false);
        }
        ~DEBUG_INDENTATION() {
            #pragma omp critical
            {
                if(close_indentation.back()) {
                    std::cout << DEBUG_STRING;
                    for(uint indent=0; indent<DEBUG_INDENTATION::close_indentation.size(); ++indent) {
                        if(indent<DEBUG_INDENTATION::close_indentation.size()-1) std::cout << "│   ";
                        else std::cout << "┷   ";
                    }
                    std::cout << std::endl;
                }
                close_indentation.pop_back();
            }
        }
    };
    std::vector<bool> DEBUG_INDENTATION::close_indentation;
} // end anonymous

#define DEBUG_INDENT auto DEBUG_INDENTATION_tmp = DEBUG_INDENTATION();

#define IF_DEBUG(level) if(level<=FORCE_DEBUG_LEVEL || (FORCE_DEBUG_LEVEL==0 && level<=DEBUG_LEVEL))

#define DEBUG_ERROR(message) {                                          \
        std::cerr << "Error(" << __FILE__ << ":" << __LINE__ << "): " << message << std::endl; \
    }

#define DEBUG_WARNING(message) {                                        \
        std::cerr << "Warning(" << __FILE__ << ":" << __LINE__ << "): " << message << std::endl; \
    }

#define DEBUG_OUT(level,message) {                                      \
        IF_DEBUG(level) {                                               \
            std::cout << DEBUG_STRING;                                  \
            for(uint indent=0; indent<DEBUG_INDENTATION::close_indentation.size(); ++indent) { \
                if(indent<DEBUG_INDENTATION::close_indentation.size()-1) std::cout << "│   "; \
                else std::cout << "├   ";                               \
            }                                                           \
            DEBUG_INDENTATION::close_indentation.assign(DEBUG_INDENTATION::close_indentation.size(),true); \
            std::cout << message << std::endl;                          \
        }                                                               \
    }

#define DEBUG_DEAD_LINE {                                       \
        DEBUG_ERROR("This line should never be reached");       \
    }

#define DEBUG_EXPECT(condition) {                       \
        if(!(condition)) {                                              \
            DEBUG_ERROR("Condition '" << #condition << "' is not fulfilled"); \
        }                                                               \
    }

#define DEBUG_EXPECT_APPROX(value_1, value_2) {  \
        if(fabs((value_1)-(value_2))>1e-10) {                           \
            DEBUG_ERROR("Not approximately equal ('" << #value_1 << "' and '" << #value_2 << "')"); \
            DEBUG_ERROR("    " << #value_1 << " = " << value_1);        \
            DEBUG_ERROR("    " << #value_2 << " = " << value_2);        \
            DEBUG_ERROR("    " << "difference = " << (value_1)-(value_2)); \
        }                                                               \
    }

#else // DEBUG

#define DEBUG_INDENT {}

#define IF_DEBUG(level) if(false)

#define DEBUG_ERROR(message) {}

#define DEBUG_WARNING(message) {}

#define DEBUG_OUT(level,message) {}

#define DEBUG_DEAD_LINE {}

#define DEBUG_EXPECT(condition) {}

#define DEBUG_EXPECT_APPROX(value_1, value_2) {}

#endif // DEBUG

#else
static_assert(false, "Including this file multiple times is usually not a good idea. Include 'debug_exclude.h' to exclude.");
#endif /* DEBUG_H_ */
