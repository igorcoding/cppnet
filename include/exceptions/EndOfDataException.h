#ifndef CPPNET_EXCEPTIONS_ENDOFDATAEXCEPTION_H
#define CPPNET_EXCEPTIONS_ENDOFDATAEXCEPTION_H

#include "CppNetException.h"

class EndOfDataException : public CppNetException {
public:
    EndOfDataException()
        : CppNetException("End of data")
    { }

    EndOfDataException(const char* message)
        : CppNetException(message)
    { }

    EndOfDataException(const std::string& message)
        : CppNetException(message)
    { }
};

#endif //CPPNET_EXCEPTIONS_ENDOFDATAEXCEPTION_H
