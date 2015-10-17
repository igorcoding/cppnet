#ifndef CPPNET_EXCEPTIONS_CPPNETEXCEPTION_H
#define CPPNET_EXCEPTIONS_CPPNETEXCEPTION_H

#include <exception>
#include <string>

class CppNetException : public std::exception {
public:
    CppNetException(const char* message)
        : _message(message)
    { }

    CppNetException(const std::string& message)
        : _message(message)
    { }

    virtual const char* what() const noexcept {
        return _message.c_str();
    }

private:
    std::string _message;
};

#endif //CPPNET_EXCEPTIONS_CPPNETEXCEPTION_H
