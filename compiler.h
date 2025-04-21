#ifndef COMPILER_H
#define COMPILER_H

#include "vm.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <stdexcept>
#include <algorithm>

// Token types for our language
enum class TokenType {
    END_OF_SOURCE,  // End of source (renamed from END)
    IDENT,          // Identifier
    NUMBER,         // Number literal
    PLUS,           // +
    MINUS,          // -
    STAR,           // *
    SLASH,          // /
    LPAREN,         // (
    RPAREN,         // )
    SEMICOLON,      // ;
    
    // Keywords
    LET,            // let
    PRINT,          // print
    WHILE,          // while
    DO,             // do
    END_KEYWORD,    // end (renamed from END)
    IF,             // if
    THEN            // then
};

// Token structure
struct Token {
    TokenType type;
    std::string text;  // Raw lexeme
    int32_t value;     // Numeric value (if applicable)
};

// Lexer class for tokenizing the source code
class Lexer {
private:
    std::string source;
    size_t position;
    Token currentToken;
    
    // Helper methods
    void skipWhitespace();
    void readNextToken();
    bool isIdentStart(char c) const;
    bool isIdentPart(char c) const;
    
public:
    explicit Lexer(const std::string& source);
    
    // Get the current token without advancing
    const Token& peek() const;
    
    // Get the current token and advance to the next one
    const Token& next();
};

// Compiler class for translating source code to VM instructions
class Compiler {
private:
    Lexer lexer;
    std::unordered_map<std::string, int32_t> variables;  // Variable name to register mapping
    int32_t nextRegister;
    std::vector<Instruction> instructions;
    
    // Helper methods for code generation
    void emit(OpCode opcode, int32_t operand = 0);
    int32_t getRegisterForVariable(const std::string& name);
    
    // Parsing methods (recursive descent)
    void parseProgram();
    void parseStatement();
    void parseAssignment();
    void parsePrintStatement();
    void parseWhileStatement();
    void parseIfStatement();
    void parseExpression();
    void parseTerm();
    void parseFactor();
    
    // Helper for expecting a specific token type
    void expect(TokenType type);
    
public:
    Compiler();
    
    // Compile source code to VM instructions
    std::vector<Instruction> compile(const std::string& source);
};

#endif // COMPILER_H 