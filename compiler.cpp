#include "compiler.h"
#include <iostream>
#include <sstream>

// Lexer implementation
Lexer::Lexer(const std::string& source) : source(source), position(0) {
    readNextToken();
}

void Lexer::skipWhitespace() {
    while (position < source.size() && std::isspace(static_cast<unsigned char>(source[position]))) {
        position++;
    }
}

bool Lexer::isIdentStart(char c) const {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool Lexer::isIdentPart(char c) const {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

void Lexer::readNextToken() {
    skipWhitespace();
    
    if (position >= source.size()) {
        currentToken = {TokenType::END_OF_SOURCE, "", 0};
        return;
    }
    
    char c = source[position];
    
    // Handle single-character tokens
    switch (c) {
        case '+': position++; currentToken = {TokenType::PLUS, "+", 0}; return;
        case '-': position++; currentToken = {TokenType::MINUS, "-", 0}; return;
        case '*': position++; currentToken = {TokenType::STAR, "*", 0}; return;
        case '/': position++; currentToken = {TokenType::SLASH, "/", 0}; return;
        case '(': position++; currentToken = {TokenType::LPAREN, "(", 0}; return;
        case ')': position++; currentToken = {TokenType::RPAREN, ")", 0}; return;
        case ';': position++; currentToken = {TokenType::SEMICOLON, ";", 0}; return;
    }
    
    // Handle numbers
    if (std::isdigit(static_cast<unsigned char>(c))) {
        size_t start = position;
        while (position < source.size() && std::isdigit(static_cast<unsigned char>(source[position]))) {
            position++;
        }
        std::string number = source.substr(start, position - start);
        int32_t value = std::stoi(number);
        // std::cout << "Lexer: Found number " << number << ", value=" << value << std::endl;
        currentToken = {TokenType::NUMBER, number, value};
        return;
    }
    
    // Handle identifiers and keywords
    if (isIdentStart(c)) {
        size_t start = position;
        position++;
        while (position < source.size() && isIdentPart(source[position])) {
            position++;
        }
        std::string identifier = source.substr(start, position - start);
        
        // Check if it's a keyword
        std::string lowercase = identifier;
        std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);
        
        if (lowercase == "let") {
            currentToken = {TokenType::LET, identifier, 0};
        } else if (lowercase == "print") {
            currentToken = {TokenType::PRINT, identifier, 0};
        } else if (lowercase == "while") {
            currentToken = {TokenType::WHILE, identifier, 0};
        } else if (lowercase == "do") {
            currentToken = {TokenType::DO, identifier, 0};
        } else if (lowercase == "end") {
            currentToken = {TokenType::END_KEYWORD, identifier, 0};
        } else if (lowercase == "if") {
            currentToken = {TokenType::IF, identifier, 0};
        } else if (lowercase == "then") {
            currentToken = {TokenType::THEN, identifier, 0};
        } else {
            currentToken = {TokenType::IDENT, identifier, 0};
        }
        return;
    }
    
    // If we get here, we encountered an unexpected character
    std::stringstream ss;
    ss << "Unexpected character: " << c;
    throw std::runtime_error(ss.str());
}

const Token& Lexer::peek() const {
    return currentToken;
}

const Token& Lexer::next() {
    Token& result = currentToken;
    readNextToken();
    return result;
}

// Compiler implementation
Compiler::Compiler() : lexer(""), nextRegister(0) {}

std::vector<Instruction> Compiler::compile(const std::string& source) {
    // std::cout << "Compiling source: " << source << std::endl;
    lexer = Lexer(source);
    variables.clear();
    instructions.clear();
    nextRegister = 0;
    
    parseProgram();
    emit(OpCode::HALT);
    
    return instructions;
}

void Compiler::emit(OpCode opcode, int32_t operand) {
    // std::cout << "Emitting instruction: opcode=" << static_cast<int>(opcode) << ", operand=" << operand << std::endl;
    instructions.push_back({opcode, operand});
    // std::cout << "  Added instruction: opcode=" << static_cast<int>(instructions.back().opcode) 
    //          << ", operand=" << instructions.back().operand << std::endl;
}

int32_t Compiler::getRegisterForVariable(const std::string& name) {
    auto it = variables.find(name);
    if (it != variables.end()) {
        return it->second;
    }
    
    // Allocate a new register for this variable
    int32_t reg = nextRegister++;
    variables[name] = reg;
    return reg;
}

void Compiler::parseProgram() {
    while (lexer.peek().type != TokenType::END_OF_SOURCE) {
        parseStatement();
    }
}

void Compiler::parseStatement() {
    switch (lexer.peek().type) {
        case TokenType::LET:
            parseAssignment();
            break;
        case TokenType::PRINT:
            parsePrintStatement();
            break;
        case TokenType::WHILE:
            parseWhileStatement();
            break;
        case TokenType::IF:
            parseIfStatement();
            break;
        default:
            throw std::runtime_error("Expected statement, got: " + lexer.peek().text);
    }
}

void Compiler::parseAssignment() {
    lexer.next(); // Consume 'let'
    
    if (lexer.peek().type != TokenType::IDENT) {
        throw std::runtime_error("Expected identifier after 'let'");
    }
    
    // Make a copy of the variable name before calling next()
    std::string varName = lexer.peek().text;
    // Now advance the lexer
    lexer.next();
    
    // Parse the expression and leave its value on the stack
    parseExpression();
    
    // Store the value in the variable's register
    int32_t reg = getRegisterForVariable(varName);
    emit(OpCode::STORE, reg);
    
    expect(TokenType::SEMICOLON);
}

void Compiler::parsePrintStatement() {
    lexer.next(); // Consume 'print'
    
    // Parse the expression and leave its value on the stack
    parseExpression();
    
    // Print the value
    emit(OpCode::PRINT);
    
    expect(TokenType::SEMICOLON);
}

void Compiler::parseWhileStatement() {
    lexer.next(); // Consume 'while'
    
    // Remember the position of the loop condition
    size_t conditionPos = instructions.size();
    
    // Parse the condition expression
    parseExpression();
    
    // For the while loop, we need to check if the condition is zero (false)
    // We'll push 0 and compare, which will set the flag to 0 if condition == 0
    emit(OpCode::PUSH, 0); // Use emit instead of direct push
    emit(OpCode::CMP);
    
    // Jump to the end if condition is false (zero)
    emit(OpCode::JZ, 0); // Placeholder, will be patched
    size_t jumpToEndPos = instructions.size() - 1;
    
    expect(TokenType::DO);
    
    // Parse the loop body
    while (lexer.peek().type != TokenType::END_KEYWORD) {
        parseStatement();
    }
    
    lexer.next(); // Consume 'end'
    
    // Jump back to the condition
    emit(OpCode::JMP, static_cast<int32_t>(conditionPos));
    
    // Patch the jump-to-end instruction
    instructions[jumpToEndPos].operand = static_cast<int32_t>(instructions.size());
    
    expect(TokenType::SEMICOLON);
}

void Compiler::parseIfStatement() {
    lexer.next(); // Consume 'if'
    
    // Parse the condition expression
    parseExpression();
    
    // Compare with zero (non-zero is true)
    emit(OpCode::PUSH, 0); // Use emit instead of direct push
    emit(OpCode::CMP);
    
    // Jump to the end if condition is false (zero)
    emit(OpCode::JZ, 0); // Placeholder, will be patched
    size_t jumpToEndPos = instructions.size() - 1;
    
    expect(TokenType::THEN);
    
    // Parse the if body
    while (lexer.peek().type != TokenType::END_KEYWORD) {
        parseStatement();
    }
    
    lexer.next(); // Consume 'end'
    
    // Patch the jump-to-end instruction
    instructions[jumpToEndPos].operand = static_cast<int32_t>(instructions.size());
    
    expect(TokenType::SEMICOLON);
}

void Compiler::parseExpression() {
    parseTerm();
    
    while (lexer.peek().type == TokenType::PLUS || lexer.peek().type == TokenType::MINUS) {
        TokenType op = lexer.next().type;
        parseTerm();
        
        if (op == TokenType::PLUS) {
            emit(OpCode::ADD);
        } else {
            emit(OpCode::SUB);
        }
    }
}

void Compiler::parseTerm() {
    parseFactor();
    
    while (lexer.peek().type == TokenType::STAR || lexer.peek().type == TokenType::SLASH) {
        TokenType op = lexer.next().type;
        parseFactor();
        
        if (op == TokenType::STAR) {
            emit(OpCode::MUL);
        } else {
            emit(OpCode::DIV);
        }
    }
}

void Compiler::parseFactor() {
    switch (lexer.peek().type) {
        case TokenType::NUMBER: {
            // Make a copy of the token before calling next()
            Token numToken = lexer.peek();
            // Now advance the lexer
            lexer.next();
            
            // Use the token's value directly
            int32_t value = numToken.value;
            
            // Use emit function to push instruction
            emit(OpCode::PUSH, value);
            break;
        }
        case TokenType::IDENT: {
            // Make a copy of the token before calling next()
            std::string varName = lexer.peek().text;
            // Now advance the lexer
            lexer.next();
            
            int32_t reg = getRegisterForVariable(varName);
            emit(OpCode::LOAD, reg);
            break;
        }
        case TokenType::LPAREN: {
            lexer.next(); // Consume '('
            parseExpression();
            expect(TokenType::RPAREN);
            break;
        }
        default:
            throw std::runtime_error("Expected factor, got: " + lexer.peek().text);
    }
}

void Compiler::expect(TokenType type) {
    if (lexer.peek().type != type) {
        throw std::runtime_error("Expected token type " + std::to_string(static_cast<int>(type)) + 
                                ", got: " + lexer.peek().text);
    }
    lexer.next();
} 