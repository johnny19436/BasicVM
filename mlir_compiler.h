#ifndef MLIR_COMPILER_H
#define MLIR_COMPILER_H

#include "vm.h"
#include "basic_vm_dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

// Token type for our simple language parser
enum class TokenType {
    IDENTIFIER,
    NUMBER,
    KEYWORD,
    OPERATOR,
    END_OF_FILE
};

// Helper function to convert TokenType to string (for error messages)
static const char* tokenTypeToString(TokenType type);

// Token structure
struct Token {
    TokenType type;
    std::string value;
    size_t line;
    size_t column;
};

// MLIR Compiler class for the Basic VM
class MLIRCompiler {
private:
    // MLIR context for building and manipulating operations
    std::unique_ptr<mlir::MLIRContext> context;
    
    // MLIR builder for creating IR
    std::unique_ptr<mlir::OpBuilder> builder;
    
    // The module operation containing all compiled code
    mlir::OwningOpRef<mlir::ModuleOp> module;
    
    // Token stream for parsing
    std::vector<Token> tokens;
    size_t currentTokenIndex;
    
    // Variable mapping for name resolution
    std::unordered_map<std::string, int32_t> variableMap;
    int32_t nextRegister;
    
    // Block mapping for control flow
    std::unordered_map<std::string, mlir::Block*> namedBlocks;
    
    // Create the basic MLIR module structure
    void createModule();
    
    // Parse source code into MLIR
    void parseSource(const std::string& source);
    
    // Lower MLIR dialect to VM instructions
    std::vector<Instruction> lowerToVMInstructions();
    
    // Optimize the VM instruction stream
    void optimizeVMInstructions(std::vector<Instruction>& program);
    
    // Helper method for optimization passes
    void runOptimizationPasses();
    
    // Tokenizer and parsing helpers
    void tokenizeSource(const std::string& source);
    Token& currentToken();
    Token consumeToken();
    bool matchToken(TokenType type, const std::string& value = "");
    void expectToken(TokenType type, const std::string& value = "");
    
    // Recursive descent parsing methods
    void parseStatement();
    void parseVarDeclaration();
    void parsePrintStatement();
    void parseWhileLoop();
    void parseIfStatement();
    void parseExpression();
    void parseTerm();
    void parseFactor();
    
    // Direct code generation methods (simplified approach)
    void parseSimpleExpression(std::vector<Instruction>& program);
    
    // Generate MLIR operations
    void emitPushConstant(int32_t value);
    void emitLoadVariable(const std::string& name);
    void emitStoreVariable(const std::string& name);
    void emitBinaryOp(const std::string& op);
    void emitPrint();
    void emitJump(mlir::Block* target);
    void emitConditionalJump(mlir::Block* target, bool jumpIfZero);
    void emitCompare();
    
    // Register allocation for variables
    int32_t getRegisterForVariable(const std::string& name);

public:
    MLIRCompiler();
    ~MLIRCompiler();
    
    // Main compilation entry point - returns VM instructions
    std::vector<Instruction> compile(const std::string& source);
    
    // Dump MLIR for debugging
    void dumpMLIR() const;
    
    // Getter for variable mapping (for testing/debugging)
    const std::unordered_map<std::string, int32_t>& getVariableMap() const {
        return variableMap;
    }
};

#endif // MLIR_COMPILER_H 