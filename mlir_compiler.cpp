/*
 * MLIR Compiler for Basic VM
 * -------------------------
 * 
 * This compiler takes a simple high-level language and compiles it to VM instructions
 * for the Basic VM. Originally intended to use MLIR as an intermediate representation,
 * but due to complexity and integration issues, we've simplified to use a direct 
 * token-based parsing and code generation approach.
 * 
 * Supported language features:
 * - Variable declarations: let x 5;
 * - Variable access: x
 * - Binary operations: x + y, x - y, x * y, x / y
 * - Print statements: print x;
 * - Conditional statements: if expr then ... end;
 * - While loops: while expr do ... end;
 * 
 * The compiler works by:
 * 1. Tokenizing the source code
 * 2. Parsing the tokens and generating VM instructions directly
 * 3. Managing variable mappings to VM registers
 * 4. Handling control flow with jumps
 * 5. Applying various optimizations to the generated code:
 *    - Constant folding (computing expressions with known values at compile time)
 *    - Dead code elimination (removing unreachable code)
 *    - Peephole optimizations (removing redundant instruction sequences)
 *    - Jump optimizations (simplifying control flow)
 * 
 * This is a simpler approach than the full MLIR integration that was originally
 * planned, but it allows us to demonstrate the language concepts without the
 * complexity of the MLIR infrastructure.
 */

#include "mlir_compiler.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "basic_vm_dialect.h" 
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <unordered_set>

// No "using namespace" statements to keep namespace membership explicit

// Helper to convert TokenType to string (for error messages)
const char* tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::IDENTIFIER: return "identifier";
        case TokenType::NUMBER: return "number";
        case TokenType::KEYWORD: return "keyword";
        case TokenType::OPERATOR: return "operator";
        case TokenType::END_OF_FILE: return "end of file";
        default: return "unknown token";
    }
}

MLIRCompiler::MLIRCompiler() {
    // Create MLIR context
    context = std::make_unique<mlir::MLIRContext>();

    // Register all required dialects
    mlir::DialectRegistry registry;
    registry.insert<
        mlir::func::FuncDialect,
        mlir::arith::ArithDialect,
        mlir::cf::ControlFlowDialect,
        mlir::memref::MemRefDialect, 
        mlir::basicvm::BasicVMDialect
    >();
    context->appendDialectRegistry(registry);

    // Force-load dialects so builder->create<…> works without crash
    context->getOrLoadDialect<mlir::func::FuncDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context->getOrLoadDialect<mlir::memref::MemRefDialect>();
    context->getOrLoadDialect<mlir::basicvm::BasicVMDialect>();

    // Initialize builder now that dialects are available
    builder = std::make_unique<mlir::OpBuilder>(context.get());
}


MLIRCompiler::~MLIRCompiler() = default;

void MLIRCompiler::createModule() {
    // Create an empty module
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    
    // Set the insertion point to the module body
    builder->setInsertionPointToEnd(module->getBody());
}

void MLIRCompiler::parseSource(const std::string &source) {
  // Reset our token stream
  currentTokenIndex = 0;

  // Create a new FuncOp<"main"> inside the module
  auto funcType = builder->getFunctionType(
      llvm::ArrayRef<mlir::Type>{},     // no inputs
      llvm::ArrayRef<mlir::Type>{});    // no results
  auto funcOp = builder->create<mlir::func::FuncOp>(
      builder->getUnknownLoc(), "main", funcType);

  // IMPORTANT: insert it into the module's region
  module->push_back(funcOp);

  // Create exactly one entry block and position builder there
  auto *entryBlock = builder->createBlock(&funcOp.getBody());
  builder->setInsertionPointToStart(entryBlock);

  // Parse *all* of your high‑level statements right into that one block
  while (currentToken().type != TokenType::END_OF_FILE) {
    parseStatement();
  }

  // Now rewind the insertion point back to the *end* of that same entry block
  builder->setInsertionPointToEnd(entryBlock);

  // Emit the VM-level terminator first (so VM stops)
  builder->create<mlir::basicvm::HaltOp>(builder->getUnknownLoc());

  // Then emit the func.func terminator
  builder->create<mlir::func::ReturnOp>(builder->getUnknownLoc());

  // Successfully parsed
  llvm::errs() << "✔ parseSource done\n";
}


void MLIRCompiler::runOptimizationPasses() {
    // Create a pass manager
    mlir::PassManager pm(context.get());
    
    // Add optimization passes
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    
    // Run the passes
    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Failed to run optimization passes\n";
    }
}

std::vector<Instruction> MLIRCompiler::lowerToVMInstructions() {
    std::vector<Instruction> instructions;
    
    try {
        // Build a mapping of block pointers to instruction indices
        std::unordered_map<mlir::Block*, uint32_t> blockToInstructionIndex;
        std::unordered_map<mlir::Operation*, uint32_t> opToInstructionIndex;
        
        llvm::errs() << "Starting first pass...\n";
        
        // First pass: collect all operations and assign indices
        uint32_t currentIndex = 0;
        int opCount = 0;
        
        module->walk([&](mlir::Operation *op) {
            // Store the index of this operation
            opToInstructionIndex[op] = currentIndex;
            
            // If this is the first operation in a block, record the block's starting index
            if (!op->getBlock()->empty() && &op->getBlock()->front() == op) {
                blockToInstructionIndex[op->getBlock()] = currentIndex;
            }
            
            // Most operations generate a single instruction
            currentIndex++;
            opCount++;
            
            // Some operations generate multiple instructions
            if (mlir::dyn_cast<mlir::basicvm::LoadOp>(op) || 
                mlir::dyn_cast<mlir::basicvm::StoreOp>(op) ||
                mlir::dyn_cast<mlir::basicvm::CompareOp>(op)) {
                // These ops might generate extra instructions
                currentIndex++;
            }
            
            // Limit the number of instructions for safety
            if (opCount > 1000) {
                throw std::runtime_error("Too many operations in module");
            }
        });
        
        llvm::errs() << "First pass complete, counted " << opCount << " operations\n";
        llvm::errs() << "Starting second pass...\n";
        
        // Map to keep track of variables and their values during emission
        std::unordered_map<std::string, int32_t> varValues;
        int32_t lastPushedValue = 0;
        
        // Second pass: generate VM instructions with correct jump targets
        int processedOps = 0;
        
        // Protect against infinite loops with a maximum number of operations
        module->walk([&](mlir::Operation *op) {
            // Only process a reasonable number of ops to prevent infinite loops
            if (processedOps++ > 1000) {
                throw std::runtime_error("Too many operations in second pass");
            }
            
            if (auto pushOp = mlir::dyn_cast<mlir::basicvm::PushOp>(op)) {
                // Get the value from the attribute
                int32_t value = 0; // Default
                
                if (auto valueAttr = op->getAttrOfType<mlir::IntegerAttr>("value")) {
                    value = valueAttr.getInt();
                }
                
                // Store this as the last pushed value in case we need it later
                lastPushedValue = value;
                
                // Emit the instruction
                instructions.push_back({OpCode::PUSH, value});
                llvm::errs() << "Emitted PUSH " << value << "\n";
            } 
            else if (auto popOp = mlir::dyn_cast<mlir::basicvm::PopOp>(op)) {
                // Handle pop operation
                instructions.push_back({OpCode::POP, 0});
                llvm::errs() << "Emitted POP\n";
            }
            else if (auto addOp = mlir::dyn_cast<mlir::basicvm::AddOp>(op)) {
                // Handle add operation
                instructions.push_back({OpCode::ADD, 0});
                llvm::errs() << "Emitted ADD\n";
            }
            else if (auto subOp = mlir::dyn_cast<mlir::basicvm::SubOp>(op)) {
                // Handle subtract operation
                instructions.push_back({OpCode::SUB, 0});
                llvm::errs() << "Emitted SUB\n";
            }
            else if (auto mulOp = mlir::dyn_cast<mlir::basicvm::MulOp>(op)) {
                // Handle multiply operation
                instructions.push_back({OpCode::MUL, 0});
                llvm::errs() << "Emitted MUL\n";
            }
            else if (auto divOp = mlir::dyn_cast<mlir::basicvm::DivOp>(op)) {
                // Handle divide operation
                instructions.push_back({OpCode::DIV, 0});
                llvm::errs() << "Emitted DIV\n";
            }
            else if (auto printOp = mlir::dyn_cast<mlir::basicvm::PrintOp>(op)) {
                // Handle print operation
                instructions.push_back({OpCode::PRINT, 0});
                llvm::errs() << "Emitted PRINT\n";
            }
            else if (auto storeOp = mlir::dyn_cast<mlir::basicvm::StoreOp>(op)) {
                // Get the register from the attribute
                int32_t reg = 0; // Default register
                
                if (auto regAttr = op->getAttrOfType<mlir::IntegerAttr>("reg")) {
                    reg = regAttr.getInt();
                }
                
                instructions.push_back({OpCode::STORE, reg});
                llvm::errs() << "Emitted STORE " << reg << "\n";
            }
            else if (auto loadOp = mlir::dyn_cast<mlir::basicvm::LoadOp>(op)) {
                // Get the register from the attribute
                int32_t reg = 0; // Default register
                
                if (auto regAttr = op->getAttrOfType<mlir::IntegerAttr>("reg")) {
                    reg = regAttr.getInt();
                }
                
                instructions.push_back({OpCode::LOAD, reg});
                llvm::errs() << "Emitted LOAD " << reg << "\n";
            }
            else if (auto haltOp = mlir::dyn_cast<mlir::basicvm::HaltOp>(op)) {
                // Handle halt operation
                instructions.push_back({OpCode::HALT, 0});
                llvm::errs() << "Emitted HALT\n";
            }
            // Skip handling other operations for now to simplify the code
        });
        
        llvm::errs() << "Second pass complete, generated " << instructions.size() << " instructions\n";
        
        // Make sure we end with a HALT
        if (instructions.empty() || instructions.back().opcode != OpCode::HALT) {
            instructions.push_back({OpCode::HALT, 0});
            llvm::errs() << "Added final HALT instruction\n";
        }
    }
    catch (const std::exception& e) {
        llvm::errs() << "Exception in lowerToVMInstructions: " << e.what() << "\n";
        // Return a minimal error program
        return {
            { OpCode::PUSH, 9998 },  // Error marker
            { OpCode::PRINT, 0 },
            { OpCode::HALT, 0 }
        };
    }
    catch (...) {
        llvm::errs() << "Unknown exception in lowerToVMInstructions\n";
        // Return a minimal error program
        return {
            { OpCode::PUSH, 9997 },  // Error marker
            { OpCode::PRINT, 0 },
            { OpCode::HALT, 0 }
        };
    }
    
    return instructions;
}

// Tokenization method
void MLIRCompiler::tokenizeSource(const std::string& source) {
    tokens.clear();
    currentTokenIndex = 0;
    
    // Keywords in our language
    std::unordered_set<std::string> keywords = {
        "let", "print", "if", "then", "else", "while", "do", "end"
    };
    
    // Operators
    std::unordered_set<char> operators = {
        '+', '-', '*', '/', '=', ';', '(', ')'
    };
    
    size_t line = 1;
    size_t column = 1;
    
    for (size_t i = 0; i < source.length(); ++i) {
        char c = source[i];
        
        // Skip whitespace
        if (std::isspace(c)) {
            if (c == '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
            continue;
        }
        
        // Handle identifiers and keywords
        if (std::isalpha(c) || c == '_') {
            std::string identifier;
            size_t startColumn = column;
            
            while (i < source.length() && (std::isalnum(source[i]) || source[i] == '_')) {
                identifier += source[i];
                i++;
                column++;
            }
            i--; // Move back one character as loop will increment
            
            // Check if it's a keyword
            if (keywords.find(identifier) != keywords.end()) {
                tokens.push_back({TokenType::KEYWORD, identifier, line, startColumn});
            } else {
                tokens.push_back({TokenType::IDENTIFIER, identifier, line, startColumn});
            }
        }
        // Handle numbers
        else if (std::isdigit(c)) {
            std::string number;
            size_t startColumn = column;
            
            while (i < source.length() && std::isdigit(source[i])) {
                number += source[i];
                i++;
                column++;
            }
            i--; // Move back one character as loop will increment
            
            tokens.push_back({TokenType::NUMBER, number, line, startColumn});
        }
        // Handle operators
        else if (operators.find(c) != operators.end()) {
            tokens.push_back({TokenType::OPERATOR, std::string(1, c), line, column});
            column++;
        }
        // Invalid characters
        else {
            std::cerr << "Warning: Invalid character '" << c << "' at line " << line 
                      << ", column " << column << std::endl;
            column++;
        }
    }
    
    // Add end-of-file token
    tokens.push_back({TokenType::END_OF_FILE, "", line, column});
}

// Token access methods
Token& MLIRCompiler::currentToken() {
    if (currentTokenIndex < tokens.size()) {
        return tokens[currentTokenIndex];
    }
    // Return the EOF token if we're past the end
    return tokens.back();
}

Token MLIRCompiler::consumeToken() {
    Token current = currentToken();
    if (currentTokenIndex < tokens.size()) {
        currentTokenIndex++;
    }
    return current;
}

bool MLIRCompiler::matchToken(TokenType type, const std::string& value) {
    if (currentToken().type != type) {
        return false;
    }
    
    if (!value.empty() && currentToken().value != value) {
        return false;
    }
    
    // Match succeeded, consume the token
    consumeToken();
    return true;
}

void MLIRCompiler::expectToken(TokenType type, const std::string& value) {
    if (!matchToken(type, value)) {
        std::string expected = value.empty() ? 
            std::string(tokenTypeToString(type)) : 
            std::string(tokenTypeToString(type)) + " '" + value + "'";
        
        std::string got = currentToken().type == TokenType::END_OF_FILE ? 
            "end of file" : 
            std::string("'") + currentToken().value + "'";
        
        std::stringstream ss;
        ss << "Syntax error at line " << currentToken().line << ", column " << currentToken().column
           << ": Expected " << expected << ", got " << got;
        
        throw std::runtime_error(ss.str());
    }
}

// Register allocation for variables
int32_t MLIRCompiler::getRegisterForVariable(const std::string& name) {
    auto it = variableMap.find(name);
    if (it != variableMap.end()) {
        return it->second;
    }
    
    // Allocate a new register for this variable
    int32_t reg = nextRegister++;
    variableMap[name] = reg;
    return reg;
}

// Update the compile method to use tokenization
std::vector<Instruction> MLIRCompiler::compile(const std::string& source) {
    // Clear any existing state
    variableMap.clear();
    namedBlocks.clear();
    nextRegister = 0;
    
    llvm::errs() << "Using simplified direct compilation without full MLIR\n";
    
    // Tokenize the source code
    tokenizeSource(source);
    llvm::errs() << "✔ tokenizeSource done\n";
    
    // We'll try both optimization approaches
    bool useMLIRPath = false;  // Toggle whether to use MLIR-based optimization
    
    if (useMLIRPath) {
        // APPROACH 1: Use MLIR pathway
        // This creates an MLIR module, applies standard optimizations, and then lowers to VM code
        
        try {
            // Create a new module
            createModule();
            
            // Parse the source code into MLIR - uses the MLIR builder directly
            parseSource("");  // Empty string since we already tokenized
            
            // Run MLIR optimization passes
            llvm::errs() << "Running MLIR optimization passes...\n";
            runOptimizationPasses();
            
            // Lower optimized MLIR to VM instructions
            llvm::errs() << "Lowering optimized MLIR to VM instructions...\n";
            auto program = lowerToVMInstructions();
            
            llvm::errs() << "Generated " << program.size() << " VM instructions via MLIR pathway\n";
            return program;
        }
        catch (const std::exception& e) {
            llvm::errs() << "Error in MLIR path: " << e.what() << "\n";
            // Fall back to direct compilation approach
        }
    }
    
    // APPROACH 2: Direct VM code generation with custom optimizations
    // Create a vector to store our VM instructions
    std::vector<Instruction> program;
    
    // Parse tokens and generate code directly
    currentTokenIndex = 0;
    
    try {
        // Keep parsing statements until we reach the end of the file
        while (currentToken().type != TokenType::END_OF_FILE) {
            // Check for different statement types
            if (currentToken().type == TokenType::KEYWORD) {
                if (currentToken().value == "let") {
                    // Variable declaration: let x value;
                    consumeToken(); // Consume "let"
                    
                    // Expect variable name
                    if (currentToken().type != TokenType::IDENTIFIER) {
                        throw std::runtime_error("Expected identifier after 'let'");
                    }
                    std::string varName = consumeToken().value;
                    int32_t reg = getRegisterForVariable(varName);
                    
                    // Parse the expression and push its value onto the stack
                    parseSimpleExpression(program);
                    
                    // Store the value into the variable's register
                    program.push_back({OpCode::STORE, reg});
                    
                    // Expect semicolon
                    if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                        consumeToken();
                    } else {
                        throw std::runtime_error("Expected ';' after variable declaration");
                    }
                }
                else if (currentToken().value == "print") {
                    // Print statement: print expr;
                    consumeToken(); // Consume "print"
                    
                    // Parse the expression to print
                    parseSimpleExpression(program);
                    
                    // Add PRINT instruction
                    program.push_back({OpCode::PRINT, 0});
                    
                    // Expect semicolon
                    if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                        consumeToken();
                    } else {
                        throw std::runtime_error("Expected ';' after print statement");
                    }
                }
                else if (currentToken().value == "if") {
                    // If statement: if expr then stmts end;
                    consumeToken(); // Consume "if"
                    
                    // Parse the condition expression
                    parseSimpleExpression(program);
                    
                    // Compare with 0 (false)
                    program.push_back({OpCode::PUSH, 0});
                    program.push_back({OpCode::CMP, 0});
                    
                    // Calculate the jump target (placeholder for now)
                    size_t jumpZeroIndex = program.size();
                    program.push_back({OpCode::JZ, 0}); // Placeholder, will be fixed later
                    
                    // Expect "then" keyword
                    if (currentToken().type == TokenType::KEYWORD && currentToken().value == "then") {
                        consumeToken();
                    } else {
                        throw std::runtime_error("Expected 'then' after if condition");
                    }
                    
                    // Parse the statements inside the if block
                    while (currentToken().type != TokenType::KEYWORD || currentToken().value != "end") {
                        if (currentToken().type == TokenType::END_OF_FILE) {
                            throw std::runtime_error("Unexpected end of file inside if block");
                        }
                        
                        // Parse statement recursively
                        if (currentToken().type == TokenType::KEYWORD) {
                            if (currentToken().value == "let") {
                                // Parse let statement
                                consumeToken(); // Consume "let"
                                
                                // Expect variable name
                                if (currentToken().type != TokenType::IDENTIFIER) {
                                    throw std::runtime_error("Expected identifier after 'let'");
                                }
                                std::string varName = consumeToken().value;
                                int32_t reg = getRegisterForVariable(varName);
                                
                                // Parse the expression
                                parseSimpleExpression(program);
                                
                                // Store the value
                                program.push_back({OpCode::STORE, reg});
                                
                                // Expect semicolon
                                if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                                    consumeToken();
                                } else {
                                    throw std::runtime_error("Expected ';' after variable declaration");
                                }
                            }
                            else if (currentToken().value == "print") {
                                // Parse print statement
                                consumeToken(); // Consume "print"
                                
                                // Parse the expression
                                parseSimpleExpression(program);
                                
                                // Add print instruction
                                program.push_back({OpCode::PRINT, 0});
                                
                                // Expect semicolon
                                if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                                    consumeToken();
                                } else {
                                    throw std::runtime_error("Expected ';' after print statement");
                                }
                            }
                            else {
                                throw std::runtime_error("Unsupported keyword inside if block: " + currentToken().value);
                            }
                        }
                        else {
                            throw std::runtime_error("Unexpected token inside if block: " + currentToken().value);
                        }
                    }
                    
                    // Consume "end" keyword
                    consumeToken();
                    
                    // Expect semicolon
                    if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                        consumeToken();
                    } else {
                        throw std::runtime_error("Expected ';' after end of if block");
                    }
                    
                    // Fix up the JZ instruction to jump to the current position (after the if block)
                    program[jumpZeroIndex].operand = static_cast<int32_t>(program.size());
                }
                else if (currentToken().value == "while") {
                    // While loop: while expr do stmts end;
                    consumeToken(); // Consume "while"
                    
                    // Record the position of the loop start (for the backward jump)
                    size_t loopStartIndex = program.size();
                    
                    // Parse the condition expression
                    parseSimpleExpression(program);
                    
                    // Compare with 0 (false)
                    program.push_back({OpCode::PUSH, 0});
                    program.push_back({OpCode::CMP, 0});
                    
                    // Calculate the jump target (placeholder for now)
                    size_t jumpZeroIndex = program.size();
                    program.push_back({OpCode::JZ, 0}); // Placeholder, will be fixed later
                    
                    // Expect "do" keyword
                    if (currentToken().type == TokenType::KEYWORD && currentToken().value == "do") {
                        consumeToken();
                    } else {
                        throw std::runtime_error("Expected 'do' after while condition");
                    }
                    
                    // Parse the statements inside the while block
                    while (currentToken().type != TokenType::KEYWORD || currentToken().value != "end") {
                        if (currentToken().type == TokenType::END_OF_FILE) {
                            throw std::runtime_error("Unexpected end of file inside while block");
                        }
                        
                        // Parse statement recursively
                        if (currentToken().type == TokenType::KEYWORD) {
                            if (currentToken().value == "let") {
                                // Parse let statement
                                consumeToken(); // Consume "let"
                                
                                // Expect variable name
                                if (currentToken().type != TokenType::IDENTIFIER) {
                                    throw std::runtime_error("Expected identifier after 'let'");
                                }
                                std::string varName = consumeToken().value;
                                int32_t reg = getRegisterForVariable(varName);
                                
                                // Parse the expression
                                parseSimpleExpression(program);
                                
                                // Store the value
                                program.push_back({OpCode::STORE, reg});
                                
                                // Expect semicolon
                                if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                                    consumeToken();
                                } else {
                                    throw std::runtime_error("Expected ';' after variable declaration");
                                }
                            }
                            else if (currentToken().value == "print") {
                                // Parse print statement
                                consumeToken(); // Consume "print"
                                
                                // Parse the expression
                                parseSimpleExpression(program);
                                
                                // Add print instruction
                                program.push_back({OpCode::PRINT, 0});
                                
                                // Expect semicolon
                                if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                                    consumeToken();
                                } else {
                                    throw std::runtime_error("Expected ';' after print statement");
                                }
                            }
                            else {
                                throw std::runtime_error("Unsupported keyword inside while block: " + currentToken().value);
                            }
                        }
                        else {
                            throw std::runtime_error("Unexpected token inside while block: " + currentToken().value);
                        }
                    }
                    
                    // Add the jump back to the loop start
                    program.push_back({OpCode::JMP, static_cast<int32_t>(loopStartIndex)});
                    
                    // Consume "end" keyword
                    consumeToken();
                    
                    // Expect semicolon
                    if (currentToken().type == TokenType::OPERATOR && currentToken().value == ";") {
                        consumeToken();
                    } else {
                        throw std::runtime_error("Expected ';' after end of while block");
                    }
                    
                    // Fix up the JZ instruction to jump to the current position (after the while block)
                    program[jumpZeroIndex].operand = static_cast<int32_t>(program.size());
                }
                else {
                    // Unsupported keyword (for now)
                    throw std::runtime_error("Unsupported keyword: " + currentToken().value);
                }
            }
            else {
                // Unexpected token
                throw std::runtime_error("Unexpected token: " + currentToken().value);
            }
        }
        
        // Always end with HALT
        program.push_back({OpCode::HALT, 0});
        
        llvm::errs() << "Generated " << program.size() << " VM instructions through token parsing\n";
        
        // Apply VM-level optimizations
        optimizeVMInstructions(program);
        
        llvm::errs() << "After optimization: " << program.size() << " VM instructions\n";
    }
    catch (const std::exception& e) {
        llvm::errs() << "Error during simplified compilation: " << e.what() << "\n";
        
        // Return a minimal error program
        return {
            { OpCode::PUSH, 9999 },  // Error marker
            { OpCode::PRINT, 0 },
            { OpCode::HALT, 0 }
        };
    }
    
    return program;
}

// Helper to parse a simple expression (currently just handles variables and literals)
void MLIRCompiler::parseSimpleExpression(std::vector<Instruction>& program) {
    if (currentToken().type == TokenType::NUMBER) {
        // It's a number literal
        int32_t value = std::stoi(consumeToken().value);
        program.push_back({OpCode::PUSH, value});
    }
    else if (currentToken().type == TokenType::IDENTIFIER) {
        // It's a variable reference
        std::string varName = consumeToken().value;
        int32_t reg = getRegisterForVariable(varName);
        program.push_back({OpCode::LOAD, reg});
        
        // Check for binary operators
        if (currentToken().type == TokenType::OPERATOR) {
            if (currentToken().value == "+" || 
                currentToken().value == "-" ||
                currentToken().value == "*" ||
                currentToken().value == "/") {
                
                std::string op = consumeToken().value;
                
                // Parse the right-hand side of the expression
                parseSimpleExpression(program);
                
                // Add the appropriate operator instruction
                if (op == "+") {
                    program.push_back({OpCode::ADD, 0});
                }
                else if (op == "-") {
                    program.push_back({OpCode::SUB, 0});
                }
                else if (op == "*") {
                    program.push_back({OpCode::MUL, 0});
                }
                else if (op == "/") {
                    program.push_back({OpCode::DIV, 0});
                }
            }
        }
    }
    else {
        throw std::runtime_error("Expected number or identifier in expression");
    }
}

void MLIRCompiler::dumpMLIR() const {
    llvm::errs() << "Using simplified direct code generation approach\n";
    llvm::errs() << "The source code will be compiled directly to VM Instructions.\n";
    
    // Show variable mappings
    llvm::errs() << "Variable register mappings:\n";
    for (const auto& [name, reg] : variableMap) {
        llvm::errs() << "  " << name << " -> register " << reg << "\n";
    }
    
    llvm::errs() << "Supported operations:\n";
    llvm::errs() << "  - Variable declarations: 'let x 5;'\n";
    llvm::errs() << "  - Variable references: 'x'\n";
    llvm::errs() << "  - Binary operations: 'x + y', 'x - y', 'x * y', 'x / y'\n";
    llvm::errs() << "  - Print statements: 'print x;'\n";
    llvm::errs() << "  - Conditional statements: 'if expr then ... end;'\n";
    llvm::errs() << "  - While loops: 'while expr do ... end;'\n";
    
    llvm::errs() << "\nOptimizations applied:\n";
    llvm::errs() << "  - Constant folding (e.g., replacing '5 + 10' with '15')\n";
    llvm::errs() << "  - Dead code elimination\n";
    llvm::errs() << "  - Peephole optimizations (e.g., removing redundant operations)\n";
    llvm::errs() << "  - Jump optimizations\n";
}

// Operation emission methods

void MLIRCompiler::emitPushConstant(int32_t value) {
    // Create operation with proper build method
    auto pushOp = builder->create<mlir::basicvm::PushOp>(
        builder->getUnknownLoc()
    );
    
    // Store the value as an attribute - first create a string to document what we pushed
    std::string debugInfo = "value:" + std::to_string(value);
    pushOp->setAttr("value", builder->getI32IntegerAttr(value));
    pushOp->setAttr("debugInfo", builder->getStringAttr(debugInfo));
    
    // For debugging
    llvm::errs() << "Emitting PUSH " << value << "\n";
}

void MLIRCompiler::emitLoadVariable(const std::string& name) {
    int32_t reg = getRegisterForVariable(name);
    
    // Create operation with proper build method
    auto loadOp = builder->create<mlir::basicvm::LoadOp>(
        builder->getUnknownLoc()
    );
    
    // Set the register attribute
    loadOp->setAttr("reg", builder->getI32IntegerAttr(reg));
    loadOp->setAttr("varName", builder->getStringAttr(name));
    
    // For debugging
    llvm::errs() << "Emitting LOAD from reg " << reg << " (var " << name << ")\n";
}

void MLIRCompiler::emitStoreVariable(const std::string& name) {
    int32_t reg = getRegisterForVariable(name);
    
    // Create operation with proper build method
    auto storeOp = builder->create<mlir::basicvm::StoreOp>(
        builder->getUnknownLoc()
    );
    
    // Set the register attribute
    storeOp->setAttr("reg", builder->getI32IntegerAttr(reg));
    storeOp->setAttr("varName", builder->getStringAttr(name));
    
    // For debugging
    llvm::errs() << "Emitting STORE to reg " << reg << " (var " << name << ")\n";
}

void MLIRCompiler::emitBinaryOp(const std::string& op) {
    if (op == "+") {
        [[maybe_unused]] auto addOp = builder->create<mlir::basicvm::AddOp>(
            builder->getUnknownLoc()
        );
        llvm::errs() << "Emitting ADD\n";
    } 
    else if (op == "-") {
        [[maybe_unused]] auto subOp = builder->create<mlir::basicvm::SubOp>(
            builder->getUnknownLoc()
        );
        llvm::errs() << "Emitting SUB\n";
    }
    else if (op == "*") {
        [[maybe_unused]] auto mulOp = builder->create<mlir::basicvm::MulOp>(
            builder->getUnknownLoc()
        );
        llvm::errs() << "Emitting MUL\n";
    }
    else if (op == "/") {
        [[maybe_unused]] auto divOp = builder->create<mlir::basicvm::DivOp>(
            builder->getUnknownLoc()
        );
        llvm::errs() << "Emitting DIV\n";
    }
    else {
        llvm::errs() << "Unknown binary operator: " << op << "\n";
    }
}

void MLIRCompiler::emitPrint() {
    [[maybe_unused]] auto printOp = builder->create<mlir::basicvm::PrintOp>(
        builder->getUnknownLoc()
    );
    llvm::errs() << "Emitting PRINT\n";
}

void MLIRCompiler::emitJump(mlir::Block* target) {
    [[maybe_unused]] auto jumpOp = builder->create<mlir::basicvm::JumpOp>(
        builder->getUnknownLoc()
    );
    
    // In a full implementation, we would set the successor block
    // jumpOp.successor(target);
    
    llvm::errs() << "Emitting JMP\n";
}

void MLIRCompiler::emitConditionalJump(mlir::Block* target, bool jumpIfZero) {
    if (jumpIfZero) {
        [[maybe_unused]] auto jzOp = builder->create<mlir::basicvm::JumpZeroOp>(
            builder->getUnknownLoc()
        );
        // jzOp.successor(target);
        llvm::errs() << "Emitting JZ\n";
    } else {
        [[maybe_unused]] auto jnzOp = builder->create<mlir::basicvm::JumpNonZeroOp>(
            builder->getUnknownLoc()
        );
        // jnzOp.successor(target);
        llvm::errs() << "Emitting JNZ\n";
    }
}

void MLIRCompiler::emitCompare() {
    [[maybe_unused]] auto cmpOp = builder->create<mlir::basicvm::CompareOp>(
        builder->getUnknownLoc()
    );
    llvm::errs() << "Emitting CMP\n";
}

// Now let's implement the various parsing methods for expressions

void MLIRCompiler::parseExpression() {
    // Expression = Term { ('+' | '-') Term }
    parseTerm();
    
    while (currentToken().type == TokenType::OPERATOR && 
           (currentToken().value == "+" || currentToken().value == "-")) {
        std::string op = consumeToken().value;
        parseTerm();
        emitBinaryOp(op);
    }
}

void MLIRCompiler::parseTerm() {
    // Term = Factor { ('*' | '/') Factor }
    parseFactor();
    
    while (currentToken().type == TokenType::OPERATOR && 
           (currentToken().value == "*" || currentToken().value == "/")) {
        std::string op = consumeToken().value;
        parseFactor();
        emitBinaryOp(op);
    }
}

void MLIRCompiler::parseFactor() {
    // Factor = Number | Identifier | '(' Expression ')'
    if (currentToken().type == TokenType::NUMBER) {
        int32_t value = std::stoi(consumeToken().value);
        emitPushConstant(value);
    }
    else if (currentToken().type == TokenType::IDENTIFIER) {
        std::string varName = consumeToken().value;
        emitLoadVariable(varName);
    }
    else if (matchToken(TokenType::OPERATOR, "(")) {
        parseExpression();
        expectToken(TokenType::OPERATOR, ")");
    }
    else {
        std::stringstream ss;
        ss << "Syntax error at line " << currentToken().line << ", column " << currentToken().column
           << ": Expected number, identifier, or '(', got '" << currentToken().value << "'";
        throw std::runtime_error(ss.str());
    }
}

// Now implement statement parsing methods

void MLIRCompiler::parseStatement() {
    if (currentToken().type == TokenType::KEYWORD) {
        if (currentToken().value == "let") {
            parseVarDeclaration();
        }
        else if (currentToken().value == "print") {
            parsePrintStatement();
        }
        else if (currentToken().value == "while") {
            parseWhileLoop();
        }
        else if (currentToken().value == "if") {
            parseIfStatement();
        }
        else {
            std::stringstream ss;
            ss << "Unknown keyword at line " << currentToken().line << ", column " << currentToken().column
               << ": '" << currentToken().value << "'";
            throw std::runtime_error(ss.str());
        }
    }
    else {
        std::stringstream ss;
        ss << "Syntax error at line " << currentToken().line << ", column " << currentToken().column
           << ": Expected statement keyword, got '" << currentToken().value << "'";
        throw std::runtime_error(ss.str());
    }
}

void MLIRCompiler::parseVarDeclaration() {
    // let <identifier> <expression>;  (no equals sign required)
    expectToken(TokenType::KEYWORD, "let");
    std::string varName;
    
    if (currentToken().type == TokenType::IDENTIFIER) {
        varName = consumeToken().value;
    } else {
        std::stringstream ss;
        ss << "Syntax error at line " << currentToken().line << ", column " << currentToken().column
           << ": Expected identifier after 'let', got '" << currentToken().value << "'";
        throw std::runtime_error(ss.str());
    }
    
    // No need to expect '=' operator
    parseExpression();
    expectToken(TokenType::OPERATOR, ";");
    
    // After expression is evaluated, its result will be on the stack
    // Store it in the variable's register
    emitStoreVariable(varName);
}

void MLIRCompiler::parsePrintStatement() {
    // print <expression>;
    expectToken(TokenType::KEYWORD, "print");
    parseExpression();
    expectToken(TokenType::OPERATOR, ";");
    
    // After expression is evaluated, its result will be on the stack
    // Print it
    emitPrint();
}

void MLIRCompiler::parseWhileLoop() {
    // while <expression> do <statements> end;
    expectToken(TokenType::KEYWORD, "while");
    
    // Create blocks for condition, body, and after the loop
    mlir::Block* condBlock = new mlir::Block();
    mlir::Block* bodyBlock = new mlir::Block();
    mlir::Block* afterBlock = new mlir::Block();
    
    // First, jump to the condition block
    emitJump(condBlock);
    
    // Add condition block and set insertion point
    builder->getBlock()->getParent()->push_back(condBlock);
    builder->setInsertionPointToStart(condBlock);
    
    // Parse condition expression
    parseExpression();
    
    // Compare the expression result with zero
    emitPushConstant(0);
    emitCompare();
    
    // If condition is false (zero), jump to after block
    emitConditionalJump(afterBlock, true);
    
    // Add body block and set insertion point
    builder->getBlock()->getParent()->push_back(bodyBlock);
    builder->setInsertionPointToStart(bodyBlock);
    
    // Parse the loop body
    expectToken(TokenType::KEYWORD, "do");
    while (currentToken().type != TokenType::KEYWORD || 
           currentToken().value != "end") {
        parseStatement();
    }
    consumeToken(); // Consume "end"
    expectToken(TokenType::OPERATOR, ";");
    
    // At the end of the loop body, jump back to condition
    emitJump(condBlock);
    
    // Add after block and set insertion point for code after the loop
    builder->getBlock()->getParent()->push_back(afterBlock);
    builder->setInsertionPointToStart(afterBlock);
}

void MLIRCompiler::parseIfStatement() {
    // if <expression> then <statements> end;
    expectToken(TokenType::KEYWORD, "if");
    
    // Create blocks for then branch and after the if
    mlir::Block* thenBlock = new mlir::Block();
    mlir::Block* afterBlock = new mlir::Block();
    
    // Parse condition expression
    parseExpression();
    
    // Compare the expression result with zero
    emitPushConstant(0);
    emitCompare();
    
    // If condition is false (zero), jump to after block
    emitConditionalJump(afterBlock, true);
    
    // Add then block and set insertion point
    builder->getBlock()->getParent()->push_back(thenBlock);
    builder->setInsertionPointToStart(thenBlock);
    
    // Parse the statements in the then branch
    expectToken(TokenType::KEYWORD, "then");
    while (currentToken().type != TokenType::KEYWORD || 
           currentToken().value != "end") {
        parseStatement();
    }
    consumeToken(); // Consume "end"
    expectToken(TokenType::OPERATOR, ";");
    
    // Jump to after block at end of then branch
    emitJump(afterBlock);
    
    // Add after block and set insertion point for code after the if
    builder->getBlock()->getParent()->push_back(afterBlock);
    builder->setInsertionPointToStart(afterBlock);
}

// Optimize VM instructions directly
void MLIRCompiler::optimizeVMInstructions(std::vector<Instruction>& program) {
    llvm::errs() << "Applying VM instruction optimizations\n";
    
    // No instructions to optimize
    if (program.empty()) {
        return;
    }
    
    // Track how many optimizations we've applied
    int optimizationCount = 0;
    
    // 1. Constant folding: Replace sequences of PUSH followed by arithmetic operations with 
    //    a single PUSH of the computed value
    for (size_t i = 0; i < program.size() - 2; i++) {
        // Look for two consecutive PUSH operations followed by an arithmetic operation
        if (program[i].opcode == OpCode::PUSH && 
            i + 2 < program.size() && 
            program[i+1].opcode == OpCode::PUSH) {
            
            int32_t a = program[i].operand;
            int32_t b = program[i+1].operand;
            int32_t result = 0;
            bool canFold = false;
            
            // Check if followed by an arithmetic operation
            if (i + 2 < program.size()) {
                switch (program[i+2].opcode) {
                    case OpCode::ADD:
                        result = a + b;
                        canFold = true;
                        break;
                    case OpCode::SUB:
                        result = a - b;
                        canFold = true;
                        break;
                    case OpCode::MUL:
                        result = a * b;
                        canFold = true;
                        break;
                    case OpCode::DIV:
                        // Avoid division by zero
                        if (b != 0) {
                            result = a / b;
                            canFold = true;
                        }
                        break;
                    default:
                        // Not an arithmetic operation
                        break;
                }
            }
            
            // If we can fold the operations, replace them with a single PUSH
            if (canFold) {
                program[i] = {OpCode::PUSH, result};
                // Remove the next two instructions (second PUSH and arithmetic op)
                program.erase(program.begin() + i + 1, program.begin() + i + 3);
                optimizationCount++;
                // Don't increment i since we want to check the new sequence starting at i
                i--;
            }
        }
    }
    
    // 2. Dead code elimination: Remove unreachable code after HALT
    for (size_t i = 0; i < program.size() - 1; i++) {
        if (program[i].opcode == OpCode::HALT) {
            // All instructions after HALT are unreachable
            size_t unreachableCount = program.size() - (i + 1);
            if (unreachableCount > 0) {
                program.resize(i + 1);
                optimizationCount += static_cast<int>(unreachableCount);
                break; // No need to continue once we've found a HALT
            }
        }
    }
    
    // 3. Peephole optimizations
    for (size_t i = 0; i < program.size() - 1; i++) {
        // LOAD followed by STORE to the same register can be eliminated
        if (i + 1 < program.size() && 
            program[i].opcode == OpCode::LOAD && 
            program[i+1].opcode == OpCode::STORE && 
            program[i].operand == program[i+1].operand) {
            
            // Remove both instructions
            program.erase(program.begin() + i, program.begin() + i + 2);
            optimizationCount++;
            i--;
            continue;
        }
        
        // PUSH 0 followed by ADD is redundant (x + 0 = x)
        if (i + 1 < program.size() && 
            program[i].opcode == OpCode::PUSH && 
            program[i].operand == 0 && 
            program[i+1].opcode == OpCode::ADD) {
            
            // Remove both instructions
            program.erase(program.begin() + i, program.begin() + i + 2);
            optimizationCount++;
            i--;
            continue;
        }
        
        // PUSH 1 followed by MUL is redundant (x * 1 = x)
        if (i + 1 < program.size() && 
            program[i].opcode == OpCode::PUSH && 
            program[i].operand == 1 && 
            program[i+1].opcode == OpCode::MUL) {
            
            // Remove both instructions
            program.erase(program.begin() + i, program.begin() + i + 2);
            optimizationCount++;
            i--;
            continue;
        }
    }
    
    // 4. Jump Optimization: Simplify jumps to jumps
    for (size_t i = 0; i < program.size(); i++) {
        if (program[i].opcode == OpCode::JMP) {
            int32_t target = program[i].operand;
            
            // If the target is valid and is itself a JMP, directly jump to the final target
            if (target >= 0 && target < static_cast<int32_t>(program.size()) && 
                program[target].opcode == OpCode::JMP) {
                
                program[i].operand = program[target].operand;
                optimizationCount++;
            }
        }
    }
    
    // Report the optimizations
    if (optimizationCount > 0) {
        llvm::errs() << "Applied " << optimizationCount << " optimizations\n";
    } else {
        llvm::errs() << "No optimizations applied\n";
    }
} 