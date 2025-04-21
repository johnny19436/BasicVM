# BasicVM - A Simple Virtual Machine

A modular, extensible virtual machine implementation in C++ with a separate assembler and MLIR-based compiler integration.

## Architecture

The project is divided into these main components:

1. **Virtual Machine (VM)**: Executes instructions and manages memory
2. **Assembler**: Converts assembly code into VM instructions
3. **MLIR Integration**: Uses LLVM's MLIR infrastructure for compilation

This separation allows for easier extension and maintenance of each component.

## Features

- Stack-based operations (PUSH, POP, arithmetic operations)
- Register storage
- Heap memory allocation and management
- Control flow (jumps, conditional execution)
- Extensible instruction set via handler functions
- Separate assembler for converting text to bytecode
- MLIR integration for modern compiler infrastructure

## MLIR Integration

The project leverages LLVM's Multi-Level Intermediate Representation (MLIR) for advanced compiler optimizations:

- MLIR-based integration for optimization and code generation
- Access to MLIR's powerful optimization passes
- A modular approach using MLIR's operation-based IR design
- A clean separation between the high-level language and VM instruction generation

### MLIR Layer Architecture

The MLIR layer is implemented through several components:

1. **MLIRIntegration**: The main interface between the VM and MLIR compiler
   - Provides a clean API for compiling high-level source code to VM instructions
   - Handles error reporting and recovery
   - Acts as a facade for the underlying MLIR infrastructure

2. **MLIRCompiler**: The core compiler implementation
   - Tokenizes and parses the source language
   - Builds MLIR operations representing the program
   - Applies optimization passes
   - Lowers MLIR operations to VM instructions

3. **BasicVMDialect**: A custom MLIR dialect for the VM
   - Defines VM-specific operations (PUSH, POP, ADD, etc.)
   - Provides serialization and deserialization of operations
   - Enables integration with MLIR's optimization infrastructure

### MLIR Integration Configuration

The MLIR integration is configured in several steps:

1. **Dialect Registration**:
   ```cpp
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
   ```

2. **Operation Registration**:
   ```cpp
   // Register operations in the BasicVM dialect
   addOperations<
     HaltOp,
     PushOp,
     PopOp,
     AddOp,
     // ... other operations
   >();
   ```

3. **Compilation Pipeline**:
   ```cpp
   // Create a pass manager
   mlir::PassManager pm(context.get());
   
   // Add optimization passes
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addPass(mlir::createCSEPass());
   ```

4. **Dual Compilation Pathways**:
   - MLIR-based: Uses full MLIR infrastructure with all optimizations
   - Direct: Fast, token-based compilation with custom optimizations

The implementation allows toggling between these pathways:
```cpp
// Toggle between MLIR path and direct compilation
bool useMLIRPath = false;  // Set to true to use full MLIR pipeline
```

### Using the MLIR Integration

The MLIR integration is the primary compiler for the VM:

```cpp
// Create an MLIR integration instance
MLIRIntegration mlirIntegration;

// Compile source code to VM instructions
std::vector<Instruction> instructions = mlirIntegration.compile(sourceCode);

// Execute the compiled instructions
VirtualMachine vm;
vm.loadProgram(instructions);
vm.execute();
```

### MLIR Optimizer

The MLIR-based optimizer applies several optimization techniques:

1. **Constant Folding**: Evaluates constant expressions at compile time
   ```cpp
   // Before: PUSH 5, PUSH 3, ADD
   // After:  PUSH 8
   ```

2. **Dead Code Elimination**: Removes unreachable code
   ```cpp
   // Remove code after HALT instruction or unreachable branches
   ```

3. **Common Subexpression Elimination**: Avoids redundant computations
   ```cpp
   // Using MLIR's CSE pass to identify and eliminate duplicate expressions
   ```

4. **Peephole Optimizations**: Pattern-based local optimizations
   ```cpp
   // Example: PUSH 0 followed by ADD can be eliminated
   ```

### Benefits of MLIR

- Modular optimization pipeline
- Extensible IR with powerful abstraction capabilities
- Easier implementation of language features
- Better optimization opportunities
- Potential for targeting multiple backends in the future

## Instruction Set

| Instruction | Description |
|-------------|-------------|
| HALT        | Stop execution |
| PUSH n      | Push value n onto stack |
| POP         | Remove top value from stack |
| ADD         | Add top two values on stack |
| SUB         | Subtract top two values on stack |
| MUL         | Multiply top two values on stack |
| DIV         | Divide top two values on stack |
| PRINT       | Print top value on stack |
| STORE r     | Store top value in register r |
| LOAD r      | Push value from register r onto stack |
| JMP addr    | Jump to address |
| JZ addr     | Jump to address if comparison flag is zero |
| JNZ addr    | Jump to address if comparison flag is not zero |
| CMP         | Compare top two values on stack |
| ALLOC       | Allocate memory on heap (size from stack), returns address |
| FREE        | Free memory on heap (address from stack) |
| STORE_HEAP  | Store value at heap address (addr, offset, value from stack) |
| LOAD_HEAP   | Load value from heap address (addr, offset from stack) |

## Building & Running

To build the project with MLIR support, you need LLVM with MLIR enabled:

```bash
# Install LLVM with MLIR support (using Homebrew on macOS)
brew install llvm

# Set up environment variables for LLVM
export LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
export MLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir

# Build the VM
make all

# Run the VM with examples
make run
```

### LLVM/MLIR Dependency Configuration

The project requires LLVM and MLIR libraries for compilation. The configuration is handled in the Makefile:

```makefile
# LLVM/MLIR configuration
LLVM_CONFIG = llvm-config
LLVM_CXXFLAGS = $(shell $(LLVM_CONFIG) --cxxflags)
LLVM_LDFLAGS = $(shell $(LLVM_CONFIG) --ldflags)
LLVM_LIBS = $(shell $(LLVM_CONFIG) --libs core support)
MLIR_LIBS = -lMLIR -lMLIRDialect -lMLIRExecutionEngine

# Project compile flags
CXXFLAGS = -std=c++17 -Wall -Wextra $(LLVM_CXXFLAGS)
LDFLAGS = $(LLVM_LDFLAGS) $(LLVM_LIBS) $(MLIR_LIBS)
```

## High-Level Language

The MLIR compiler supports a simple high-level language with the following features:

- Variable declarations and assignments
- Arithmetic expressions
- Print statements
- While loops
- If statements

### Language Grammar

```
program ::= statement*
statement ::= assignment | print_statement | while_statement | if_statement
assignment ::= "let" identifier expression ";"
print_statement ::= "print" expression ";"
while_statement ::= "while" expression "do" statement* "end"
if_statement ::= "if" expression "then" statement* "end"
expression ::= term ("+" term | "-" term)*
term ::= factor ("*" factor | "/" factor)*
factor ::= number | identifier | "(" expression ")"
```

## Project Organization

```
basic_VM/
├── main.cpp              # Main executable with test cases
├── vm.h                  # Virtual Machine API definition
├── vm.cpp                # Virtual Machine implementation
├── assembler.h           # Assembler API definition
├── assembler.cpp         # Assembler implementation
├── mlir_integration.h    # MLIR integration API definition
├── mlir_integration.cpp  # MLIR integration implementation
├── mlir_compiler.h       # MLIR compiler API definition
├── mlir_compiler.cpp     # MLIR compiler implementation
├── basic_vm_dialect.h    # BasicVM MLIR dialect definition
├── basic_vm_dialect.cpp  # BasicVM MLIR dialect implementation
└── Makefile              # Project build configuration
```

## Future Enhancements

- Enhanced MLIR dialect specifically for our VM instructions
- Advanced optimizations using MLIR's transformation infrastructure
- JIT compilation using LLVM
- More complex language features
