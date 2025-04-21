# BasicVM - A Simple Virtual Machine

A modular, extensible virtual machine implementation in C++ with a separate assembler and compiler.

## Architecture

The project is divided into three main components:

1. **Virtual Machine (VM)**: Executes instructions and manages memory
2. **Assembler**: Converts assembly code into VM instructions
3. **Compiler**: Translates a high-level language into VM instructions

This separation allows for easier extension and maintenance of each component.

## Features

- Stack-based operations (PUSH, POP, arithmetic operations)
- Register storage
- Heap memory allocation and management
- Control flow (jumps, conditional execution)
- Extensible instruction set via handler functions
- Separate assembler for converting text to bytecode
- High-level language compiler

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

## High-Level Language

The compiler supports a simple high-level language with the following features:

- Variable declarations and assignments
- Arithmetic expressions
- Print statements
- While loops
- If statements

### Language Grammar
