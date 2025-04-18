#include "vm.h"
#include <iostream>

int main() {
    VirtualMachine vm;
    
    // Example 1: Calculate 5 + 3 * 2
    std::cout << "Example 1: Calculate 5 + 3 * 2" << std::endl;
    vm.addInstruction(OpCode::PUSH, 5);  // Push 5 onto stack
    vm.addInstruction(OpCode::PUSH, 3);  // Push 3 onto stack
    vm.addInstruction(OpCode::PUSH, 2);  // Push 2 onto stack
    vm.addInstruction(OpCode::MUL);      // Multiply 3 * 2
    vm.addInstruction(OpCode::ADD);      // Add 5 + (3 * 2)
    vm.addInstruction(OpCode::PRINT);    // Print result
    vm.addInstruction(OpCode::HALT);     // Stop execution
    
    vm.execute();
    
    // Example 2: Using registers and loops to calculate factorial of 5
    std::cout << "\nExample 2: Calculate factorial of 5" << std::endl;
    
    // Assemble program from string
    std::string factorialProgram = R"(
        PUSH 5       ; n = 5
        STORE 0      ; store n in register 0
        PUSH 1       ; result = 1
        STORE 1      ; store result in register 1
        
    ; Loop start
        LOAD 0       ; load n
        PUSH 0       ; push 0 for comparison
        CMP          ; compare n with 0
        JZ 17        ; if n == 0, jump to end (instruction 17)
        
        LOAD 1       ; load result
        LOAD 0       ; load n
        MUL          ; result = result * n
        STORE 1      ; store result back to register 1
        
        LOAD 0       ; load n
        PUSH 1       ; push 1
        SUB          ; n = n - 1
        STORE 0      ; store n back to register 0
        
        JMP 4        ; jump back to loop start
        
    ; End
        LOAD 1       ; load result
        PRINT        ; print result
        HALT         ; stop execution
    )";
    
    vm.loadProgram(VirtualMachine::assembleProgram(factorialProgram));
    std::cout << "Program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
    vm.execute();
    
    return 0;
} 