#include "vm.h"
#include "assembler.h"
#include <iostream>
#include <string>

int main() {
    // Create a virtual machine and assembler
    VirtualMachine vm;
    Assembler assembler;
    
    // Define a simple program with variable assignment without equals sign
    // The program is equivalent to:
    // let x = 42;
    // print x;
    // let y = x + 10;
    // print y;
    std::string program = R"(
        ; Assign a value to register 5 (variable x)
        PUSH 42      ; Push 42 onto the stack
        STORE 5      ; Store value in register 5 (variable x)
        LOAD 5       ; Load value from register 5
        PRINT        ; Print the value
        
        ; Calculate x + 10 and store in register 6 (variable y)
        LOAD 5       ; Load x from register 5
        PUSH 10      ; Push 10 onto the stack
        ADD          ; Add x + 10
        STORE 6      ; Store result in register 6 (variable y)
        LOAD 6       ; Load y from register 6
        PRINT        ; Print the value
        
        HALT         ; Stop execution
    )";
    
    std::cout << "Assembling and executing program:\n" << program << std::endl;
    
    // Assemble the program
    auto instructions = assembler.assemble(program);
    
    // Load the program into the VM
    vm.loadProgram(instructions);
    
    // Execute the program
    std::cout << "Executing program with " << vm.getProgramSize() << " instructions..." << std::endl;
    vm.execute();
    
    std::cout << "\nExecution complete." << std::endl;
    
    return 0;
} 