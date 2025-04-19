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
    
    // Example 3: Using heap memory to create and manipulate an array
    std::cout << "\nExample 3: Using heap memory to create and manipulate an array" << std::endl;
    
    // Assemble program from string
    std::string heapProgram = R"(
        ; Allocate an array of 5 elements
        PUSH 5       ; size of array
        ALLOC        ; allocate memory, address is on stack
        STORE 2      ; store array address in register 2
        
        ; Initialize array with values 10, 20, 30, 40, 50
        LOAD 2       ; load array address
        PUSH 0       ; offset 0
        PUSH 10      ; value 10
        STORE_HEAP   ; store value at array[0]
        
        LOAD 2       ; load array address
        PUSH 1       ; offset 1
        PUSH 20      ; value 20
        STORE_HEAP   ; store value at array[1]
        
        LOAD 2       ; load array address
        PUSH 2       ; offset 2
        PUSH 30      ; value 30
        STORE_HEAP   ; store value at array[2]
        
        LOAD 2       ; load array address
        PUSH 3       ; offset 3
        PUSH 40      ; value 40
        STORE_HEAP   ; store value at array[3]
        
        LOAD 2       ; load array address
        PUSH 4       ; offset 4
        PUSH 50      ; value 50
        STORE_HEAP   ; store value at array[4]
        
        ; Sum all elements in the array
        PUSH 0       ; sum = 0
        STORE 3      ; store sum in register 3
        
        PUSH 0       ; i = 0
        STORE 4      ; store i in register 4
        
        ; Loop start
        LOAD 4       ; load i
        PUSH 5       ; array size
        CMP          ; compare i with array size
        JZ 42        ; if i == array size, jump to end (instruction 42)
        
        ; Load array[i]
        LOAD 2       ; load array address
        LOAD 4       ; load i (offset)
        LOAD_HEAP    ; load array[i]
        
        ; Add to sum
        LOAD 3       ; load sum
        ADD          ; sum += array[i]
        STORE 3      ; store updated sum
        
        ; Increment i
        LOAD 4       ; load i
        PUSH 1       ; push 1
        ADD          ; i++
        STORE 4      ; store updated i
        
        JMP 27       ; jump back to loop start (instruction 27)
        
        ; End
        LOAD 3       ; load sum
        PRINT        ; print sum
        
        ; Free the array memory
        LOAD 2       ; load array address
        FREE         ; free the memory
        
        HALT         ; stop execution
    )";
    
    vm.loadProgram(VirtualMachine::assembleProgram(heapProgram));
    std::cout << "Heap program loaded with " << vm.getProgramSize() << " instructions" << std::endl;

    vm.execute();
    
    return 0;
} 