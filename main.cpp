#include "vm.h"
#include "assembler.h"
#include "mlir_integration.h"
#include <iostream>
#include <memory>
#include <functional>

// Function prototypes for each test case
void runDirectInstructionExample(VirtualMachine& vm);
void runFactorialExample(VirtualMachine& vm, Assembler& assembler);
void runHeapMemoryExample(VirtualMachine& vm, Assembler& assembler);
void runMLIRSimpleExample(VirtualMachine& vm);
void runMLIRLoopExample(VirtualMachine& vm);
void runMLIRComplexExample(VirtualMachine& vm);
void runOptimizationExample(VirtualMachine& vm);

int main() {
    VirtualMachine vm;
    Assembler assembler;
    
    // Run each example
    // runDirectInstructionExample(vm);
    // runFactorialExample(vm, assembler);
    // runHeapMemoryExample(vm, assembler);
    
    // First check if MLIR installation is available
    bool mlirAvailable = MLIRIntegration::checkMLIRInstallation();
    std::cout << "\nMLIR available: " << (mlirAvailable ? "Yes" : "No") << std::endl;
    
    if (mlirAvailable) {
        runMLIRSimpleExample(vm);
        runMLIRLoopExample(vm);
        runMLIRComplexExample(vm);
        runOptimizationExample(vm);
    }
    else {
        std::cout << "MLIR integration not available, skipping MLIR examples" << std::endl;
    }
    
    return 0;
}

// Example 1: Directly adding instructions to the VM
void runDirectInstructionExample(VirtualMachine& vm) {
    std::cout << "Example 1: Calculate 5 + 3 * 2" << std::endl;
    
    // Reset VM state
    vm.reset();
    
    // Add instructions directly
    vm.addInstruction(OpCode::PUSH, 5);  // Push 5 onto stack
    vm.addInstruction(OpCode::PUSH, 3);  // Push 3 onto stack
    vm.addInstruction(OpCode::PUSH, 2);  // Push 2 onto stack
    vm.addInstruction(OpCode::MUL);      // Multiply 3 * 2
    vm.addInstruction(OpCode::ADD);      // Add 5 + (3 * 2)
    vm.addInstruction(OpCode::PRINT);    // Print result
    vm.addInstruction(OpCode::HALT);     // Stop execution
    
    std::cout << "Program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
    vm.execute();
}

// Example 2: Using the assembler to create a factorial program
void runFactorialExample(VirtualMachine& vm, Assembler& assembler) {
    std::cout << "\nExample 2: Calculate factorial of 5" << std::endl;
    
    // Reset VM state
    vm.reset();
    
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
    
    vm.loadProgram(assembler.assemble(factorialProgram));
    std::cout << "Program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
    vm.execute();
}

// Example 3: Heap memory management example
void runHeapMemoryExample(VirtualMachine& vm, Assembler& assembler) {
    std::cout << "\nExample 3: Using heap memory to create and manipulate an array" << std::endl;
    
    // Reset VM state
    vm.reset();
    
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
    
    vm.loadProgram(assembler.assemble(heapProgram));
    std::cout << "Program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
    vm.execute();
}

// Example 4: Simple MLIR example
void runMLIRSimpleExample(VirtualMachine& vm) {
    std::cout << "\nExample 4: Using the MLIR integration for a simple program" << std::endl;
    
    // Reset VM state
    vm.reset();
    
    // Define source code for MLIR compilation
    // Note: The language syntax is "let identifier expression" (no equals sign)
    std::string highLevelProgram = "let x 5; print x;";
    
    std::cout << "Source code:\n" << highLevelProgram << std::endl;
    
    try {
        // Create an MLIR integration instance
        MLIRIntegration mlirIntegration;
        
        // Compile using MLIR
        std::cout << "[runMLIRSimpleExample] before compile\n";
        std::vector<Instruction> mlirCompiledProgram = mlirIntegration.compile(highLevelProgram);
        std::cout << "[runMLIRSimpleExample] after compile\n";
        
        // Show MLIR debug information
        mlirIntegration.dumpMLIR();
        
        // Load and execute the compiled program
        vm.loadProgram(mlirCompiledProgram);
        std::cout << "MLIR compiled program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
        vm.execute();
    } 
    catch (const std::exception& e) {
        std::cerr << "Error in MLIR simple example: " << e.what() << std::endl;
    }
}

// Example 5: MLIR with while loop
void runMLIRLoopExample(VirtualMachine& vm) {
    std::cout << "\nExample 5: Using MLIR with a while loop" << std::endl;
    
    // Reset VM state
    vm.reset();
    
    // Define a program with a while loop (direct assignment syntax, no equals sign)
    std::string loopProgram = "let x 5;\n"
                              "while x do\n"
                              "  print x;\n"
                              "  let x x - 1;\n"
                              "end;";
    
    std::cout << "Source code:\n" << loopProgram << std::endl;
    
    try {
        // Create an MLIR integration instance
        MLIRIntegration mlirIntegration;
        
        // Compile using MLIR
        std::vector<Instruction> mlirCompiledProgram = mlirIntegration.compile(loopProgram);
        
        // Show MLIR debug information
        mlirIntegration.dumpMLIR();
        
        // Load and execute the compiled program
        vm.loadProgram(mlirCompiledProgram);
        std::cout << "MLIR compiled program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
        vm.execute();
    }
    catch (const std::exception& e) {
        std::cerr << "Error in MLIR loop example: " << e.what() << std::endl;
    }
}

// Example 6: Complex MLIR example with conditionals
void runMLIRComplexExample(VirtualMachine& vm) {
    std::cout << "\nExample 6: More complex MLIR example with conditionals" << std::endl;
    
    // Reset VM state
    vm.reset();
    
    try {
        // Define a more complex program for MLIR compilation
        // Note: Direct assignment syntax (no equals signs)
        std::string complexProgram = 
            "let a 10;\n"
            "let b 20;\n"
            "let c a + b;\n"
            "print c;\n"
            "if c then\n"
            "  print 100;\n"
            "end;";
            
        std::cout << "Complex MLIR Source code:\n" << complexProgram << std::endl;
        
        // Create an MLIR integration instance
        MLIRIntegration mlirIntegration;
        
        // Compile using MLIR
        std::vector<Instruction> mlirCompiledProgram = mlirIntegration.compile(complexProgram);
        
        // Show MLIR debug information
        mlirIntegration.dumpMLIR();
        
        // Execute the program
        vm.loadProgram(mlirCompiledProgram);
        std::cout << "Complex MLIR compiled program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
        vm.execute();
    }
    catch (const std::exception& e) {
        std::cerr << "Error using MLIR compiler for complex program: " << e.what() << std::endl;
    }
}

// Example 7: Optimization demonstration
void runOptimizationExample(VirtualMachine& vm) {
    std::cout << "\nExample 7: Demonstrating optimizer capabilities" << std::endl;
    
    // Reset VM state
    vm.reset();
    
    // Define code with optimization opportunities
    std::string optimizableCode = 
        "let a 10;\n"
        "let b 20;\n"
        "let c a + b;\n"      // Can be constant-folded to 30
        "let d 5 * 4;\n"      // Can be constant-folded to 20
        "let zero 0;\n"
        "let x c + zero;\n"   // Addition with 0 can be eliminated
        "let y d * 1;\n"      // Multiplication by 1 can be eliminated
        "print x;\n"          // Should print 30
        "print y;\n";         // Should print 20
    
    std::cout << "Source code with optimization opportunities:\n" << optimizableCode << std::endl;
    
    try {
        // Set optimization flag to true (will be passed through to MLIRCompiler)
        bool enableOptimizations = true;
        
        // Create an MLIR integration instance with optimizations enabled
        MLIRIntegration mlirIntegration;
        
        // Compile using MLIR
        std::cout << "[runOptimizationExample] Compiling with optimizations..." << std::endl;
        std::vector<Instruction> optimizedProgram = mlirIntegration.compile(optimizableCode);
        
        // Show MLIR debug information
        mlirIntegration.dumpMLIR();
        
        // Load and execute the compiled program
        vm.loadProgram(optimizedProgram);
        std::cout << "Program loaded with " << vm.getProgramSize() << " instructions" << std::endl;
        vm.execute();
        
        // Optionally show the actual instructions
        std::cout << "\nGenerated VM instructions:" << std::endl;
        for (size_t i = 0; i < optimizedProgram.size(); i++) {
            std::cout << i << ": " << opcodeToString(optimizedProgram[i].opcode) 
                      << " " << optimizedProgram[i].operand << std::endl;
        }
    } 
    catch (const std::exception& e) {
        std::cerr << "Error in optimization example: " << e.what() << std::endl;
    }
} 