#ifndef VM_H
#define VM_H

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>

// Opcodes for our virtual machine
enum class OpCode : uint8_t {
    HALT,       // Stop execution
    PUSH,       // Push value onto stack
    POP,        // Pop value from stack
    ADD,        // Add top two values on stack
    SUB,        // Subtract top two values on stack
    MUL,        // Multiply top two values on stack
    DIV,        // Divide top two values on stack
    PRINT,      // Print top value on stack
    STORE,      // Store value in register
    LOAD,       // Load value from register
    JMP,        // Jump to address
    JZ,         // Jump if zero
    JNZ,        // Jump if not zero
    CMP,        // Compare top two values on stack
};

// Instruction structure
struct Instruction {
    OpCode opcode;
    int32_t operand;
};

class VirtualMachine {
private:
    std::vector<Instruction> program;
    std::vector<int32_t> stack;
    std::unordered_map<int32_t, int32_t> registers;
    uint32_t pc;  // Program counter
    bool running;
    int32_t cmpFlag;  // Result of last comparison

public:
    VirtualMachine();
    
    // Load a program into the VM
    void loadProgram(const std::vector<Instruction>& newProgram);
    
    // Add a single instruction to the program
    void addInstruction(OpCode opcode, int32_t operand = 0);
    
    // Execute the loaded program
    void execute();
    
    // Reset the VM state
    void reset();
    
    // Debugging helper to print the current state
    void dumpState() const;
    
    // Helper to create a program from a string of assembly-like code
    static std::vector<Instruction> assembleProgram(const std::string& code);
    
    // Add to public section of VirtualMachine class
    size_t getProgramSize() const { return program.size(); }
};

#endif // VM_H 