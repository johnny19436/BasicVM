#ifndef VM_H
#define VM_H

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

// Forward declaration
class VirtualMachine;

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
    ALLOC,      // Allocate memory on heap, returns address
    FREE,       // Free memory on heap
    STORE_HEAP, // Store value at heap address
    LOAD_HEAP,  // Load value from heap address
};

// Instruction structure
struct Instruction {
    OpCode opcode;
    int32_t operand;
};

// Instruction handler type
using InstructionHandler = std::function<void(VirtualMachine&, const Instruction&)>;

class VirtualMachine {
private:
    std::vector<Instruction> program;
    std::vector<int32_t> stack;
    std::unordered_map<int32_t, int32_t> registers;
    std::unordered_map<int32_t, std::vector<int32_t>> heap; // Heap memory
    uint32_t nextHeapAddress;                              // Next available heap address
    uint32_t pc;  // Program counter
    bool running;
    int32_t cmpFlag;  // Result of last comparison
    
    // Map of instruction handlers
    std::unordered_map<OpCode, InstructionHandler> instructionHandlers;
    
    // Initialize instruction handlers
    void initInstructionHandlers();

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
    
    // Get program size
    size_t getProgramSize() const { return program.size(); }
    
    // Heap operations
    int32_t allocateHeapMemory(int32_t size);
    void freeHeapMemory(int32_t address);
    void storeHeapValue(int32_t address, int32_t offset, int32_t value);
    int32_t loadHeapValue(int32_t address, int32_t offset);
    
    // Stack operations (for instruction handlers)
    void pushStack(int32_t value);
    int32_t popStack();
    int32_t peekStack() const;
    bool stackEmpty() const;
    size_t stackSize() const;
    
    // Register operations
    void storeRegister(int32_t reg, int32_t value);
    int32_t loadRegister(int32_t reg) const;
    
    // Program counter operations
    void incrementPC();
    void jumpPC(uint32_t address);
    uint32_t getPC() const;
    
    // Comparison flag operations
    void setCmpFlag(int32_t value);
    int32_t getCmpFlag() const;
    
    // Running state operations
    void setRunning(bool state);
    bool isRunning() const;
    
    // Get instruction at current PC
    const Instruction& getCurrentInstruction() const;
    
    // Add a custom instruction handler
    void addInstructionHandler(OpCode opcode, InstructionHandler handler);
    
    // Get instruction opcode and operand (for debugging)
    OpCode getInstructionOpcode(size_t index) const;
    int32_t getInstructionOperand(size_t index) const;
};

#endif // VM_H 