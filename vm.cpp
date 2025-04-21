#include "vm.h"
#include <iostream>
#include <sstream>
#include <algorithm>

VirtualMachine::VirtualMachine() {
    initInstructionHandlers();
    reset();
}

void VirtualMachine::reset() {
    pc = 0;
    running = false;
    cmpFlag = 0;
    stack.clear();
    registers.clear();
    heap.clear();
    nextHeapAddress = 1; // Start at 1, 0 can be used as null
}

void VirtualMachine::loadProgram(const std::vector<Instruction>& newProgram) {
    program = newProgram;
    reset();
}

void VirtualMachine::addInstruction(OpCode opcode, int32_t operand) {
    program.push_back({opcode, operand});
}

void VirtualMachine::execute() {
    if (program.empty()) {
        std::cerr << "No program loaded!" << std::endl;
        return;
    }
    
    pc = 0;
    running = true;
    
    while (running && pc < program.size()) {
        const Instruction& instr = program[pc];
        
        auto handlerIt = instructionHandlers.find(instr.opcode);
        if (handlerIt != instructionHandlers.end()) {
            handlerIt->second(*this, instr);
        } else {
            std::cerr << "Error: Unknown opcode: " << static_cast<int>(static_cast<uint8_t>(instr.opcode)) << std::endl;
            running = false;
        }
    }
}

// Stack operations
void VirtualMachine::pushStack(int32_t value) {
    stack.push_back(value);
}

int32_t VirtualMachine::popStack() {
    if (stack.empty()) {
        std::cerr << "Error: Stack underflow on pop" << std::endl;
        return 0;
    }
    int32_t value = stack.back();
    stack.pop_back();
    return value;
}

int32_t VirtualMachine::peekStack() const {
    if (stack.empty()) {
        std::cerr << "Error: Stack underflow on peek" << std::endl;
        return 0;
    }
    return stack.back();
}

bool VirtualMachine::stackEmpty() const {
    return stack.empty();
}

size_t VirtualMachine::stackSize() const {
    return stack.size();
}

// Register operations
void VirtualMachine::storeRegister(int32_t reg, int32_t value) {
    registers[reg] = value;
}

int32_t VirtualMachine::loadRegister(int32_t reg) const {
    auto it = registers.find(reg);
    if (it != registers.end()) {
        return it->second;
    }
    return 0; // Default value for non-existent registers
}

// Program counter operations
void VirtualMachine::incrementPC() {
    pc++;
}

void VirtualMachine::jumpPC(uint32_t address) {
    pc = address;
}

uint32_t VirtualMachine::getPC() const {
    return pc;
}

// Comparison flag operations
void VirtualMachine::setCmpFlag(int32_t value) {
    cmpFlag = value;
}

int32_t VirtualMachine::getCmpFlag() const {
    return cmpFlag;
}

// Running state operations
void VirtualMachine::setRunning(bool state) {
    running = state;
}

bool VirtualMachine::isRunning() const {
    return running;
}

const Instruction& VirtualMachine::getCurrentInstruction() const {
    return program[pc];
}

void VirtualMachine::addInstructionHandler(OpCode opcode, InstructionHandler handler) {
    instructionHandlers[opcode] = handler;
}

// Initialize all instruction handlers
void VirtualMachine::initInstructionHandlers() {
    // HALT
    instructionHandlers[OpCode::HALT] = [](VirtualMachine& vm, const Instruction&) {
        vm.setRunning(false);
    };
    
    // PUSH
    instructionHandlers[OpCode::PUSH] = [](VirtualMachine& vm, const Instruction& instr) {
        vm.pushStack(instr.operand);
        vm.incrementPC();
    };
    
    // POP
    instructionHandlers[OpCode::POP] = [](VirtualMachine& vm, const Instruction&) {
        if (!vm.stackEmpty()) {
            vm.popStack();
        } else {
            std::cerr << "Error: Stack underflow on POP" << std::endl;
            vm.setRunning(false);
        }
        vm.incrementPC();
    };
    
    // ADD
    instructionHandlers[OpCode::ADD] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackSize() < 2) {
            std::cerr << "Error: Stack underflow on ADD" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t b = vm.popStack();
        int32_t a = vm.popStack();
        vm.pushStack(a + b);
        vm.incrementPC();
    };
    
    // SUB
    instructionHandlers[OpCode::SUB] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackSize() < 2) {
            std::cerr << "Error: Stack underflow on SUB" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t b = vm.popStack();
        int32_t a = vm.popStack();
        vm.pushStack(a - b);
        vm.incrementPC();
    };
    
    // MUL
    instructionHandlers[OpCode::MUL] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackSize() < 2) {
            std::cerr << "Error: Stack underflow on MUL" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t b = vm.popStack();
        int32_t a = vm.popStack();
        vm.pushStack(a * b);
        vm.incrementPC();
    };
    
    // DIV
    instructionHandlers[OpCode::DIV] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackSize() < 2) {
            std::cerr << "Error: Stack underflow on DIV" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t b = vm.popStack();
        int32_t a = vm.popStack();
        if (b == 0) {
            std::cerr << "Error: Division by zero" << std::endl;
            vm.setRunning(false);
            return;
        }
        vm.pushStack(a / b);
        vm.incrementPC();
    };
    
    // PRINT
    instructionHandlers[OpCode::PRINT] = [](VirtualMachine& vm, const Instruction&) {
        if (!vm.stackEmpty()) {
            std::cout << vm.peekStack() << std::endl;
        } else {
            std::cerr << "Error: Stack underflow on PRINT" << std::endl;
            vm.setRunning(false);
        }
        vm.incrementPC();
    };
    
    // STORE
    instructionHandlers[OpCode::STORE] = [](VirtualMachine& vm, const Instruction& instr) {
        if (vm.stackEmpty()) {
            std::cerr << "Error: Stack underflow on STORE" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t value = vm.popStack();
        vm.storeRegister(instr.operand, value);
        vm.incrementPC();
    };
    
    // LOAD
    instructionHandlers[OpCode::LOAD] = [](VirtualMachine& vm, const Instruction& instr) {
        int32_t value = vm.loadRegister(instr.operand);
        vm.pushStack(value);
        vm.incrementPC();
    };
    
    // JMP
    instructionHandlers[OpCode::JMP] = [](VirtualMachine& vm, const Instruction& instr) {
        vm.jumpPC(instr.operand);
    };
    
    // JZ
    instructionHandlers[OpCode::JZ] = [](VirtualMachine& vm, const Instruction& instr) {
        // std::cout << "JZ: cmpFlag=" << vm.getCmpFlag() << ", target=" << instr.operand << std::endl;
        if (vm.getCmpFlag() == 0) {
            vm.jumpPC(instr.operand);
        } else {
            vm.incrementPC();
        }
    };
    
    // JNZ
    instructionHandlers[OpCode::JNZ] = [](VirtualMachine& vm, const Instruction& instr) {
        if (vm.getCmpFlag() != 0) {
            vm.jumpPC(instr.operand);
        } else {
            vm.incrementPC();
        }
    };
    
    // CMP
    instructionHandlers[OpCode::CMP] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackSize() < 2) {
            std::cerr << "Error: Stack underflow on CMP" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t b = vm.popStack();
        int32_t a = vm.popStack();
        vm.setCmpFlag(a - b);
        // std::cout << "CMP: " << a << " - " << b << " = " << vm.getCmpFlag() << std::endl;
        vm.incrementPC();
    };
    
    // ALLOC
    instructionHandlers[OpCode::ALLOC] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackEmpty()) {
            std::cerr << "Error: Stack underflow on ALLOC" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t size = vm.popStack();
        int32_t address = vm.allocateHeapMemory(size);
        vm.pushStack(address);
        vm.incrementPC();
    };
    
    // FREE
    instructionHandlers[OpCode::FREE] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackEmpty()) {
            std::cerr << "Error: Stack underflow on FREE" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t address = vm.popStack();
        vm.freeHeapMemory(address);
        vm.incrementPC();
    };
    
    // STORE_HEAP
    instructionHandlers[OpCode::STORE_HEAP] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackSize() < 3) {
            std::cerr << "Error: Stack underflow on STORE_HEAP" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t value = vm.popStack();
        int32_t offset = vm.popStack();
        int32_t address = vm.popStack();
        vm.storeHeapValue(address, offset, value);
        vm.incrementPC();
    };
    
    // LOAD_HEAP
    instructionHandlers[OpCode::LOAD_HEAP] = [](VirtualMachine& vm, const Instruction&) {
        if (vm.stackSize() < 2) {
            std::cerr << "Error: Stack underflow on LOAD_HEAP" << std::endl;
            vm.setRunning(false);
            return;
        }
        int32_t offset = vm.popStack();
        int32_t address = vm.popStack();
        int32_t value = vm.loadHeapValue(address, offset);
        vm.pushStack(value);
        vm.incrementPC();
    };
}

void VirtualMachine::dumpState() const {
    std::cout << "=== VM State ===" << std::endl;
    std::cout << "PC: " << pc << std::endl;
    std::cout << "Running: " << (running ? "true" : "false") << std::endl;
    std::cout << "Compare Flag: " << cmpFlag << std::endl;
    
    std::cout << "Stack (" << stack.size() << " elements):" << std::endl;
    for (size_t i = 0; i < stack.size(); i++) {
        std::cout << i << ": " << stack[i] << std::endl;
    }
    
    std::cout << "Registers:" << std::endl;
    for (const auto& reg : registers) {
        std::cout << "R" << reg.first << ": " << reg.second << std::endl;
    }
    
    std::cout << "Heap (" << heap.size() << " blocks):" << std::endl;
    for (const auto& block : heap) {
        std::cout << "Address " << block.first << " (size " << block.second.size() << "):" << std::endl;
        for (size_t i = 0; i < block.second.size(); i++) {
            std::cout << "  [" << i << "]: " << block.second[i] << std::endl;
        }
    }
    
    std::cout << "================" << std::endl;
}

int32_t VirtualMachine::allocateHeapMemory(int32_t size) {
    if (size <= 0) {
        std::cerr << "Error: Invalid heap allocation size" << std::endl;
        return 0;
    }
    
    int32_t address = nextHeapAddress++;
    heap[address] = std::vector<int32_t>(size, 0); // Initialize with zeros
    return address;
}

void VirtualMachine::freeHeapMemory(int32_t address) {
    if (heap.find(address) != heap.end()) {
        heap.erase(address);
    } else {
        std::cerr << "Error: Attempt to free invalid heap address: " << address << std::endl;
    }
}

void VirtualMachine::storeHeapValue(int32_t address, int32_t offset, int32_t value) {
    if (heap.find(address) == heap.end()) {
        std::cerr << "Error: Invalid heap address: " << address << std::endl;
        return;
    }
    
    std::vector<int32_t>& block = heap[address];
    if (offset < 0 || offset >= static_cast<int32_t>(block.size())) {
        std::cerr << "Error: Heap access out of bounds: address=" << address 
                  << ", offset=" << offset << ", size=" << block.size() << std::endl;
        return;
    }
    
    block[offset] = value;
}

int32_t VirtualMachine::loadHeapValue(int32_t address, int32_t offset) {
    if (heap.find(address) == heap.end()) {
        std::cerr << "Error: Invalid heap address: " << address << std::endl;
        return 0;
    }
    
    std::vector<int32_t>& block = heap[address];
    if (offset < 0 || offset >= static_cast<int32_t>(block.size())) {
        std::cerr << "Error: Heap access out of bounds: address=" << address 
                  << ", offset=" << offset << ", size=" << block.size() << std::endl;
        return 0;
    }
    
    return block[offset];
}

OpCode VirtualMachine::getInstructionOpcode(size_t index) const {
    if (index < program.size()) {
        return program[index].opcode;
    }
    return OpCode::HALT;
}

int32_t VirtualMachine::getInstructionOperand(size_t index) const {
    if (index < program.size()) {
        return program[index].operand;
    }
    return 0;
} 