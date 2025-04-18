#include "vm.h"
#include <iostream>
#include <sstream>
#include <algorithm>

VirtualMachine::VirtualMachine() {
    reset();
}

void VirtualMachine::reset() {
    pc = 0;
    running = false;
    cmpFlag = 0;
    stack.clear();
    registers.clear();
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
    
    std::cout << "Starting execution with " << program.size() << " instructions" << std::endl;
    
    while (running && pc < program.size()) {
        Instruction& instr = program[pc];
        
        // std::cout << "Executing instruction at PC=" << pc 
        //           << ", OpCode=" << static_cast<int>(static_cast<uint8_t>(instr.opcode))
        //           << ", Operand=" << instr.operand << std::endl;
        
        switch (instr.opcode) {
            case OpCode::HALT:
                running = false;
                break;
                
            case OpCode::PUSH:
                stack.push_back(instr.operand);
                pc++;
                break;
                
            case OpCode::POP:
                if (!stack.empty()) {
                    stack.pop_back();
                } else {
                    std::cerr << "Error: Stack underflow on POP" << std::endl;
                    running = false;
                }
                pc++;
                break;
                
            case OpCode::ADD: {
                if (stack.size() < 2) {
                    std::cerr << "Error: Stack underflow on ADD" << std::endl;
                    running = false;
                    break;
                }
                int32_t b = stack.back(); stack.pop_back();
                int32_t a = stack.back(); stack.pop_back();
                stack.push_back(a + b);
                pc++;
                break;
            }
                
            case OpCode::SUB: {
                if (stack.size() < 2) {
                    std::cerr << "Error: Stack underflow on SUB" << std::endl;
                    running = false;
                    break;
                }
                int32_t b = stack.back(); stack.pop_back();
                int32_t a = stack.back(); stack.pop_back();
                stack.push_back(a - b);
                pc++;
                break;
            }
                
            case OpCode::MUL: {
                if (stack.size() < 2) {
                    std::cerr << "Error: Stack underflow on MUL" << std::endl;
                    running = false;
                    break;
                }
                int32_t b = stack.back(); stack.pop_back();
                int32_t a = stack.back(); stack.pop_back();
                stack.push_back(a * b);
                pc++;
                break;
            }
                
            case OpCode::DIV: {
                if (stack.size() < 2) {
                    std::cerr << "Error: Stack underflow on DIV" << std::endl;
                    running = false;
                    break;
                }
                int32_t b = stack.back(); stack.pop_back();
                int32_t a = stack.back(); stack.pop_back();
                if (b == 0) {
                    std::cerr << "Error: Division by zero" << std::endl;
                    running = false;
                    break;
                }
                stack.push_back(a / b);
                pc++;
                break;
            }
                
            case OpCode::PRINT:
                if (!stack.empty()) {
                    std::cout << stack.back() << std::endl;
                } else {
                    std::cerr << "Error: Stack underflow on PRINT" << std::endl;
                    running = false;
                }
                pc++;
                break;
                
            case OpCode::STORE: {
                if (stack.empty()) {
                    std::cerr << "Error: Stack underflow on STORE" << std::endl;
                    running = false;
                    break;
                }
                int32_t value = stack.back(); stack.pop_back();
                registers[instr.operand] = value;
                pc++;
                break;
            }
                
            case OpCode::LOAD:
                stack.push_back(registers[instr.operand]);
                pc++;
                break;
                
            case OpCode::JMP:
                pc = instr.operand;
                break;
                
            case OpCode::JZ:
                if (cmpFlag == 0) {
                    pc = instr.operand;
                } else {
                    pc++;
                }
                break;
                
            case OpCode::JNZ:
                if (cmpFlag != 0) {
                    pc = instr.operand;
                } else {
                    pc++;
                }
                break;
                
            case OpCode::CMP: {
                if (stack.size() < 2) {
                    std::cerr << "Error: Stack underflow on CMP" << std::endl;
                    running = false;
                    break;
                }
                int32_t b = stack.back(); stack.pop_back();
                int32_t a = stack.back(); stack.pop_back();
                
                if (a < b) cmpFlag = -1;
                else if (a > b) cmpFlag = 1;
                else cmpFlag = 0;
                
                pc++;
                break;
            }
                
            default:
                std::cerr << "Error: Unknown opcode: " << static_cast<int>(static_cast<uint8_t>(instr.opcode)) << std::endl;
                running = false;
                break;
        }
    }
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
    std::cout << "================" << std::endl;
}

std::vector<Instruction> VirtualMachine::assembleProgram(const std::string& code) {
    std::vector<Instruction> result;
    std::istringstream stream(code);
    std::string line;
    
    // Simple mapping from instruction names to opcodes
    std::unordered_map<std::string, OpCode> opcodeMap = {
        {"HALT", OpCode::HALT},
        {"PUSH", OpCode::PUSH},
        {"POP", OpCode::POP},
        {"ADD", OpCode::ADD},
        {"SUB", OpCode::SUB},
        {"MUL", OpCode::MUL},
        {"DIV", OpCode::DIV},
        {"PRINT", OpCode::PRINT},
        {"STORE", OpCode::STORE},
        {"LOAD", OpCode::LOAD},
        {"JMP", OpCode::JMP},
        {"JZ", OpCode::JZ},
        {"JNZ", OpCode::JNZ},
        {"CMP", OpCode::CMP}
    };
    
    while (std::getline(stream, line)) {
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Trim leading whitespace
        size_t firstNonSpace = line.find_first_not_of(" \t");
        if (firstNonSpace == std::string::npos) {
            continue; // Line is all whitespace
        }
        line = line.substr(firstNonSpace);
        
        // Skip comment lines
        if (line[0] == ';' || line[0] == '#') {
            continue;
        }
        
        // Remove inline comments
        size_t commentPos = line.find(';');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        
        // Trim trailing whitespace
        size_t lastNonSpace = line.find_last_not_of(" \t");
        if (lastNonSpace != std::string::npos) {
            line = line.substr(0, lastNonSpace + 1);
        }
        
        // Skip if line is now empty after removing comments
        if (line.empty()) {
            continue;
        }
        
        std::istringstream lineStream(line);
        std::string opcodeName;
        lineStream >> opcodeName;
        
        // Convert to uppercase for case-insensitive comparison
        std::transform(opcodeName.begin(), opcodeName.end(), opcodeName.begin(), ::toupper);
        
        if (opcodeMap.find(opcodeName) == opcodeMap.end()) {
            std::cerr << "Unknown instruction: " << opcodeName << std::endl;
            continue;
        }
        
        OpCode opcode = opcodeMap[opcodeName];
        int32_t operand = 0;
        
        // Instructions that need an operand
        if (opcode == OpCode::PUSH || opcode == OpCode::STORE || opcode == OpCode::LOAD ||
            opcode == OpCode::JMP || opcode == OpCode::JZ || opcode == OpCode::JNZ) {
            lineStream >> operand;
        }
        
        result.push_back({opcode, operand});
    }
    
    return result;
} 