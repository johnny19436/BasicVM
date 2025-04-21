#include "assembler.h"
#include <sstream>
#include <iostream>
#include <algorithm>

Assembler::Assembler() {
    initOpcodeMap();
}

void Assembler::initOpcodeMap() {
    opcodeMap = {
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
        {"CMP", OpCode::CMP},
        {"ALLOC", OpCode::ALLOC},
        {"FREE", OpCode::FREE},
        {"STORE_HEAP", OpCode::STORE_HEAP},
        {"LOAD_HEAP", OpCode::LOAD_HEAP}
    };
}

bool Assembler::parseLine(const std::string& line, std::vector<Instruction>& result) {
    // Skip empty lines
    if (line.empty()) {
        return true;
    }
    
    // Trim leading whitespace
    size_t firstNonSpace = line.find_first_not_of(" \t");
    if (firstNonSpace == std::string::npos) {
        return true; // Line is all whitespace
    }
    std::string trimmedLine = line.substr(firstNonSpace);
    
    // Skip comment lines
    if (trimmedLine[0] == ';' || trimmedLine[0] == '#') {
        return true;
    }
    
    // Remove inline comments
    size_t commentPos = trimmedLine.find(';');
    if (commentPos != std::string::npos) {
        trimmedLine = trimmedLine.substr(0, commentPos);
    }
    
    // Trim trailing whitespace
    size_t lastNonSpace = trimmedLine.find_last_not_of(" \t");
    if (lastNonSpace != std::string::npos) {
        trimmedLine = trimmedLine.substr(0, lastNonSpace + 1);
    }
    
    // Skip if line is now empty after removing comments
    if (trimmedLine.empty()) {
        return true;
    }
    
    std::istringstream lineStream(trimmedLine);
    std::string opcodeName;
    lineStream >> opcodeName;
    
    // Convert to uppercase for case-insensitive comparison
    std::transform(opcodeName.begin(), opcodeName.end(), opcodeName.begin(), ::toupper);
    
    if (!isValidInstruction(opcodeName)) {
        std::cerr << "Unknown instruction: " << opcodeName << std::endl;
        return false;
    }
    
    OpCode opcode = opcodeMap[opcodeName];
    int32_t operand = 0;
    
    // Instructions that need an operand
    if (opcode == OpCode::PUSH || opcode == OpCode::STORE || opcode == OpCode::LOAD ||
        opcode == OpCode::JMP || opcode == OpCode::JZ || opcode == OpCode::JNZ) {
        lineStream >> operand;
    }
    
    result.push_back({opcode, operand});
    return true;
}

std::vector<Instruction> Assembler::assemble(const std::string& code) {
    std::vector<Instruction> result;
    std::istringstream stream(code);
    std::string line;
    
    while (std::getline(stream, line)) {
        if (!parseLine(line, result)) {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }
    
    return result;
}

void Assembler::addInstruction(const std::string& name, OpCode opcode) {
    opcodeMap[name] = opcode;
}

OpCode Assembler::getOpcode(const std::string& name) const {
    auto it = opcodeMap.find(name);
    if (it != opcodeMap.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown instruction: " + name);
}

bool Assembler::isValidInstruction(const std::string& name) const {
    return opcodeMap.find(name) != opcodeMap.end();
} 