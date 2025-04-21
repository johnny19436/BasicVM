#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include "vm.h"
#include <string>
#include <vector>
#include <unordered_map>

class Assembler {
private:
    // Map of instruction names to opcodes
    std::unordered_map<std::string, OpCode> opcodeMap;
    
    // Initialize the opcode map
    void initOpcodeMap();
    
    // Parse a single line of assembly code
    bool parseLine(const std::string& line, std::vector<Instruction>& result);

public:
    Assembler();
    
    // Assemble a string of assembly code into VM instructions
    std::vector<Instruction> assemble(const std::string& code);
    
    // Add a new instruction to the assembler
    void addInstruction(const std::string& name, OpCode opcode);
    
    // Get the opcode for an instruction name
    OpCode getOpcode(const std::string& name) const;
    
    // Check if an instruction name is valid
    bool isValidInstruction(const std::string& name) const;
};

#endif // ASSEMBLER_H 