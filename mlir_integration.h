#ifndef MLIR_INTEGRATION_H
#define MLIR_INTEGRATION_H

#include "vm.h"
#include "mlir_compiler.h"
#include <string>
#include <vector>
#include <memory>

/**
 * Integration class for the MLIR compiler
 * This class serves as the interface between the VM and the MLIR compiler
 */
class MLIRIntegration {
private:
    // Use the real MLIR compiler implementation
    std::unique_ptr<MLIRCompiler> compiler;
    
public:
    MLIRIntegration();
    ~MLIRIntegration();
    
    // Compile source code to VM instructions
    // useMLIRPath: when true, use MLIR optimization passes; when false, use our custom optimizer
    // enableOptimizations: when true, apply optimizations (via either path); when false, skip optimizations
    std::vector<Instruction> compile(const std::string& source, bool useMLIRPath = false, bool enableOptimizations = true);
    
    // Print MLIR module for debugging
    void dumpMLIR();
    
    // Check if LLVM/MLIR is properly installed
    static bool checkMLIRInstallation();
};

#endif // MLIR_INTEGRATION_H 