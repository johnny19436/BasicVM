#include "mlir_integration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <iostream>
#include <fstream>
#include <system_error>
#include <llvm/Support/InitLLVM.h>

MLIRIntegration::MLIRIntegration()
  : compiler(std::make_unique<MLIRCompiler>()) {
}

MLIRIntegration::~MLIRIntegration() = default;

std::vector<Instruction> MLIRIntegration::compile(const std::string &source) {
    std::cerr << "[MLIRIntegration::compile] → entering\n";
    
    std::vector<Instruction> program;
    try {
        program = compiler->compile(source);
        std::cerr << "[MLIRIntegration::compile] ← returned successfully, size=" << program.size() << "\n";
    } 
    catch (const std::exception &e) {
        std::cerr << "Error during MLIR compilation: " << e.what() << std::endl;
        program = {
            { OpCode::PUSH,  8888 },  // integration error marker
            { OpCode::PRINT,    0 },
            { OpCode::HALT,     0 }
        };
    }
    catch (...) {
        std::cerr << "Unknown error during MLIR compilation" << std::endl;
        program = {
            { OpCode::PUSH,  8887 },  // integration unknown error marker
            { OpCode::PRINT,    0 },
            { OpCode::HALT,     0 }
        };
    }

    // Make sure we always end with a HALT
    if (program.empty() || program.back().opcode != OpCode::HALT) {
        program.push_back({ OpCode::HALT, 0 });
    }
    
    std::cerr << "[MLIRIntegration::compile] ← final program size=" << program.size() << "\n";
    return program;
}

void MLIRIntegration::dumpMLIR() {
    std::cout << "MLIR Module Dump:" << std::endl;
    compiler->dumpMLIR();
}

bool MLIRIntegration::checkMLIRInstallation() {
    try {
        // Minimal check: can we load the dialects we need?
        mlir::MLIRContext ctx;
        mlir::DialectRegistry registry;
        registry.insert<
          mlir::func::FuncDialect,
          mlir::arith::ArithDialect,
          mlir::cf::ControlFlowDialect,
          mlir::basicvm::BasicVMDialect
        >();
        ctx.appendDialectRegistry(registry);
        std::cout << "MLIR installation check successful" << std::endl;
        return true;
    } 
    catch (const std::exception &e) {
        std::cerr << "MLIR installation check failed: " << e.what() << std::endl;
        return false;
    }
}
